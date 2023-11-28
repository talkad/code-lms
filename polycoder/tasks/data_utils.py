""" Tasks data utilities."""

from itertools import cycle
from functools import partial
import numpy as np
import torch
import re


def unwrap_lists(list_data, data_type=torch.Tensor):
    # Handle possibly nested lists.
    if not isinstance(list_data, list):
        return list_data
    else:
        if isinstance(list_data[0], data_type):
            return list_data
        else:
            return unwrap_lists(list_data[0], data_type)


def broadcast_data_list(keys, data, is_tensor=True, data_type=torch.Tensor):
    from megatron import mpu, print_rank_0
    """Broadcast data from rank zero of each model parallel group to the
        members of the same model parallel group.

        Arguments:
            keys: list of keys in the data dictionary to be broadcasted
            data: data dictionary of string keys and cpu tensor values.
        """
    # Pack on rank zero.
    if mpu.get_model_parallel_rank() == 0:
        data_list = []
        # Convert dict to list
        for k in keys:
            if isinstance(data[k], list):
                # Handle nested lists
                data[k] = unwrap_lists(data[k], data_type)
                if is_tensor:
                    data[k] = [d.squeeze().cuda() for d in data[k]]
            else:
                if is_tensor:
                    data[k] = data[k].squeeze().cuda()
            data_list.append(data[k])
    else:
        data_list = [None] * len(keys)

    # Broadcast
    torch.distributed.broadcast_object_list(data_list,
                                            mpu.get_model_parallel_src_rank(),
                                            group=mpu.get_model_parallel_group())

    output = {k: data for k, data in zip(keys, data_list)}
    return output


def collate_fn_pad(batch, data_keys_to_collate=[],
                   pad_keys=['text', 'input_ids', 'attention_mask'],
                   report_lengths=False):
    from torch.utils.data._utils.collate import default_collate
    elem = batch[0]
    elem_type = type(elem)

    try:
        out_dict = {}
        for key in elem:
            item_list = [d[key] for d in batch]
            # Custom behavior: pad this baby!
            if key in pad_keys:
                item_list = unwrap_lists(item_list)
                # Handle nested list[list[Tensor]]
                padded_item = torch.nn.utils.rnn.pad_sequence(item_list, batch_first=True)
                out_dict.update({key: padded_item})
                if report_lengths and key in ['text', 'input_ids', 'attention_mask']:
                    lengths = torch.IntTensor([sample.size(dim=0) for sample in item_list])
                    out_dict.update({'lengths': lengths})
            elif key in data_keys_to_collate:
                # Default collate behavior for a dictionary, according to pytorch 2.0.0
                out_dict.update({key: default_collate(item_list)})
            else:
                # print(f'VAV DEBUG custom list of lists: {key, item_list}')
                # Custom behavior for fields that are lists of lists
                out_dict.update({key: item_list})
        return elem_type(out_dict)
    except TypeError:
        raise ValueError(f"This mapping type {elem_type} may not support `__init__(iterable)`.")


class SeqLengthSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, bucket_boundaries, shuffle=True):
        super().__init__(data_source)
        if isinstance(data_source['length'], torch.Tensor):
            self.ind_n_len = data_source['length'].numpy()
        else:
            self.ind_n_len = np.array(data_source['length'])
        self.bucket_boundaries = bucket_boundaries
        bin_inds = np.digitize(self.ind_n_len, self.bucket_boundaries)
        outside_bins = np.sum(bin_inds == 0)
        self.data_len = len(data_source) - outside_bins
        if outside_bins > 0:
            print(f'Excluding {outside_bins} samples because they fall outside length bins')
            if outside_bins < 25:
                print(f'Excluded samples have lengths: {self.ind_n_len[bin_inds == 0]}')
            else:
                excluded = self.ind_n_len[bin_inds == 0]
                print(f'Excluded sample lengths at 5%, 50%, and 95%: {np.percentile(excluded, [5, 50, 95])}')
        uniq_bins = np.unique(bin_inds)
        data_buckets = {}
        for b in uniq_bins:
            if b == 0:
                continue
            data_buckets[b] = np.where(bin_inds == b)[0]
        self.data_buckets = data_buckets
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        iter_list = []
        for k in self.data_buckets.keys():
            if self.shuffle:
                np.random.shuffle(self.data_buckets[k])  # this is an in-place operation.
            # VAV: the following line of code will NOT yield perfect batches -- some of the leftovers
            # get grouped into bigger batches!
            iter_list += (np.array_split(self.data_buckets[k],
                                         int(self.data_buckets[k].shape[0] / self.batch_size)))
        # shuffle(iter_list)  # shuffle batches so that they aren't ordered by bucket size
        for i in iter_list:
            yield i.tolist()  # as it was stored in an array

    def __len__(self):
        return self.data_len


def make_data_loader_with_padding(dataset, neox_args, seq_length_bins=None, drop_last=True, shuffle=True):
    """Build dataloader given an input dataset. Minor modification of megatron.data.data_utils"""
    from megatron.data.samplers import DistributedBatchSampler
    from megatron import mpu, print_rank_0
    from megatron.data.data_utils import make_data_loader

    if dataset is None:
        return None
    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = neox_args.batch_size * world_size
    print_rank_0(f"VAV DEBUG global batch {global_batch_size}, each device {neox_args.batch_size}")
    num_workers = neox_args.num_workers

    collate_fn = partial(collate_fn_pad,
                         data_keys_to_collate=neox_args.data_keys_collate,
                         pad_keys=neox_args.pad_data_keys)

    if seq_length_bins:
        s1_batch = global_batch_size
        s2_batch = neox_args.batch_size if (world_size > 1) else 1

    if seq_length_bins:
        sampler = SeqLengthSampler(dataset, batch_size=s1_batch, bucket_boundaries=seq_length_bins,
                                   shuffle=shuffle)
        batch_sampler = DistributedBatchSampler(sampler=sampler,
                                                batch_size=s2_batch,
                                                drop_last=drop_last,
                                                rank=rank,
                                                world_size=world_size)
    else:
        if shuffle:
            print_rank_0("WARNING: You requested data shuffling, but it likely doesn't work with built-in samplers")
        # Use a simple sampler with distributed batch sampler.
        sampler = torch.utils.data.SequentialSampler(dataset)
        batch_sampler = DistributedBatchSampler(sampler=sampler,
                                                batch_size=global_batch_size,
                                                drop_last=drop_last,
                                                rank=rank,
                                                world_size=world_size)

    print_rank_0(f'Dataset has {len(batch_sampler)} samples')
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       sampler=None,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       collate_fn=collate_fn,
                                       pin_memory=True)


def build_dataloaders(neox_args, get_dataset_fn, pad_sequences=False, length_bins=None, drop_last=True, shuffle=True):
    from megatron import mpu, print_rank_0
    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Ensure only the first/last pipeline stages have data loaders
    if neox_args.is_pipe_parallel:
        is_first_stage = mpu.get_pipe_parallel_rank() == 0
        is_last_stage = mpu.get_pipe_parallel_rank() == mpu.get_pipe_parallel_world_size() - 1
        pipe_load = is_first_stage or is_last_stage
    else:
        pipe_load = True

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0 and pipe_load:
        # Number of train/valid/test samples.
        train_iters = neox_args.train_iters
        eval_iters = (train_iters // neox_args.eval_interval + 1) * neox_args.eval_iters
        test_iters = neox_args.eval_iters
        train_val_test_num_samples = [train_iters * neox_args.train_batch_size,
                                      eval_iters * neox_args.train_batch_size,
                                      test_iters * neox_args.train_batch_size]

        if neox_args.train_data_paths:
            raise NotImplementedError
        else:
            # when just data_path is provided
            # split dataset into train, valid and test from data_path
            train_ds, valid_ds, test_ds = get_dataset_fn(neox_args, train_val_test_num_samples)

        # Build dataloaders.
        if pad_sequences:
            train_dataloader = make_data_loader_with_padding(train_ds, neox_args, length_bins,
                                                             drop_last=drop_last, shuffle=shuffle)
            valid_dataloader = make_data_loader_with_padding(valid_ds, neox_args, length_bins,
                                                             drop_last=drop_last, shuffle=shuffle)
            test_dataloader = make_data_loader_with_padding(test_ds, neox_args, length_bins,
                                                             drop_last=drop_last, shuffle=shuffle)
        else:
            train_dataloader = make_data_loader(train_ds, neox_args=neox_args)
            valid_dataloader = make_data_loader(valid_ds, neox_args=neox_args)
            test_dataloader = make_data_loader(test_ds, neox_args=neox_args)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and neox_args.train_iters > 0
        do_valid = valid_dataloader is not None and neox_args.eval_iters > 0
        do_test = test_dataloader is not None and neox_args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    if neox_args.is_pipe_parallel:
        # Only first/last pipeline stages have data loaders, so pipeline parallelism should
        # broadcast globally instead of just the model parallel group.
        torch.distributed.broadcast(flags, src=0)
    else:
        torch.distributed.broadcast(flags,
                                    mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())
    neox_args.do_train = flags[0].item()
    neox_args.do_valid = flags[1].item()
    neox_args.do_test = flags[2].item()

    # Shift the start iterations.
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = (neox_args.iteration * neox_args.gradient_accumulation_steps) % \
                                                    len(train_dataloader)
        print_rank_0('setting training data start iteration to {}'.
                     format(train_dataloader.batch_sampler.start_iter))
    if valid_dataloader is not None:
        start_iter_val = ((neox_args.iteration * neox_args.gradient_accumulation_steps) // neox_args.eval_interval) * \
                         neox_args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % \
                                                    len(valid_dataloader)
        print_rank_0('setting validation data start iteration to {}'.
                     format(valid_dataloader.batch_sampler.start_iter))

    return train_dataloader, valid_dataloader, test_dataloader


def build_data_iterators(neox_args, get_dataset_fn, pad_sequences, length_bins):
    """Modified to accept different dataset and handle variable length sequences"""
    dataloaders = build_dataloaders(neox_args, get_dataset_fn, pad_sequences, length_bins)

    # Build iterators.
    output = []
    for i, data_loader in enumerate(dataloaders):
        if data_loader is not None:
            if i > 0:
                output.append(cycle(iter(data_loader)))
            else:
                output.append(iter(data_loader))

    return output


def get_batch(neox_args, data_iterator, keys, custom_batch_fn):
    """Generate a batch, assuming a sequential model."""
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    return custom_batch_fn(
        neox_args=neox_args,
        tokenizer=neox_args.tokenizer,
        keys=keys,
        data=data
    )


def get_batch_pipe(data, keys, custom_batch_fn, neox_args):
    """Generate a batch, assuming a pipeline module."""

    tokens, (labels, loss_mask), attention_mask, position_ids = custom_batch_fn(
        neox_args, tokenizer=neox_args.tokenizer, keys=keys, data=data
    )
    # VAV: need to return data in this order to be compatible with deepspeed pipelining code
    # unpack data
    return (tokens, position_ids, attention_mask), (labels, loss_mask)


# def clean_text(text):
#     """Remove new lines and multiple spaces and adjust end of sentence dot."""
#
#     text = text.replace("\n", " ")
#     text = re.sub(r'\s+', ' ', text)
#     for _ in range(3):
#         text = text.replace(' . ', '. ')
#
#     return text
#
#
# def build_sample(ids, types, paddings, label, unique_id):
#     """Convert to numpy and return a sample consumed by the batch producer."""
#
#     ids_np = np.array(ids, dtype=np.int64)
#     types_np = np.array(types, dtype=np.int64)
#     paddings_np = np.array(paddings, dtype=np.int64)
#     sample = ({'text': ids_np,
#                'types': types_np,
#                'padding_mask': paddings_np,
#                'label': int(label),
#                'uid': int(unique_id)})
#
#     return sample
#
#
# def build_tokens_types_paddings_from_text(text_a, text_b,
#                                           tokenizer, max_seq_length):
#     """Build token types and paddings, trim if needed, and pad if needed."""
#
#     text_a_ids = tokenizer.tokenize(text_a)
#     text_b_ids = None
#     if text_b is not None:
#         text_b_ids = tokenizer.tokenize(text_b)
#
#     return build_tokens_types_paddings_from_ids(text_a_ids, text_b_ids,
#                                                 max_seq_length, tokenizer.cls,
#                                                 tokenizer.sep, tokenizer.pad)
#
#
# def build_tokens_types_paddings_from_ids(text_a_ids, text_b_ids, max_seq_length,
#                                          cls_id, sep_id, pad_id):
#     """Build token types and paddings, trim if needed, and pad if needed."""
#
#     ids = []
#     types = []
#     paddings = []
#
#     # [CLS].
#     ids.append(cls_id)
#     types.append(0)
#     paddings.append(1)
#
#     # A.
#     len_text_a = len(text_a_ids)
#     ids.extend(text_a_ids)
#     types.extend([0] * len_text_a)
#     paddings.extend([1] * len_text_a)
#
#     # [SEP].
#     ids.append(sep_id)
#     types.append(0)
#     paddings.append(1)
#
#     # B.
#     if text_b_ids is not None:
#         len_text_b = len(text_b_ids)
#         ids.extend(text_b_ids)
#         types.extend([1] * len_text_b)
#         paddings.extend([1] * len_text_b)
#
#     # Cap the size.
#     trimmed = False
#     if len(ids) >= max_seq_length:
#         max_seq_length_m1 = max_seq_length - 1
#         ids = ids[0:max_seq_length_m1]
#         types = types[0:max_seq_length_m1]
#         paddings = paddings[0:max_seq_length_m1]
#         trimmed = True
#
#     # [SEP].
#     if (text_b_ids is not None) or trimmed:
#         ids.append(sep_id)
#         if text_b_ids is None:
#             types.append(0)
#         else:
#             types.append(1)
#         paddings.append(1)
#
#     # Padding.
#     padding_length = max_seq_length - len(ids)
#     if padding_length > 0:
#         ids.extend([pad_id] * padding_length)
#         types.extend([pad_id] * padding_length)
#         paddings.extend([0] * padding_length)
#
#     return ids, types, paddings
