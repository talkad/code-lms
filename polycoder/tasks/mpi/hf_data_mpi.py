import glob
import os, sys

import datasets as trd
from datasets import Sequence, Value
import numpy
import torch


def build_mpi_dataset(args, rebuild=False):
    if (not rebuild) and os.path.exists(os.path.join(args.data_path, "test/data.arrow")):
        tokenized_datasets = trd.load_from_disk(args.data_path)
    else:
        # Build the dataset
        from ..tokenizer import build_tokenizer
        tokenizer = build_tokenizer(args)

        feature_types = trd.Features({
            "program": Value("string"),
            "code": Value("string"),
            "mpi_labels": Value("string"),
        })

        dataset_dir = args.data_path
        dpath = glob.glob(f'{dataset_dir}/*target*.jsonl')
        d = trd.load_dataset('json', data_files=dpath, features=feature_types,
                             split=['train[0%:80%]', 'train[80%:90%]', 'train[90%:100%]'])
        d = trd.DatasetDict({'train': d[0], 'validation': d[1], 'test': d[2]})

        def tokenize_and_parse(example):
            tmp = f'{example["code"]}\n{example["mpi_labels"]}'
            example["input_ids"] = tokenizer.tokenize(tmp) + [tokenizer.eod_id]
            example["length"] = len(example["input_ids"])
            # example["input_ids"] = tokenizer.tokenize(example["code"])
            # example["mpi_completion"] = tokenizer.tokenize(example["mpi_labels"])
            tmp = example["mpi_labels"].split(';) ')
            mpi_list = [t + ';)' for t in tmp]
            example["mpi_indiv"] = mpi_list  #[tokenizer.tokenize(m) for m in mpi_list[:-1]]
            return example

        # JSON fields are:
        #   program: a string identifier
        #   code: text of the source code, with each line numbered
        #   mpi_labels: the (location, mpi_function) tuples to predict as outputs

        tokenized_dataset = d.map(tokenize_and_parse, batched=False)

        tokenized_dataset.set_format(type="torch",
                                     columns=['input_ids'],  # 'mpi_completion'],
                                     output_all_columns=True)  #, 'mpi_indiv'])

        tokenized_dataset.save_to_disk(args.data_path)

        return tokenized_dataset["train"], tokenized_dataset["validation"], tokenized_dataset["test"]


#def mpi_batch_fn(neox_args, tokenizer, keys, data, datatype=torch.int64):
#    """Support function for get_batch / get_batch pipe (to avoid code repetition). See tasks/data_utils.py."""
#    data_b = mpu.broadcast_data(keys, data, datatype)
#
#    # Unpack.
#    tokens_ = data_b["input_ids"].long().squeeze()
#
#    if tokens_.ndim > 2:
#        # this is to debug an intermittent error that we're getting in batching
#        print(data)
#        print(tokens_.shape)
#        import pdb; pdb.set_trace()
#        # assert len(einds) == 1, f"Not sure how to handle batch! Error indices: {data_b['error_inds']}"
#        # assert data_b["error_inds"][0].shape[0] == nbatch, \
#        #     f"Error indices are unexpected shape {[x.shape for x in data_b['error_inds']]} for batch size {nbatch}"
#        # # Catch the rare case where all of these are the same length
#        # einds = torch.split(data_b["error_inds"][0], 1)
#        # einds = [r.squeeze() for r in einds]
#
#    labels = tokens_[:, 1:]
#    tokens = tokens_[:, :-1].contiguous()
#
#    nbatch, seq_length = tokens.shape
#    # Create masks for loss function. Mask values of 1 are kept. Default 0.
#
#    # Placeholder for MPI labels once we figure out how to score them
#    # labels = (lm_labels, torch.empty((0, 1)))  #data_b["mpi_labels"])
#    # Get the masks and position ids.
#    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
#        tokens, 0, True
#    )
#
#    return tokens, (labels, loss_mask), attention_mask, position_ids

