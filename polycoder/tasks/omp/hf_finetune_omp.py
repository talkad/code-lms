# Also tested for newest transformers version
# Tested for transformers==4.23.1, torch==1.12.1

import argparse
from functools import partial
import numpy as np
import os, sys
import tqdm
from typing import Literal

import torch
from torch.utils.data import DataLoader, Sampler, BatchSampler
from dataclasses import dataclass, field
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

from accelerate import Accelerator

# Imports needs for FSDP
#from accelerate import FullyShardedDataParallelPlugin
#from torch.distributed.fsdp.fully_sharded_data_parallel import (
#    MixedPrecision, CPUOffload,
#    FullOptimStateDictConfig, FullStateDictConfig
#)
##from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
# Imports to try gradient checkpointing
#from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
#    checkpoint_wrapper,
#    CheckpointImpl,
#    apply_activation_checkpointing,
#)

# Dataset specific import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(f'PATH is now {sys.path}')
from tasks.omp.hf_data_omp import build_omp_dataset as dataset_builder
from tasks.data_utils import SeqLengthSampler  # sampler that batches out longest sequences first -- with some randomness within length bins. Will ensure better memory allocation


class DistributedSampler(Sampler):
    """ 
    VAV NB: this function taken from torchnlp
    Iterable wrapper that distributes data across multiple workers.

    Args:
        iterable (iterable)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within ``num_replicas``.

    Example:
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=0))
        [0, 2, 4, 6, 8]
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=1))
        [1, 3, 5, 7, 9]
    """

    def __init__(self, iterable, num_replicas=None, rank=None):
        self.iterable = iterable
        self.num_replicas = num_replicas
        self.rank = rank

        if num_replicas is None or rank is None:  # pragma: no cover
            if not torch.distributed.is_initialized():
                raise RuntimeError('Requires `torch.distributed` to be initialized.')

            self.num_replicas = (
                torch.distributed.get_world_size() if num_replicas is None else num_replicas)
            self.rank = torch.distributed.get_rank() if rank is None else rank

        if self.rank >= self.num_replicas:
            raise IndexError('`rank` must be smaller than the `num_replicas`.')

    def __iter__(self):
        return iter(
            [e for i, e in enumerate(self.iterable) if (i - self.rank) % self.num_replicas == 0])

    def __len__(self):
        return len(self.iterable)


class DistributedBatchSampler(BatchSampler):
    """ `BatchSampler` wrapper that distributes across each batch multiple workers.

    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.

    Example:
        >>> from torch.utils.data.sampler import BatchSampler
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(12)))
        >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        >>>
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=0))
        [[0, 2], [4, 6], [8, 10]]
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=1))
        [[1, 3], [5, 7], [9, 11]]
    """

    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs

    def __iter__(self):
        for batch in self.batch_sampler:
            yield list(DistributedSampler(batch, **self.kwargs))

    def __len__(self):
        return len(self.batch_sampler)


def main(data_args, training_args, model):
    # CPU acceleration, fp16 is averaging ~30s/it

    # My FSDP setup runs into an OOM error for max length sequences, even at batch size 1.
    # FSDP with 2 GPUs (48GB A40 GPUs) averaging ~5s/it with CPU offloading of parameters
    # FSDP with 2 GPUs (48GB A40 GPUs) averaging ~3s/it with all above, but bfloat16
    #fsdp_plugin = FullyShardedDataParallelPlugin(cpu_offload=CPUOffload(offload_params=True),
    #    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    #    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    #    mixed_precision_policy=MixedPrecision(param_dtype=torch.bfloat16,
    #                                          reduce_dtype=torch.bfloat16,
    #                                          buffer_dtype=torch.bfloat16),
    #    activation_checkpointing=True,
    #    limit_all_gathers=True,
    #)
    #accel = Accelerator(fsdp_plugin=fsdp_plugin)
    #print(fsdp_plugin)
    accel = Accelerator(cpu=True)

    # Model needs to be prepared before the optimizer
    model = model.to(accel.device)
    model = accel.prepare(model)
    print(f'DEVICE IS {model.device}')
    print(model)
    print(accel)

    # Attempt to get activation checkpointing to work...
    #check_fn = lambda subm: isinstance(subm, torch.nn.Linear)
    #non_reentrant_wrapper = partial(
    #    checkpoint_wrapper,
    #    offload_to_cpu=True,
    #    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    #)
    #apply_activation_checkpointing(
    #    model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    #)

    # TODO: make this also compatible with Tokompiler
    # Build tokenizer
    tokenizer = GPT2Tokenizer(vocab_file=data_args.vocab_file, merges_file=data_args.merge_file, padding=True,
                              truncation=True, model_input_names=['input_ids'])
    tokenizer.pad_token = tokenizer.eos_token
    datasets = dataset_builder(None, None, tokenizer=tokenizer,
                               data_path=data_args.data_path, save_dataset=data_args.save_dataset)
    # NOTE: The following map() function re-tokenizes the dataset...even though that already happens in the original dataset builder, I am overwriting that data field since the standard HF method has a fast method for hashing/caching all dataset transforms. Since this is redundant, you can remove the tokenization from the dataset builder code if you wish.
    newd = []
    for i in range(len(datasets)):
        d = datasets[i]
        outd = d.map(lambda examples: tokenizer(examples['code']), remove_columns=['source', 'code'])
        newd.append(outd)
    traind, vald, testd = newd

    world_size = accel.num_processes
    accel.print(f"Reading world_size as {world_size}...")
    sampler_batch_sz = training_args.per_device_train_batch_size * world_size
    distr_batch_sz = training_args.per_device_train_batch_size if world_size > 1 else 1

    # Build data loader
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    length_sampler = SeqLengthSampler(traind, bucket_boundaries=[2048, 1526, 1024, 0],
                                      batch_size=sampler_batch_sz)
    batch_sampler = DistributedBatchSampler(length_sampler, num_replicas=world_size, rank=accel.process_index)
    train_loader = DataLoader(dataset=traind, batch_size=training_args.per_device_train_batch_size,
                              collate_fn=collator, sampler=None, batch_sampler=batch_sampler)
    val_sampler0 = SeqLengthSampler(vald, bucket_boundaries=[2048, 1526, 1024, 0],
                                    batch_size=sampler_batch_sz)
    val_batch_sampler = DistributedBatchSampler(val_sampler0, num_replicas=world_size, rank=accel.process_index)
    val_loader = DataLoader(dataset=vald, batch_size=training_args.per_device_train_batch_size,
                            collate_fn=collator, shuffle=False,
                            sampler=None, batch_sampler=val_batch_sampler)

    # Setup optimizer and learning rate scheduler
    num_epochs, learning_rate, grad_accum_steps = int(training_args.num_train_epochs), training_args.learning_rate, \
        int(training_args.gradient_accumulation_steps)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=100,
                                                   num_training_steps=(len(train_loader) * num_epochs) // grad_accum_steps,)
    # SGD is a lower memory footprint optimizer...likely don't need
#    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
#    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * num_epochs) // grad_accum_steps)

    train_loader, val_loader, optimizer, lr_scheduler = \
        accel.prepare([train_loader, val_loader, optimizer, lr_scheduler])

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm.tqdm(train_loader, miniters=2, desc=f"Epoch {epoch}")
        loss_total = 0.0 
        for step, batch in enumerate(pbar):
            del batch['length']
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accel.device)
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / grad_accum_steps
            accel.backward(loss)
            if step % grad_accum_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            loss_total += loss.detach().clone().item()
            if (step > 0) and (step % 10 == 0):
                pbar.set_postfix({"avg_train_loss": loss_total / step})
            # TODO: add checkpoint saving
            if step % training_args.save_steps == 0:
                accel.print("Saving checkpoint...") 
                model.save_pretrained(training_args.output_dir + f'/e{epoch}_s{step}',
                                      is_main_process=accel.is_main_process,
                                      save_function=accel.save)
                accel.print("saved!") 

        model.eval()
        pbar = tqdm.tqdm(val_loader, miniters=2, desc=f"Epoch {epoch} Validation")
        val_loss_total = 0.0
        for step, batch in enumerate(pbar):  #val_loader):
            del batch['length']
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accel.device)
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            val_loss_total += loss.detach().clone().item()
            if ((step > 0) and (step % 10 == 0)) or (step == len(pbar)):
                pbar.set_postfix({"avg_val_loss": val_loss_total / step})
            #predictions = outputs.logits.argmax(dim=-1)
            #predictions, references = accel.gather_for_metrics((predictions, batch["labels"]))
            # metric.add_batch(
            #     predictions=predictions,
            #     references=references,
            # )

        # eval_metric = metric.compute()
        # Use accelerator.print to print only on the main process.
        # accel.print(f"epoch {epoch}:", eval_metric)

    accel.print("Finished training epochs!")
    accel.print("Saving last checkpoint...") 
    model.save_pretrained(training_args.output_dir + '/e{epoch}_end',
                          is_main_process=accel.is_main_process,
                          save_function=accel.save)
    accel.print("saved!") 


def load_model(ckpt_home, ckpt_name, device=None):
    model = GPTNeoXForCausalLM.from_pretrained(ckpt_home + ckpt_name)
    if device:
        model = model.to(torch.device(device))
    return model


# I've hard-coded the defaults here, but these can be passed as input arguments to the script
@dataclass
class DatasetArguments:
    vocab_file: str = field(default='/home/vyvo/hpcoder/code-lms/polycoder/megatron/tokenizer/gpt_vocab/gpt2-vocab.json')
    merge_file: str = field(default='/home/vyvo/hpcoder/code-lms/polycoder/megatron/tokenizer/gpt_vocab/gpt2-merges.txt')
    data_path: str = field(default=f'/export/data/vyvo/OMP_Dataset')
    save_dataset: bool = field(default=True)
    tokenizer_type: str = field(default="GPT2BPETokenizer")
        # Tokenizer to use. Should be one of ["GPT2BPETokenizer", "HFTokenizer", "HFGPT2Tokenizer", "CharLevelTokenizer", "Tokompiler"]
    """
    Following arguments leftover from megatron, but still needed to build tokenizer using its functions
    """
    rank: int = field(default=0)
    make_vocab_size_divisible_by: int = field(default=128)
    model_parallel_size: int = field(default=1)


if __name__ == "__main__":
    parser = HfArgumentParser((DatasetArguments, TrainingArguments))  # This class will allow HF training arguments to be passed as inputs
    
    data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.no_cuda = True
    training_args.fp16 = True  #True
    training_args.use_ipex = True
    #training_args.ddp_backend = 'ccl'
    training_args.gradient_checkpointing = True
    training_args.per_device_train_batch_size = 1
    training_args.per_device_eval_batch_size = 1
    training_args.gradient_accumulation_steps = 4  #8
    training_args.save_steps = 1000
    training_args.save_total_limit = 5

    # TODO: put these in argparser args
    ckpt_home = '/home/vyvo/hpcoder/checkpoints/hf_conversions/'
    ckpt_name = 'allc_gpt2tok_2-7B/'
    model = load_model(ckpt_home, ckpt_name, device='cpu')

    print(data_args)
    print(training_args)

    main(data_args, training_args, model)

