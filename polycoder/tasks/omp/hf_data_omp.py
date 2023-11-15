import glob
import os, sys

import datasets as trd
from datasets import Sequence, Value
import numpy
import torch


def build_omp_dataset(args, rebuild=False):
    if (not rebuild) and os.path.exists(os.path.join(args.data_path, "test/data.arrow")):
        # Load the already constructed dataset
        tokenized_datasets = trd.load_from_disk(args.data_path)
    else:
        # Build the dataset
        from ..tokenizer import build_tokenizer
        tokenizer = build_tokenizer(args)
        if args.tokenizer_type.lower() == 'GPT2BPETokenizer'.lower():
            eos_token = tokenizer.eod_id
        elif args.tokenizer_type.lower() == 'Tokompiler'.lower():
            eos_token = tokenizer.eod
        else:
            raise NotImplementedError(f"We do not support the tokenizer type {args.tokenizer_type}")

        feature_types = trd.Features({
            "code": Value("string"),
            "pragma": Value("string"),
            "hash": Value("string"),
        })

        dpath = glob.glob(f'{args.data_path}/*.jsonl')
        d = trd.load_dataset('json', data_files=dpath, features=feature_types,
                             split=['train[0%:80%]', 'train[80%:90%]', 'train[90%:100%]'])
        d = trd.DatasetDict({'train': d[0], 'validation': d[1], 'test': d[2]})

        def tokenize_and_parse(example, eos_token=eos_token):
            tmp = f'{example["code"]}\n{example["pragma"]}'
            example["input_ids"] = tokenizer.tokenize(tmp) + [eos_token]  #[tokenizer.eod_id]
            example["length"] = len(example["input_ids"])
            return example

        # JSON fields are:
        #   hash: an alphanumeric identifier
        #   code: text of the source code
        #   pragma: the pragma to predict given the input code

        tokenized_dataset = d.map(tokenize_and_parse, batched=False)

        tokenized_dataset.set_format(type="torch",
                                     columns=['input_ids'],
                                     output_all_columns=True)
        if args.save:
            tokenized_dataset.save_to_disk(args.data_path)

        return tokenized_dataset["train"], tokenized_dataset["validation"], tokenized_dataset["test"]


#def omp_batch_fn(neox_args, tokenizer, keys, data, datatype=torch.int64):
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
#
#    labels = tokens_[:, 1:]
#    tokens = tokens_[:, :-1].contiguous()
#
#    nbatch, seq_length = tokens.shape
#    # Create masks for loss function. Mask values of 1 are kept. Default 0.
#
#    # Get the masks and position ids.
#    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
#        tokens, 0, True
#    )
#
#    return tokens, (labels, loss_mask), attention_mask, position_ids

