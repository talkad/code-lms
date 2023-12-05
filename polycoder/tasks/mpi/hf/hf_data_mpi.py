import pdb
import json
import glob
import os, sys

import datasets as trd
from datasets import Sequence, Value
import numpy
import torch

from tokenizer.tokenizer import build_tokenizer

# sys.path.append('/home/nadavsc/LIGHTBITS/code-lms/polycoder/tasks/tokenizer')
with open(r'/home/nadavsc/LIGHTBITS/mpiricalplus/source/dataset/mpi.code-snippets', 'r') as f:
    file = json.load(f)
extended_tokens = [prefix.lower() for prefix in file.keys()]


def max_length_dataset_filter(args, max_length=512):
    tokenizer = build_tokenizer(args)
    tokenizer.tokenizer.add_tokens(extended_tokens)
    if args.tokenizer_type.lower() == 'GPT2BPETokenizer'.lower():
        eos_token = tokenizer.eod_id
    elif args.tokenizer_type.lower() == 'Tokompiler'.lower():
        eos_token = tokenizer.eod

    dataset_dir = args.data_path
    dpath = f'{dataset_dir}/mccplus_target_replaced_dataset.jsonl'
    with open(dpath, 'r') as db:
        with open(os.path.join(dataset_dir, f'mccplus_target_replaced_dataset_{max_length}.jsonl'), 'w') as target_db:
            for program in db:
                json_obj = json.loads(program)
                code = tokenizer.tokenize(json_obj['code'])[0]
                mpi_label = tokenizer.tokenize(json_obj["mpi_labels"])[0]

                full = code + [2] + mpi_label + [eos_token]
                if len(full) <= 512:
                    target_db.write(program)


def build_mpi_dataset(args, rebuild=False):
    if (not rebuild) and os.path.exists(os.path.join(args.data_path, "dataset_dict.json")):
        print('Loading dataset')
        tokenized_dataset = trd.load_from_disk(args.data_path)
    else:
        # Build the dataset
        print('Building dataset')
        tokenizer = build_tokenizer(args)

        if args.tokenizer_type.lower() == 'GPT2BPETokenizer'.lower():
            eos_token = tokenizer.eod_id

        elif args.tokenizer_type.lower() == 'Tokompiler'.lower():
            eos_token = tokenizer.eod

        else:
            raise NotImplementedError(f'{args.tokenizer_type} tokenizer type is not supported')

        tokenizer.tokenizer.add_tokens(extended_tokens)

        feature_types = trd.Features({
            "program": Value("string"),
            "code": Value("string"),
            "mpi_labels": Value("string"),
        })

        dataset_dir = args.data_path
        dpath = glob.glob(f'{dataset_dir}/*target*.jsonl')
        d = trd.load_dataset('json', data_files=dpath, features=feature_types, split=['train[0%:80%]', 'train[80%:100%]'])
        d = trd.DatasetDict({'train': d[0], 'test': d[1]})

        def tokenize_and_parse(example, eos_token=eos_token):
            code = example["code"]
            mpi_labels = example["mpi_labels"]

            if args.is_replaced:
                sep_token = '[SEP]'
                eos_token = '[EOS]'
            else:
                sep_token = '\n'
                eos_token = '' # eos equals to padding

            example["full"] = f'{code} {sep_token} parallel {mpi_labels} {eos_token}'
            return example


        # JSON fields are:
        #   program: a string identifier
        #   code: text of the source code, with each line numbered
        #   mpi_labels: the (location, mpi_function) tuples to predict as outputs

        tokenized_dataset = d.map(tokenize_and_parse, batched=False)
        tokenized_dataset.set_format(output_all_columns=True)

        tokenized_dataset.save_to_disk(args.data_path)

    return tokenized_dataset["train"], tokenized_dataset["test"]
