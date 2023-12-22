import glob
import os, sys
import re
import datasets as trd
from datasets import Sequence, Value
import numpy
import torch
import copy
import json

sys.path.append("../../tokenizer")
from tokompiler.lexicalization import lexicalize


def pragma2dict(pragma):
    """
    Convert an openmp pragma into dictionary.
    do private ( var_501 )  ->   {private: {vars: [var_501]}}

    Assumes legal omp pragmas
    """
    result = {}
    pattern = r' (private)\s*\((.*?)\)| (reduction)\s*\((.+?):(.*?)\)| (simd)'

    matches = re.findall(pattern, pragma)
    for match in matches:
        private, private_vars, reduction, reduction_op, reduction_vars, simd = match
        
        if private:
            result['private'] = {}
            result['private']['vars'] = private_vars.split()
            # result['private']['vars'] = private_vars.replace(' ','').split(',')
        elif reduction:
            result['reduction'] = {}
            result['reduction']['operator'] = reduction_op
            result['reduction']['vars'] = reduction_vars.split()
            # result['reduction']['vars'] = reduction_vars.replace(' ','').split(',')
        elif simd:
            result['simd'] = {}
            result['simd']['vars'] = []

    return result


def pragma2str(pragma):
    '''
    {private: {vars: [var_501]}} -> do private ( var_501 )
    '''
    result = ''

    for clause, inner_clause in pragma.items():
        if clause == 'private':
            result += f"private ( {', '.join(inner_clause['vars'])} )"
        if clause == 'reduction':
            result += f"reduction ( {inner_clause['operator']} : {', '.inner_clause['vars'])} )"


def remove_pragma(code):
    buf = []

    for line in code.split('\n'):
        if line.lstrip().startswith('#pragma'):
            continue
        buf.append(line)

    return '\n'.join(buf)


def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    return [json.loads(line.strip()) for line in lines if 'for' in line]


def build_omp_dataset(args, rebuild=False):
    if (not rebuild) and os.path.exists(os.path.join(args.data_path, "test/data.arrow")):
        # Load the already constructed dataset
        tokenized_datasets = trd.load_from_disk(args.data_path)
    else:
        # Build the dataset
        from tokenizer import build_tokenizer
        tokenizer = build_tokenizer(args)

        if args.tokenizer_type.lower() == 'GPT2BPETokenizer'.lower():
            eos_token = tokenizer.eod_id

        elif args.tokenizer_type.lower() == 'Tokompiler'.lower():
            eos_token = tokenizer.eod

            tokenizer.add_tokens(['private', 'reduction', 'simd'])

        else:
            raise NotImplementedError(f"We do not support the tokenizer type {args.tokenizer_type}")

        feature_types = trd.Features({
            "code": Value("string"),
            "pragma": Value("string"),
            "hash": Value("string"),
        })
        
        if args.do_eval:
            train_data_path = os.path.join(args.data_path, 'HPCorpus_omp_replaced.jsonl')
            test_data_path = os.path.join(args.data_path, 'HPCorpus_omp_replaced.jsonl')

            train_dataset = read_jsonl(train_data_path)
            test_dataset = read_jsonl(test_data_path)
        else:
            train_data_path = os.path.join(args.data_path, args.data_device, 'replaced' if args.is_replaced else 'source', 'train.jsonl')
            test_data_path = os.path.join(args.data_path, args.data_device, 'replaced' if args.is_replaced else 'source', 'test.jsonl')

            train_dataset = read_jsonl(train_data_path)
            test_dataset = read_jsonl(test_data_path)

        columns = train_dataset[0].keys()
        train_dataset = trd.Dataset.from_dict({col: [item[col] for item in train_dataset] for col in columns})
        test_dataset = trd.Dataset.from_dict({col: [item[col] for item in test_dataset] for col in columns})

        d = trd.DatasetDict({'train': train_dataset, 'test': test_dataset})
        

        def tokenize_and_parse(example, eos_token=eos_token):
            code = remove_pragma(example["code"])
            if args.is_replaced:
                code = lexicalize(code, replaced=True)

            pragma = example["pragma"]
            pragma = pragma2str(pragma2dict(pragma))

            if args.is_replaced:
                pragma = pragma.replace('_', ' ')

            if args.do_eval or args.do_test:
                code = code
                example["full"] = code
                example["label"] = pragma
            else:

                if args.is_replaced:
                    sep_token = '[SEP]'
                    eos_token = '[EOS]'
                else:
                    sep_token = '\n'
                    eos_token = '' # eos equals to padding - appended it at tokenization

                example["full"] = f'{code} {sep_token} parallel {pragma} {eos_token}'

            return example

        tokenized_dataset = d.map(tokenize_and_parse, batched=False)

        tokenized_dataset.set_format(output_all_columns=True)

        if args.save:
            tokenized_dataset.save_to_disk(args.data_path)

    return tokenized_dataset["train"], tokenized_dataset["test"]



