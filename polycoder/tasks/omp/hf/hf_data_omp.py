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
    do private ( var_501 )  ->   {private,vars: [var_501]}

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


def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [json.loads(line.strip()) for line in lines]


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

        train_data_path = os.path.join(args.data_path, args.data_device, 'replaced' if args.is_replaced else 'source', 'train.jsonl')
        test_data_path = os.path.join(args.data_path, args.data_device, 'replaced' if args.is_replaced else 'source', 'test.jsonl')

        train_dataset = read_jsonl(train_data_path)
        test_dataset = read_jsonl(test_data_path)

        columns = train_dataset[0].keys()
        train_dataset = trd.Dataset.from_dict({col: [item[col] for item in train_dataset] for col in columns})
        test_dataset = trd.Dataset.from_dict({col: [item[col] for item in test_dataset] for col in columns})

        d = trd.DatasetDict({'train': train_dataset, 'test': test_dataset})
        

        def tokenize_and_parse(example, eos_token=eos_token):
            # import pdb; pdb.set_trace()

            code = example["code"]
            pragma = example["pragma"]

            if args.is_replaced:
                code = lexicalize(code, replaced=args.is_replaced)

            # ### Simplified pragma ###
            # pragma_dict = pragma2dict(pragma)
            # pragma = f"for {'|| private' if 'private' in pragma_dict else ''} {' '.join(pragma_dict['private']['vars']) if 'private' in pragma_dict else ''} {'|| reduction' if 'reduction' in pragma_dict else ''} {pragma_dict['reduction']['operator'] + ' : ' + ' '.join(pragma_dict['reduction']['vars']) if 'reduction' in pragma_dict else ''} "
            # #########################

            ### Data Preproccess ###
            pragma = pragma[pragma.find('for'):]
            pragma = pragma.replace('parallel', '')

            if args.is_replaced:
                pragma = pragma.replace('_', ' ')
                pragma = pragma.replace('(', ' ( ').replace(')', ' ) ').replace(':', ' : ').replace(',', ' , ')
            #########################

            if args.is_replaced:
                sep_id, _ = tokenizer.tokenize('[SEP]')
                sep_id = sep_id[0]
                eos_id = tokenizer.eod

                code, _ = tokenizer.tokenize(code)
                pragma, _ = tokenizer.tokenize(pragma)
            else:
                sep_id = tokenizer.tokenize('\n')[0]
                eos_id = tokenizer.eod_id

                code = tokenizer.tokenize(code)
                pragma = tokenizer.tokenize(pragma)

            if args.do_eval: # for PPL evaluation only the code is used without the pragma
                full = code + [eos_id]
            elif args.do_test:
                full = code + [sep_id]
            else:
                full =  code + [sep_id] + pragma + [eos_id]  

            example["input_ids"] = full

            ## PADDING and MASKING ##
            max_length = 512
            example["input_ids"] = example["input_ids"][:max_length]
            example["input_ids"] += (max_length - len(example["input_ids"])) * [tokenizer.pad_id]

            labels = example["input_ids"].copy()
            example["labels"] = labels
            
            if args.do_eval or args.do_test:
                example["mask"] = [1] * len(code) + [1]  + [0] * (max_length - len(code) -1)
                example["mask"] = example["mask"][:max_length]
            else:
                example["mask"] = [0] * len(code) + [1] * (len(pragma)+2) + [0] * (max_length - len(code) - len(pragma)-2)
                example["mask"] = example["mask"][:max_length]
            ##########################

            labels = example["input_ids"].copy()
            example["labels"] = labels

            example["length"] = len(example["input_ids"])

            return example

        # JSON fields are:
        #   hash: an alphanumeric identifier
        #   code: text of the source code
        #   pragma: the pragma to predict given the input code

        tokenized_dataset = d.map(tokenize_and_parse, batched=False)

        tokenized_dataset.set_format(type="torch",
                                     columns=['input_ids', 'labels', 'mask'],
                                     output_all_columns=True)
        if args.save:
            tokenized_dataset.save_to_disk(args.data_path)

    return tokenized_dataset["train"], tokenized_dataset["test"]



