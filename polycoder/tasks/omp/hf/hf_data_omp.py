import glob
import os, sys
import re
import datasets as trd
from datasets import Sequence, Value
import numpy
import torch

sys.path.append("/mnt/lbosm1/home/Share/code-lms/polycoder/tasks/tokenizer")
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
            tokenizer.enable_padding(length=256)

        else:
            raise NotImplementedError(f"We do not support the tokenizer type {args.tokenizer_type}")

        feature_types = trd.Features({
            "code": Value("string"),
            "pragma": Value("string"),
            "hash": Value("string"),
        })

        dpath = os.path.join(args.data_path, args.data_device, 'replaced' if args.is_replaced else 'source', args.data_filename)
        d = trd.load_dataset('json', data_files=[dpath], features=feature_types,
                             split=['train[0%:80%]', 'train[80%:90%]', 'train[90%:100%]'])
        d = trd.DatasetDict({'train': d[0], 'validation': d[1], 'test': d[2]})

        def tokenize_and_parse(example, eos_token=eos_token):
            code = lexicalize(example["code"], replaced=args.is_replaced)
            pragma = example["pragma"]

            ### Simplified pragma ###
            pragma_dict = pragma2dict(pragma)
            pragma = f"private {' '.join(pragma_dict['private']['vars']) if 'private' in pragma_dict else ''} || reduction {pragma_dict['reduction']['operator'] + ' : ' + ' '.join(pragma_dict['reduction']['vars']) if 'reduction' in pragma_dict else ''} "

            if args.is_replaced:
                pragma = pragma.replace('_', ' ')
            #########################

            tmp = f'{code} [SEP] {pragma}'
            example["input_ids"] = tokenizer.tokenize(f'{code} [SOS] ')[0]
            example["labels"] = tokenizer.tokenize(f'{pragma} [EOS] ')[0]

            # example["input_ids"] = tokenizer.tokenize(tmp)[0] + [eos_token]
            example["length"] = len(example["input_ids"])
            return example

        # JSON fields are:
        #   hash: an alphanumeric identifier
        #   code: text of the source code
        #   pragma: the pragma to predict given the input code

        tokenized_dataset = d.map(tokenize_and_parse, batched=False)

        tokenized_dataset.set_format(type="torch",
                                     columns=['input_ids', 'labels'],
                                     output_all_columns=True)
        if args.save:
            tokenized_dataset.save_to_disk(args.data_path)

    return tokenized_dataset["train"], tokenized_dataset["validation"], tokenized_dataset["test"]



