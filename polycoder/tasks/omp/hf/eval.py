import os, sys
import transformers
# NOTE: This has only been tested for transformers==4.23.1, torch==1.12.1
from transformers import GPTNeoXForCausalLM
import argparse

# sys.path.insert(0, '..')
# import hf_data_omp as data_omp


ckpt_home = '/home/talkad/shared/models/hf_checkpoints'
ckpt_name = 'allc_tokom_700M/'

model = GPTNeoXForCausalLM.from_pretrained(os.path.join(ckpt_home, ckpt_name))


# parser = argparse.ArgumentParser()
# parser.add_argument('-t', '--tokenizer_type', type=str, default='GPT2BPETokenizer')
# parser.add_argument('-v', '--vocab_file', type=str, default='./megatron/tokenizer/gpt_vocab/gpt2-vocab.json')
# parser.add_argument('-m', '--merge_file', type=str, default='./megatron/tokenizer/gpt_vocab/gpt2-merges.txt')
# parser.add_argument('-d', '--data_path', type=str, default=f'{os.path.expanduser("~")}/data/OMP_Dataset/c-cpp/source/')
# parser.add_argument('--save', type=bool, default=True)
# # The following arguments are leftover from megatron settings -- you can keep the defaults
# parser.add_argument('--rank', type=int, default=0)
# parser.add_argument('--make_vocab_size_divisible_by', type=int, default=128)
# parser.add_argument('--model_parallel_size', type=int, default=1)

# args = parser.parse_known_args()[0]

# print(args)


# # Load dataset. This will build the dataset if it does not exist.
# traind, vald, testd = data_omp.build_omp_dataset(args)

