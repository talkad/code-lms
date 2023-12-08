import json
import sys
from tqdm import tqdm
from metrics import *
from transformers import GPT2Tokenizer



tokenizer = GPT2Tokenizer(vocab_file='../../../../../megatron/tokenizer/gpt_vocab/gpt2-vocab.json', 
                                merges_file='../../../../../megatron/tokenizer/gpt_vocab/gpt2-merges.txt')


def tokenize(code, is_replaced=False):
    if is_replaced:
        return code.split()
    else:
        return tokenizer.tokenize(code)
    


generation_file = '/mnt/lbosm1/home/Share/OMPify/CompCoder/comparison_models/GPT3.5_Turbo/context_600.jsonl'
total_bleu, total_code_bleu, total_code_bert = 0, 0, 0

with open(generation_file, 'r') as f:
    for idx, line in tqdm(enumerate(f, start=1)):
        js = json.loads(line.strip())

        label = js['label']
        pred = js['pred']

        total_bleu += calc_bleu(tokenize(pred), tokenize(label))
        total_code_bleu += calc_code_bleu(pred, label)
        total_code_bert += calc_code_bert_score(pred, label)

        print(f'Bleu: {total_bleu/idx}')
        print(f'CodeBleu: {total_code_bleu/idx}')
        print(f'CodeBERTScore: {total_code_bert/idx}')

    print(f'Bleu: {total_bleu/500}')
    print(f'CodeBleu: {total_code_bleu/500}')
    print(f'CodeBERTScore: {total_code_bert/500}')






