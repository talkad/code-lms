import json
from tqdm import tqdm
from calc_metrics import *

def tokenize(code, is_replaced=False):
    pass


generation_file = ''
total_bleu, total_code_bleu, total_code_bert = 0, 0, 0

with open(generation_file, 'r') as f:
    for line in tqdm(f):
        js = json.loads(line.strip())

        label = js['label']
        pred = js['pred']






