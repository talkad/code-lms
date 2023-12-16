import os
import torch
import math
import json
import logging
import hf_data_omp as data_omp
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, Trainer, TrainingArguments
from tqdm.auto import tqdm
from torch.nn.functional import cross_entropy, one_hot
from torch.utils.data import DataLoader, Dataset
from transformers import GPTNeoXForCausalLM
from tokenizer import TokompilerTokenizer, _GPT2BPETokenizer
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
import pygments
from pygments.lexers import get_lexer_by_name
from nltk.translate.bleu_score import sentence_bleu



logger = logging.getLogger()
lexer = get_lexer_by_name('cpp')
LEX_VOCAB=sum([len(v) for v in lexer.tokens.values()])
logger.info(f' --- lex vocab amount is {LEX_VOCAB}')



class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.data[idx]['input_ids']),
                'labels': torch.tensor(self.data[idx]['labels'])}


def tokenize(args, tokenizer, sample):
    max_size = 2048

    if not args.is_replaced:
        encodings = tokenizer(sample['full'], max_length=max_size, add_special_tokens=True, truncation=True, padding=True)

        if len(encodings['input_ids']) < max_size:
            encodings['input_ids'].append(tokenizer.eos_token_id)
    else:
        encodings = {}
        encodings['input_ids'] = tokenizer(sample['full'], max_length=max_size, add_special_tokens=True, truncation=True, padding=True)
        encodings['labels'] = encodings['input_ids'][:]

    return encodings


def decode(logits, labels, tokenizer, ignore_token_id, is_replaced=False):
    pred, label = [], []

    # preds = torch.argmax(logits,dim=-1)
    preds = logits
   
    labels = labels[labels!=ignore_token_id]
    preds = preds[0][:len(labels)]

    for token_id in preds.tolist():
        pred.append(tokenizer.decode([token_id]))

    for token_id in labels.tolist():
        label.append(tokenizer.decode([token_id]))

    if is_replaced:
        is_not_special = lambda token: token not in tokenizer.tokenizer.special_tokens
        label = list(filter(is_not_special, label))
        pred = pred[:len(label)]

    return pred, label


def calculate_metrics(logits, labels, tokenizer, ignore_token_id=1):
    y = labels
    logits = logits

    ce = cross_entropy(logits, one_hot(y, num_classes=logits.shape[-1]).to(torch.float32), reduction='sum')

    # pred, label = decode(logits, labels, tokenizer, ignore_token_id)

    return {'ce': ce}
            # 'bleu': sentence_bleu([label], pred), 
            # 'code_bleu' : 0, 
            # 'code_bert': 0, 
            # 'acc': torch.sum(labels==preds).item()/len(preds)}



def eval(args):
    logger.info(f'start compcoder HPC evaluation {args.model_name}')

    # TOKENIZER
    tokom_extended_tokens = ['parallel', 'private', 'reduction']

    if args.is_replaced:
        tokenizer = TokompilerTokenizer(vocab_path='/mnt/lbosm1/home/Share/code-lms/polycoder/tasks/tokenizer/tokompiler/tokenizer_vocab/vocab.txt') #args.vocab_file)
        tokenizer.add_tokens(tokom_extended_tokens)
        tokenizer.enable_padding(length=2048)
    else:
        tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B", 
                                  truncation=True, model_input_names=['input_ids'])
        # tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file, padding=True,
        #                         truncation=True, model_input_names=['input_ids'])
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    # DATA
    datasets = data_omp.build_omp_dataset(args)
    codes = datasets[1]['code']
    
    newd = []
    for i in range(len(datasets)):
        d = datasets[i]
        outd = d.map(lambda examples: tokenize(args, tokenizer, examples), remove_columns=['code', 'full'])     
        newd.append(outd)

    traind, testd = newd

    if args.is_replaced:

        train_data = []
        for ids, labels in tqdm(zip(traind['input_ids'], traind['labels'])):
            train_data.append({'input_ids': ids, 'labels': labels})

        test_data = []
        for ids, labels in tqdm(zip(testd['input_ids'], testd['labels'])):
            test_data.append({'input_ids': ids, 'labels': labels})

        test_loader = DataLoader(dataset=CustomDataset(test_data), batch_size=1, shuffle=False)
    else:
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        test_loader = DataLoader(dataset=testd, batch_size=1, collate_fn=collator, shuffle=False)


    # get model
    # model = GPTNeoXForCausalLM.from_pretrained(os.path.join(args.models_dir, args.model_name))    
    # model = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B", torch_dtype=torch.float16)
    model = GPTNeoXForCausalLM.from_pretrained('/home/talkad/shared/models/hf_checkpoints/allc_gpt2tok_700M')

    model.eval()
    import pdb; pdb.set_trace()

    model.to(args.device)

    # EVAL
    context_size = 600
    progress_bar = tqdm(range(args.num_epochs * len(test_loader)))

    total_ce, total_tokens = 0.0, 0
    num_token = 0
    batches_to_print = 100
    total_bleu, steps = 0, 0
    total_acc = 0
    
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(test_loader):
            tensor_batch = {k: v.to(args.device) for k, v in batch.items() if k in ['input_ids', 'labels', 'mask']}
            
            input_ids = tensor_batch['input_ids']
            mask = torch.ones_like(input_ids)

            if args.is_replaced:
                mask[input_ids==tokenizer.pad_id] = 0
                mask[input_ids==tokenizer.eod] = 0
            else:
                mask[input_ids==tokenizer.eos_token_id] = 0

            input_ids = input_ids[mask==1]
            tokens_amount = input_ids.shape[-1]
            num_token += tokens_amount

            # ### PPL and Accracy ########################################################
            # # import pdb; pdb.set_trace()
            # outputs = model(input_ids=tensor_batch['input_ids'][:,:2048])
            # logits = outputs.logits
            # preds = torch.argmax(logits,dim=-1)
            # # print(tokenizer.decode(preds[0].tolist()))

            # metrics = calculate_metrics(logits, tensor_batch['input_ids'][:,:2048], tokenizer)

            # preds = tensor_batch['input_ids'][tensor_batch['input_ids']!=1]

            # code = tokenizer.decode(tensor_batch['input_ids'][0,:2048][:-1])
            
            # total_tokens += len(list(pygments.lex(code, lexer)))

            # total_ce += metrics['ce'].item()

            # # total_bleu += metrics['bleu']
            # # total_acc = metrics['acc']
            # ############################################################################

            input_ids = tensor_batch['input_ids'][:, :context_size]
            mask = mask[:, :context_size]

            try:
                outputs = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=max(0, tokens_amount-context_size))
                logits = outputs
                
                pred, label = decode(logits, tensor_batch['input_ids'], tokenizer, tokenizer.eod if args.is_replaced else tokenizer.eos_token_id, is_replaced=args.is_replaced)

                with open(f'generations/polycoder_replaced_{context_size}.jsonl', 'a+') as f:
                    sep = ' ' if args.is_replaced else ''
                    f.write(json.dumps({'label': sep.join(label),
                                        'pred': sep.join(pred)}) + '\n')
                
            except Exception as e:
                print(e)

            steps += 1

            if (batch_idx + 1) % batches_to_print == 0:
                ce_tensor = torch.tensor(total_ce)
                perplexity=torch.exp(ce_tensor/(total_tokens*LEX_VOCAB))
                print(total_ce , num_token)
                print(f'PPL = {perplexity} | {math.exp(total_ce/num_token)}')
                # print(f'BLEU = {total_bleu/steps} ') 
                # print(f'ACC = {total_acc/steps} ')
                print(f'AVG TOKENS = {num_token/steps}')
                
            progress_bar.update(1)

    # ce_tensor = torch.tensor(total_ce)
    # perplexity=torch.exp(ce_tensor/(total_tokens*LEX_VOCAB))
    # print(f'PPL = {perplexity}')

