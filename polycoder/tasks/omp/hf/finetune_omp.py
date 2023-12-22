import os
import torch
import logging
import hf_data_omp as data_omp
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, Trainer, TrainingArguments
from tqdm.auto import tqdm
from torch.nn.functional import cross_entropy, one_hot
from torch.utils.data import DataLoader, Dataset
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizer import TokompilerTokenizer

from torch.cuda.amp import autocast

logger = logging.getLogger()


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




def finetune(args):
    
    logger.info(f'start finetune {args.model_name}')

    # TOKENIZER
    tokom_extended_tokens = ['parallel', 'private', 'reduction']

    if args.is_replaced:
        tokenizer = TokompilerTokenizer(vocab_path=args.vocab_file)
        tokenizer.add_tokens(tokom_extended_tokens)
        tokenizer.enable_padding(length=2048)
    else:
        tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file, padding=True,
                                truncation=True, model_input_names=['input_ids'])
        tokenizer.pad_token = tokenizer.eos_token


    # DATA
    datasets = data_omp.build_omp_dataset(args)

    newd = []
    for i in range(len(datasets)):
        d = datasets[i]
        outd = d.map(lambda examples: tokenize(args, tokenizer, examples), remove_columns=['pragma', 'code', 'hash', 'full'])     
        newd.append(outd)

    traind, testd = newd

    if args.is_replaced:

        train_data = []
        for ids, labels in tqdm(zip(traind['input_ids'], traind['labels'])):
            train_data.append({'input_ids': ids, 'labels': labels})

        test_data = []
        for ids, labels in tqdm(zip(testd['input_ids'], testd['labels'])):
            test_data.append({'input_ids': ids, 'labels': labels})

        train_loader = DataLoader(dataset=CustomDataset(train_data), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=CustomDataset(test_data), batch_size=args.batch_size)
    else:
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        train_loader = DataLoader(dataset=traind, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
        test_loader = DataLoader(dataset=testd, batch_size=args.batch_size, collate_fn=collator)


    # MODEL
    model = GPTNeoXForCausalLM.from_pretrained(os.path.join(args.models_dir, args.model_name))        
    model.train()

    # update model embeddings
    if args.is_replaced:
        embedding_layer = model.get_input_embeddings()
        num_embeddings = embedding_layer.weight.shape[0]
        new_num_embeddings = num_embeddings+len(tokom_extended_tokens)
        model.resize_token_embeddings(new_num_embeddings)
        logger.info(f'Embedding layer has changed: {num_embeddings} -> {new_num_embeddings}')


    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps,
                      weight_decay=args.weight_decay)

    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=100,
                                                   num_training_steps=(len(train_loader) * args.num_epochs),)
    
    model.to(args.device)

    # TRAIN
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, miniters=2, desc=f"Epoch {epoch}")
        loss_total = 0.0 

        for step, batch in enumerate(train_loader):
            tensor_batch = {k: v.to(args.device) for k, v in batch.items() if k in ['input_ids', 'labels', 'mask', 'attention_mask']}
            
            with autocast():
                outputs = model(**tensor_batch)
                loss = outputs.loss 

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loss_total += loss.detach().clone().item()

            if (step > 0) and (step % 10 == 0):
                logger.info(f'loss: {loss_total / (step+1)}')
                pbar.set_postfix({"avg_train_loss": loss_total / step})
                
        print(f'save checkpoint {epoch}')
        model.save_pretrained(os.path.join(args.save_dir, 'compcoder'), from_pt=True) 

