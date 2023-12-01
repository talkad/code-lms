import os
import torch
import logging
import hf_data_omp as data_omp
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, Trainer, TrainingArguments
from tqdm.auto import tqdm
from torch.nn.functional import cross_entropy, one_hot
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger()


tokenizer = GPT2Tokenizer(vocab_file='../../../megatron/tokenizer/gpt_vocab/gpt2-vocab.json', merges_file='../../../megatron/tokenizer/gpt_vocab/gpt2-merges.txt', padding=True,
                              truncation=True, model_input_names=['input_ids'])
tokenizer.pad_token = tokenizer.eos_token


def finetune(args):
    
    logger.info(f'start finetune {args.model_name}')

    # DATA
    datasets = data_omp.build_omp_dataset(args)
    
    newd = []
    for i in range(len(datasets)):
        d = datasets[i]
        outd = d.map(lambda examples: tokenizer(examples['full'], max_length=2048, add_special_tokens=True, truncation=True, padding=True), remove_columns=['pragma', 'code', 'hash', 'full'])     
        newd.append(outd)

    traind, testd = newd
   
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_loader = DataLoader(dataset=traind, batch_size=args.batch_size, collate_fn=collator)
    test_loader = DataLoader(dataset=testd, batch_size=args.batch_size, collate_fn=collator)


    # MODEL
    model = GPTNeoXForCausalLM.from_pretrained(os.path.join(args.models_dir, args.model_name))        
    model.train()

    # update model embeddings
    if args.is_replaced:
        extended_tokens = ['private', 'reduction', 'simd']

        embedding_layer = model.get_input_embeddings()
        num_embeddings = embedding_layer.weight.shape[0]
        new_num_embeddings = num_embeddings+len(extended_tokens)
        model.resize_token_embeddings(new_num_embeddings)
        logger.info(f'Embedding layer has changed: {num_embeddings} -> {new_num_embeddings}')


    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps,
                      weight_decay=args.weight_decay)

    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=100,
                                                   num_training_steps=(len(train_loader) * args.num_epochs),)
    
    model.to(args.device)
    # import pdb; pdb.set_trace()

    # TRAIN
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, miniters=2, desc=f"Epoch {epoch}")
        loss_total = 0.0 

        for step, batch in enumerate(pbar):
            tensor_batch = {k: v.to(args.device) for k, v in batch.items() if k in ['input_ids', 'labels', 'mask', 'attention_mask']}

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

        # VALIDATION       
        # val_loss = 0.0
        # for step_val, batch_val in enumerate(test_loader):
        #     tensor_batch = {k: v.to(args.device) for k, v in batch_val.items() if k in ['input_ids', 'labels', 'mask', 'attention_mask']}

        #     outputs = model(**tensor_batch)
        #     loss = outputs.loss 
        #     val_loss += loss.detach().clone().item()
        # logger.info(f'val loss:  {val_loss / (step_val+1)}')
                

        print('save model')
        model.save_pretrained(os.path.join(args.save_dir, 'poly_bpe'), from_pt=True) 

    model.save_pretrained(os.path.join(args.save_dir, 'poly_bpe'), from_pt=True) 

