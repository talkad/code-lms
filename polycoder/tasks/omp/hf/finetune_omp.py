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


def calculate_metrics(logits, labels):
    # import pdb; pdb.set_trace()
    y = labels
    logits = logits

    ce_loss = cross_entropy(logits, one_hot(y, num_classes=logits.shape[-1]).to(torch.float32), reduction='mean')

    # preds = torch.argmax(logits,dim=-1) 
    # accuracy = (preds==y).to(torch.float32).mean()

    # num_tokenized_tokens = len(y)
    # perplexity = torch.exp(ce_loss / num_tokenized_tokens)

    return {'loss': ce_loss}  # , 'accuracy': accuracy, 'perplexity': perplexity}


def finetune(args):
    logger.info(f'start finetune {args.model_name}')

    # get data

    tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B", torch_dtype=torch.float16,)

    # tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file, padding=True,
    #                           truncation=True, model_input_names=['input_ids'])
    # tokenizer.pad_token = tokenizer.eos_token


    datasets = data_omp.build_omp_dataset(args)

    newd = []
    for i in range(len(datasets)):
        d = datasets[i]
        outd = d.map(lambda examples: tokenizer(examples['code'], max_length=1024, truncation=True, padding=True), remove_columns=['pragma', 'code', 'hash'])
        newd.append(outd)
    traind, testd = newd
   
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_loader = DataLoader(dataset=traind, batch_size=args.batch_size, collate_fn=collator)

    # MODEL
    # model = GPTNeoXForCausalLM.from_pretrained(os.path.join(args.models_dir, args.model_name))        
    model.train()

    # freeze parameters model
    if args.freeze:
        freeze_layers = list(range(0, 5))
        for name, param in model.named_parameters():
            if any([f'layers.{layer}' in name for layer in freeze_layers]):      
                param.requires_grad = False

    for name, param in model.named_parameters():
        logger.info(f'param: {name} grad is {param.requires_grad}')

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
    # import pdb ; pdb.set_trace()

    # TRAIN
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, miniters=2, desc=f"Epoch {epoch}")
        loss_total = 0.0 

        for step, batch in enumerate(pbar):
            tensor_batch = {k: v.to(args.device) for k, v in batch.items() if k in ['input_ids', 'labels', 'mask', 'attention_mask']}

            outputs = model(**tensor_batch)
            # logits = outputs.logits

            # metrics = calculate_metrics(logits, tensor_batch['input_ids'])
            loss = outputs.loss # metrics['loss']

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loss_total += loss.detach().clone().item()

            if (step > 0) and (step % 100 == 0):
                print(f'{loss_total} / {step}')
                pbar.set_postfix({"avg_train_loss": loss_total / step})

        print('save model')
        model.save_pretrained(os.path.join(args.save_dir, 'poly_orig'), from_pt=True) 


    model.save_pretrained(os.path.join(args.save_dir, 'poly_orig'), from_pt=True) 

