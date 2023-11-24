import os
import torch
import logging
import hf_data_omp as data_omp
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, Trainer, TrainingArguments
from tqdm.auto import tqdm
from torch.nn.functional import cross_entropy, one_hot
from torch.utils.data import DataLoader


logger = logging.getLogger()


def calculate_metrics(logits, labels, mask):
    # import pdb; pdb.set_trace()
    # TODO: add PPL xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # TODO: evaluate with GPT 3.5 turbo

    # 1. hyper param v / attention tuning v / PPL v /
    # 2. BPE 
    # 3. GPT3.5 (Pragformers question) - test only
    # 4. tokom preprocess

    y = labels[mask==1][1:]
    logits = logits[mask==1][:-1]

    ce_loss = cross_entropy(logits, one_hot(y, num_classes=logits.shape[-1]).to(torch.float32))

    preds = torch.argmax(logits,dim=-1) 
    accuracy = (preds==y).to(torch.float32).mean()

    num_tokenized_tokens = len(y)
    perplexity = torch.exp(ce_loss / num_tokenized_tokens)

    return {'loss': ce_loss, 'accuracy': accuracy, 'perplexity': perplexity}



def finetune(args, model):
    logger.info('start finetune')

    # get data
    train, val, test = data_omp.build_omp_dataset(args)
    train_dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=args.batch_size, shuffle=True)

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
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                                   num_training_steps=args.training_steps)

    model.to(args.device)
    
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.train()


    # TRAIN
    progress_bar = tqdm(range(args.num_epochs * len(train_dataloader)))

    running_loss, running_acc, running_ppl = 0.0, 0.0, 0.0
    batches_to_print = 50 

    
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            # import pdb; pdb.set_trace()
            tensor_batch = {k: v.to(args.device) for k, v in batch.items() if k in ['input_ids', 'labels', 'mask']}

            # ##### TRY ATTENTION #####
            # index_of_one = (tensor_batch['mask'] == 1).nonzero(as_tuple=True)[0]

            # attention_mask = torch.zeros_like(tensor_batch['mask'])
            # if len(index_of_one) > 0:
            #     attention_mask[:index_of_one[0]] = 1
            # #########################

            outputs = model(input_ids=tensor_batch['input_ids'])
            logits = outputs.logits

            metrics = calculate_metrics(logits, tensor_batch['labels'], tensor_batch['mask'])
            loss = metrics['loss']

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            running_acc += metrics['accuracy'].item()
            running_ppl += metrics['perplexity'].item()

            if (batch_idx + 1) % batches_to_print == 0:
                mean_loss = running_loss / batches_to_print
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_dataloader)}, Mean Loss: {mean_loss:.4f}, mean acc {running_acc/batches_to_print:.4f}, mean PPL {running_ppl/batches_to_print:.4f}')
                running_loss, running_acc, running_ppl = 0.0, 0.0, 0.0

            progress_bar.update(1)

    model.save_pretrained(os.path.join(args.save_dir, 'poly_tokom_attention'), from_pt=True) 

