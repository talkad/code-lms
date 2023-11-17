import os
import torch
import logging
import hf_data_omp as data_omp
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, Trainer, TrainingArguments
from tqdm.auto import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

logger = logging.getLogger()


from tokenizer import TokompilerTokenizer
tokenizer = TokompilerTokenizer(vocab_path='/mnt/lbosm1/home/Share/code-lms/polycoder/tasks/tokenizer/tokompiler/tokenizer_vocab/vocab.txt')
tokenizer.add_tokens(['private', 'reduction', 'simd'])
tokenizer.enable_padding(length=252)


def preprocess_logits_for_metrics(logits, labels):
    logits = logits[0] 

    logits = logits.view([-1, logits.shape[-1]])
    labels = labels.view([-1])

    mask = labels != -100
    logits = logits[mask]
    labels = labels[mask]

    return cross_entropy(logits, labels, reduction='sum').reshape([1, 1])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print('aaa')
    logger.info(f'prediction: {predictions}\nlabel: {labels}\n\n')

    cross = eval_pred.predictions[0]
    return {'entropy': cross.sum()}


def finetune(args, model):
    logger.info('start finetune')

    # get data
    train, val, test = data_omp.build_omp_dataset(args)
    train_dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=args.batch_size, shuffle=True)

    # update model embeddings
    extended_tokens = ['private', 'reduction', 'simd']

    embedding_layer = model.get_input_embeddings()
    num_embeddings = embedding_layer.weight.shape[0]
    new_num_embeddings = num_embeddings+len(extended_tokens)
    model.resize_token_embeddings(new_num_embeddings)
    logger.info(f'Embedding layer has changed: {num_embeddings} -> {new_num_embeddings}')

    # train
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps,
                      weight_decay=args.weight_decay)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                                   num_training_steps=args.training_steps)


    model.to(args.device)
    model.train()

    # training_args = TrainingArguments(
    #     output_dir=args.save_dir,
    #     per_device_train_batch_size=args.batch_size,
    #     num_train_epochs=float('inf'),
    #     max_steps=args.training_steps,
    #     # num_train_epochs=args.epochs,
    #     # num_training_steps=args.training_steps,
    #     save_strategy="epoch",
    #     evaluation_strategy="epoch",
    #     logging_dir='./logs',
    #     eval_accumulation_steps=1,
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataloader,
    #     eval_dataset=test_dataloader,
    #     compute_metrics=compute_metrics,
    #     preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    #     optimizers=(optimizer, scheduler)
    # )

    # trainer.train()
    # # trainer.evaluate()



    

    progress_bar = tqdm(range(args.num_epochs * len(train_dataloader)))

    running_loss = 0.0  # Variable to accumulate the loss
    batches_to_print = 500  # Number of batches after which to print the mean loss

    
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            # import pdb; pdb.set_trace()
            tensor_batch = {k: v.to(args.device) for k, v in batch.items() if k in ['input_ids', 'labels']}
            outputs = model(**tensor_batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            if (batch_idx + 1) % batches_to_print == 0:
                mean_loss = running_loss / batches_to_print
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_dataloader)}, Mean Loss: {mean_loss:.4f}')
                running_loss = 0.0  # Reset running loss
                
                preds = torch.argmax(outputs['logits'][0], dim=-1)
                preds[preds > 1189] = 4
                logger.info(f"{batch['code'][0]}\n{tokenizer.detokenize(batch['labels'][0].tolist())}\n{tokenizer.detokenize(preds.tolist())}")

            progress_bar.update(1)

