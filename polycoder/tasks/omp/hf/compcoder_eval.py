import os
import torch
import logging
import hf_data_omp as data_omp
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, Trainer, TrainingArguments
from tqdm.auto import tqdm
from torch.nn.functional import cross_entropy, one_hot
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM
import pygments
from pygments.lexers import get_lexer_by_name
from tokenizer import TokompilerTokenizer, _GPT2BPETokenizer



logger = logging.getLogger()
lexer = get_lexer_by_name('c++')
LEX_VOCAB=sum([len(v) for v in lexer.tokens.values()])
logger.info(f'lex vocan amount is {LEX_VOCAB}')


def calculate_metrics(logits, labels):
    y = labels
    logits = logits

    ce = cross_entropy(logits, one_hot(y, num_classes=logits.shape[-1]).to(torch.float32), reduction='sum')
    my_ce = cross_entropy(logits, one_hot(y, num_classes=logits.shape[-1]).to(torch.float32), reduction='mean')

    return {'ce': ce, 'my_ce': my_ce}



def eval(args):
    logger.info(f'start compcoder HPC evaluation {args.model_name}')

    # get data
    train, test = data_omp.build_omp_dataset(args)
    test_dataloader = DataLoader(test, batch_size=args.batch_size, shuffle=True)


    # set tokenizer
    if args.is_replaced:
        logger.info("set tokompiler tokenizer")
        tokenizer = TokompilerTokenizer(vocab_path='/mnt/lbosm1/home/Share/code-lms/polycoder/tasks/tokenizer/tokompiler/tokenizer_vocab/vocab.txt')
        tokenizer.add_tokens(['private', 'reduction', 'simd'])
        tokenizer.enable_padding(length=252)
    else:
        logger.info("set BPE tokenizer")

        if args.vocab_file is None:
            lgger.info("WARNING: No vocab file found, loading Huggingface's pretrained GPT2Tokenizer")
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
   

    # get model
    model = GPTNeoXForCausalLM.from_pretrained(os.path.join(args.models_dir, args.model_name))        
    model.eval()
    
    # update model embeddings
    if args.is_replaced:
        extended_tokens = ['private', 'reduction', 'simd']

        embedding_layer = model.get_input_embeddings()
        num_embeddings = embedding_layer.weight.shape[0]
        new_num_embeddings = num_embeddings+len(extended_tokens)
        model.resize_token_embeddings(new_num_embeddings)
        logger.info(f'Embedding layer has changed: {num_embeddings} -> {new_num_embeddings}')

    model.to(args.device)

    # EVAL
    progress_bar = tqdm(range(args.num_epochs * len(test_dataloader)))

    total_ce, total_tokens = 0.0, 0
    my_ce, num_token = 0.0,  0
    batches_to_print = 100
    
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            
            tensor_batch = {k: v.to(args.device) for k, v in batch.items() if k in ['input_ids', 'labels', 'mask']}

            outputs = model(input_ids=tensor_batch['input_ids'])
            logits = outputs.logits

            metrics = calculate_metrics(logits, tensor_batch['labels'])
            
            total_tokens += sum([len(list(pygments.lex(code, lexer))) for code in batch['code']])
            num_token += sum([len(tokenizer.tokenize(code)) for code in batch['code']]) 

            # print(num_token)
            my_ce += metrics['my_ce'].item()
            total_ce += metrics['ce'].item()

            if (batch_idx + 1) % batches_to_print == 0:
                ce_tensor = torch.tensor(total_ce)
                perplexity=torch.exp(ce_tensor/(total_tokens*LEX_VOCAB))

                my_ce = torch.tensor(my_ce)
                my_ppl = torch.exp(my_ce/num_token)
                print(f'PPL = {perplexity} | {my_ppl}')
                
            progress_bar.update(1)

    ce_tensor = torch.tensor(total_ce)
    perplexity=torch.exp(ce_tensor/(total_tokens*LEX_VOCAB))
    print(f'PPL = {perplexity}')

