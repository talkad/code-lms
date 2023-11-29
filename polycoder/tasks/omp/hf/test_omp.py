import os
import torch
import logging
import hf_data_omp as data_omp
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm.auto import tqdm
from torch.nn.functional import cross_entropy, one_hot
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM
from prettytable import PrettyTable
from tokenizer import TokompilerTokenizer, _GPT2BPETokenizer



logger = logging.getLogger()




def concat_vars(pragma):
    unified_vars = []
    tokens = pragma.split()

    for idx, token in enumerate(tokens):
        if token.isnumeric():
            continue

        if token in ['var', 'arr', 'struct', 'arg'] and idx < len(tokens) - 1 and tokens[idx + 1].isnumeric():
            unified_vars.append(f'{token}_{tokens[idx + 1]}')
        else:
            unified_vars.append(token)

    return ' '.join(unified_vars)


def test(args):
    logger.info('start test')

    train, test = data_omp.build_omp_dataset(args)
    test_dataloader = DataLoader(test, batch_size=1)

    # get model
    model = GPTNeoXForCausalLM.from_pretrained(os.path.join(args.save_dir, 'poly_bpe_vy'))

    model.to(args.device)
    model.eval()

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



    progress_bar = tqdm(range(len(test_dataloader)))

    pred_table = PrettyTable()
    pred_table.field_names = ["Label", "Pred"]
    pred_table.align["Label"] = "l"
    pred_table.align["Pred"] = "l"

    
    for batch_idx, batch in enumerate(test_dataloader):
        # import pdb; pdb.set_trace()
        tensor_batch = {k: v.to(args.device) for k, v in batch.items() if k in ['input_ids', 'labels', 'mask']}
        # mask = tensor_batch['mask']
        # labels = tensor_batch['labels']
        outputs = model(input_ids=tensor_batch['input_ids'])
        logits = outputs.logits

        # logits = logits[mask==1][:-2]
        preds = torch.argmax(logits,dim=-1)
        preds = preds[preds!=50256]

        try:
            pred = tokenizer.detokenize(preds.tolist())
        except:
            pred = ''

        pred_table.add_row([batch['pragma'][0], 
                            pred[pred.rfind('\n')+1:]])

       
        progress_bar.update(1)

    with open('poly_bpe_vy.log', 'w') as f:
        f.write(str(pred_table))

