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
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup


logger = logging.getLogger()

tokenizer = GPT2Tokenizer(vocab_file='../../../megatron/tokenizer/gpt_vocab/gpt2-vocab.json', merges_file='../../../megatron/tokenizer/gpt_vocab/gpt2-merges.txt', padding=True,
                              truncation=True, model_input_names=['input_ids'])
tokenizer.pad_token = tokenizer.eos_token



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

    # train, test = data_omp.build_omp_dataset(args)
    # test_dataloader = DataLoader(test, batch_size=1)

    # tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file, model_input_names=['input_ids'])
    # tokenizer.pad_token = tokenizer.eos_token
    
    datasets = data_omp.build_omp_dataset(args)
    pragmas = [sample['pragma'] for sample in datasets[1]]

    newd = []
    for i in range(len(datasets)):
        d = datasets[i]
        outd = d.map(lambda examples: tokenizer(examples['full'], max_length=2048, add_special_tokens=True, truncation=True, padding=True), remove_columns=['pragma', 'code', 'hash', 'full'])     
        newd.append(outd)

    traind, testd = newd

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    test_dataloader = DataLoader(dataset=testd, batch_size=1, collate_fn=collator, shuffle=False)
    # import pdb; pdb.set_trace()
    assert len(pragmas) == len(test_dataloader)

    # get model
    model = GPTNeoXForCausalLM.from_pretrained(os.path.join(args.save_dir, 'poly_check_vy'))

    model.to(args.device)
    model.eval()

    # # set tokenizer
    # if args.is_replaced:
    #     logger.info("set tokompiler tokenizer")
    #     tokenizer = TokompilerTokenizer(vocab_path='/mnt/lbosm1/home/Share/code-lms/polycoder/tasks/tokenizer/tokompiler/tokenizer_vocab/vocab.txt')
    #     tokenizer.add_tokens(['private', 'reduction', 'simd'])
    #     tokenizer.enable_padding(length=252)
    # else:
    #     logger.info("set BPE tokenizer")

    #     if args.vocab_file is None:
    #         lgger.info("WARNING: No vocab file found, loading Huggingface's pretrained GPT2Tokenizer")
    #     # tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    #     tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file)



    progress_bar = tqdm(range(len(test_dataloader)))

    pred_table = PrettyTable()
    pred_table.field_names = ["Label", "Pred"]
    pred_table.align["Label"] = "l"
    pred_table.align["Pred"] = "l"

    
    for batch_idx, batch in enumerate(test_dataloader):
        import pdb; pdb.set_trace()
        tensor_batch = {k: v.to(args.device) for k, v in batch.items() if k in ['input_ids', 'labels', 'mask', 'attention_mask']}

        outputs = model(**tensor_batch)
        logits = outputs.logits

        preds = torch.argmax(logits,dim=-1)
        preds = preds[preds!=50256]

        try:
            pred = tokenizer.decode(preds.tolist())
            label =  tokenizer.decode(tensor_batch['input_ids'][0])
        except:
            pred = ''

        pred_table.add_row([label[label.rfind('#pragma'):] if '#pragma' in label else 'None', 
                            pred[pred.rfind('#pragma'):] if '#pragma' in pred else 'None'])

        progress_bar.update(1)

    with open('reslts/aaaaaaaaaaa.log', 'w') as f:
        f.write(str(pred_table))

