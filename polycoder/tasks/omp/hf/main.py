import os
import argparse
import torch
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig, AutoTokenizer

def main(args):

    # Load Model
    config = GPTNeoXConfig.from_pretrained(os.path.join(args.models_dir, args.model_name, 'config.json'))
    model = GPTNeoXForCausalLM(config=config)
    model.load_state_dict(torch.load(os.path.join(args.models_dir, args.model_name, 'pytorch_model.bin')))

    model_dict = model.state_dict()
    pretrained_dict = torch.load(os.path.join(args.models_dir, args.model_name, 'pytorch_model.bin'))

    print(model_dict)
    # Match layer names
    # new_pretrained_dict = {}
    # for k, v in pretrained_dict.items():
    #     if k in model_dict:
    #         print('ok', k)
    #         new_pretrained_dict[k] = v
    #     else:
    #         print('no', k)

    # Load modified state_dict
    model.load_state_dict(new_pretrained_dict, strict=False)

    print(f'----- model{args.model_name} loaded -----')

    if args.task == 'train':
        pass
    else:
        model.eval()
        pass


if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    parser.add_argument('--task',
                        default='train',
                        choices=['train', 'test'],
                        help='Specify the task (train or test)')

    parser.add_argument('--models-dir',
                        help='Specify the directory for models')

    parser.add_argument('--model-name',
                        help='Specify the model name')

    parser.add_argument('--device',
                        default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Specify the device (cpu or cuda)')

    main_args = parser.parse_args()
    main(main_args)