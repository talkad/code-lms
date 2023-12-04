#!/bin/bash

python main.py      --do_finetune                                                       \
                    --models_dir /home/talkad/shared/models/hf_checkpoints              \
                    --batch_size 4                                                      \
                    --num_epochs 10                                                     \
                    --device cuda                                                       \
                    --data_filename train.jsonl                                         \
                    --tokenizer_type GPT2BPETokenizer                                   \
                    --logger bpe_orig.log                                               
