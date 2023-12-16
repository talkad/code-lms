#!/bin/bash

python main.py      --do_eval                                                           \
                    --models_dir /home/talkad/shared/models/hf_checkpoints              \
                    --batch_size 1                                                     \
                    --model_name allc_gpt2tok_700M                                      \
                    --num_epochs 1                                                      \
                    --device cuda                                                      \
                    --data_filename cpu_openmp_unique.jsonl                             \
                    --tokenizer_type GPT2BPETokenizer                                   \
                    --data_path  /home/talkad/LIGHTBITS_SHARE                           \
                    --logger eval_debug.log
