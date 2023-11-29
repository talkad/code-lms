#!/bin/bash

python main.py      --do_test                                                          \
                    --models_dir /home/talkad/shared/models/hf_checkpoints              \
                    --batch_size 8                                                      \
                    --model_name allc_gpt2tok_700M                                      \
                    --num_epochs 3                                                      \
                    --device cuda                                                       \
                    --data_filename cpu_openmp_unique.jsonl                             \
                    --tokenizer_type GPT2BPETokenizer                                   \
                    --freeze                                                            \
                    --logger debug.log
