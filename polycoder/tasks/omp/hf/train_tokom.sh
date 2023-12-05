#!/bin/bash

python main.py      --do_test                                                      \
                    --models_dir /home/talkad/shared/models/hf_checkpoints              \
                    --model_name allc_tokom_700M                                        \
                    --num_epochs 4                                                      \
                    --batch_size 2                                                      \
                    --device cuda                                                       \
                    --data_filename cpu_openmp_unique.jsonl                             \
                    --is_replaced                                                       \
                    --tokenizer_type Tokompiler                                         \
                    --vocab_file /mnt/lbosm1/home/Share/code-lms/polycoder/tasks/tokenizer/tokompiler/tokenizer_vocab/vocab.txt     \
                    --logger debug_tokom.log


