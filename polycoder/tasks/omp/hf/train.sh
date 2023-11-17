#!/bin/bash

python main.py      --do_finetune                                                       \
                    --models_dir /home/talkad/shared/models/hf_checkpoints              \
                    --model_name allc_tokom_700M                                        \
                    --device cuda                                                       \
                    --data_filename cpu_openmp_unique.jsonl                             \
                    --is_replaced                                                       \
                    --vocab_file /mnt/lbosm1/home/Share/code-lms/polycoder/tasks/tokenizer/tokompiler/tokenizer_vocab/vocab.txt \
                    --logger debug.log