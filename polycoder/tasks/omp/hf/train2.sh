#!/bin/bash

python main.py      --do_finetune                                                       \
                    --models_dir /home/talkad/LIGHTBITS_SHARE                           \
                    --model_name allc_gpt2tok_2-7B                                      \
                    --device cuda                                                       \
                    --data_filename cpu_openmp_unique.jsonl                             \
                    --tokenizer_type GPT2BPETokenizer                                   \
                    --logger debug.log


                                        # --vocab_file /mnt/lbosm1/home/Share/code-lms/polycoder/tasks/tokenizer/tokompiler/tokenizer_vocab/vocab.txt \
