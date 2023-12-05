#!/bin/bash

python main.py      --do_eval                                                           \
		    --task omp                                                          \
                    --models_dir /home/talkad/shared/models/hf_checkpoints              \
                    --batch_size 1                                                      \
		    --max_length 2048                                                   \
                    --model_name allc_gpt2tok_700M                                      \
                    --num_epochs 1                                                      \
                    --device cuda                                                       \
                    --data_filename cpu_openmp_unique.jsonl                             \
                    --vocab_file /mnt/lbosm1/home/Share/code-lms/polycoder/tasks/tokenizer/tokompiler/tokenizer_vocab/vocab.txt     \
                    --tokenizer_type GPT2BPETokenizer                                   \
                    --logger omp_eval_debug.log
