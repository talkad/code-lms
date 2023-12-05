#!/bin/bash

python main.py      --do_finetune                                                       \
		    --task omp                                                          \
                    --models_dir /home/talkad/shared/models/hf_checkpoints              \
                    --batch_size 4                                                      \
		    --max_length 2048                                                   \
                    --num_epochs 10                                                     \
                    --device cuda                                                       \
                    --data_filename train.jsonl                                         \
                    --tokenizer_type GPT2BPETokenizer                                   \
                    --logger omp_bpe_orig.log                                               
