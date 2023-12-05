#!/bin/bash

python main.py      --do_finetune                                                       \
		    --task omp                                                          \
                    --models_dir /home/talkad/shared/models/hf_checkpoints              \
                    --batch_size 4                                                      \
		    --max_length 2048                                                   \
                    --model_name allc_gpt2tok_700M                                      \
                    --num_epochs 10                                                     \
                    --device cuda                                                       \
                    --data_filename cpu_openmp_unique.jsonl                             \
                    --tokenizer_type GPT2BPETokenizer                                   \
                    --logger bpe_orig.log                                               \
                    --freeze                                                            \
