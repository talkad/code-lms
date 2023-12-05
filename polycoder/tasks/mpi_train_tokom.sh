#!/bin/bash

python main.py      --do_finetune                                                       \
		    --task mpi                                                          \
                    --models_dir /home/nadavsc/shared/models/hf_checkpoints             \
                    --model_name allc_tokom_700M                                        \
                    --num_epochs 2                                                      \
                    --batch_size 13                                                     \
		    --max_length 512                                                    \
                    --device cuda                                                       \
                    --is_replaced                                                       \
                    --tokenizer_type Tokompiler                                         \
                    --vocab_file /mnt/lbosm1/home/Share/code-lms/polycoder/tasks/tokenizer/tokompiler/tokenizer_vocab/vocab.txt     \
                    --logger mpi_debug_tokom.log     \
		    --data_path /home/nadavsc/LIGHTBITS/mpiricalplus/dataset/dataset_saved/tokompiler/512

