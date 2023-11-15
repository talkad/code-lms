#!/bin/bash

python main.py      --task train                                                        \
                    --models-dir /home/talkad/LIGHTBITS_SHARE/hf_checkpoints            \
                    --model-name allc_tokom_700M                                        \
                    --device cuda