# CompCoder - Polycoder trained on HPCorpus

This is based heavily on the repo shared at: [https://github.com/VHellendoorn/Code-LMs/](https://github.com/VHellendoorn/Code-LMs/), using GPT-NeoX and DeepSpeed optimizations for speeding up pre-training.

Please refer to that repository for links to the Docker image, the original Polycoder checkpoints, and other such details.

## Tokompiler integration
This repository contains a novel tokenizer from another repo as a git submodule. When you initially setup this repo, please run `git submodule init` followed by `git submodule update` to setup the link, and pull the current status of that repo.

## Huggingface checkpoints
Our pretrained model checkpoints have been converted to Huggingface format.
An example of how load these checkpoints and the datasets for two downstream tasks are in the `test_hf_eval.ipynb` iPython notebook.

## Using the GPT-NeoX/Megatron libraries for training, fine-tuning, etc.
All of the configuration files are in `polycoder/configs`. As with the original repository, the model parameters have been included in a separate YAML file so they can be re-used with different scripts (e.g. same `small.yml` file for the small model would be used with training, fine-tuning, generation, etc.).

### Example fine-tuning command
The `deepy.py` file will initialize DeepSpeed, and then launch the file provided after it. Here this will launch the script to finetune a previously trained model on the downstream OpenMP task.

`sudo ./deepy.py tasks/omp/finetune_omp.py configs/ft_downstream.yml configs/700M.yml`

## TODOs
1. Accomodate expanded tokenizer vocabulary in fine-tuning. This will mean creating a model with a larger embedding layer and loading the checkpointed weights into the majority of that model.
