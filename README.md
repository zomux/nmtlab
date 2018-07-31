# nmtlab - A PyTorch-based framework for Neural Machine Translation research

This is a framework to allow you to tweak EVERY part of NMT models. It is designed to be simple and clear, yet flexible.

# Installation

Create conda environment

```bash
conda create --name nmtlab python>3.6 --no-default-packages
conda activate nmtlab
```

Install pytorch, please check https://pytorch.org,

The command depending on CUDA versions, the default one is:

```bash
conda install pytorch torchvision -c pytorch
```

Clone nmtlab repository
```bash
git clone --recurse-submodules https://github.com/zomux/nmtlab
cd nmtlab
```

Install other prerequisite packages:
```bash
pip install -r requirements.txt
```

(Option) Install horovod for multi-gpu support

Step 1. Install Open MPI

https://www.open-mpi.org/faq/?category=building#easy-build

Step 2. Install horovod through pypi
```bash
pip install horovod 
```

# Tutorial for using nmtlab

In this tutorial, we are going to train a NMT model on IWSLT15 Vietnam-English task, and evaluate it.

Make sure you are in the root directory of nmtlab repository. First, create the directory for experiment and download corpus:
```bash
mkdir ./private
bash scripts/download_iwslt15.sh
```

Preprocess the corpus 
(trucasing, subword segmentation and extracting vocabulary):
```bash
bash scripts/preprocess_iwslt15.sh
```

Create dataset configuration
```bash
cp examples/dataset.json.example private/dataset.json
```

Train a RNMT+ model and evaluate it
```bash
./bin/run.py -d private/dataset.json -tok iwslt15_vien --opt_gpus 1 --opt_model rnmt_plus --opt_weightdecay --train --test --evaluate
```

Instead of run the command with `--train --test --evaluate`, you can simply use `--all`.

(Option) Run the experiment with multiple GPUs

Make sure you have installed Open MPI and horovod, then use the following command to run with 4 GPUs:
```bash
mpirun -np 4 -H localhost:4 -bind-to none -map-by slot -x LD_LIBRARY_PATH -x PATH \
./bin/run.py -d private/dataset.json -tok iwslt15_vien --opt_gpus 4 --opt_model rnmt_plus --opt_weightdecay --all
```

# Using nmtlab in Python

# Design your NMT model



Raphael Shu, 2018.7

