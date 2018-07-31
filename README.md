nmtlab - A PyTorch-based research framework for Neural Machine Translation

This is a framework to allow you to tweak EVERY part of NMT models. It is designed to be simple and clear, yet flexible.

# Installation

Create conda environment

```bash
conda create --name nmtlab python>3.6 --no-default-packages
conda activate nmtlab
```

Install pytorch, please check https://pytorch.org,

The command differs across CUDA versions, the default one is:

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


# Tutorial in Python

Raphael Shu, 2018.7

