# nmtlab - A PyTorch-based Neural Machine Translation research framework for research purpose

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
./bin/run.py -d private/dataset.json -t iwslt15_vien --opt_gpus 1 --opt_model rnmt_plus --opt_weightdecay --train --test --evaluate
```

Instead of run the command with `--train --test --evaluate`, you can simply use `--all`.

(Option) Run the experiment with multiple GPUs

Make sure you have installed Open MPI and horovod, then use the following command to run with 4 GPUs:
```bash
mpirun -np 4 -H localhost:4 -bind-to none -map-by slot -x LD_LIBRARY_PATH -x PATH \
./bin/run.py -d private/dataset.json -t iwslt15_vien --opt_gpus 4 --opt_model rnmt_plus --opt_weightdecay --all
```

I got a BLEU score of 21.99 in my 4 GPU environment.

# Using nmtlab in Python

nmtlab is designed to be directly used in Python. Here is an example code in `examples/rnmt_experiment.py`.

First, we import basic packages:
```python
from torch import optim

from nmtlab import MTTrainer, MTDataset
from nmtlab.models import RNMTPlusModel
from nmtlab.schedulers import AnnealScheduler
from nmtlab.decoding import BeamTranslator
from nmtlab.evaluation.moses_bleu import MosesBLEUEvaluator
from nmtlab.utils import OPTS

from argparse import ArgumentParser
```

Next we define the options, note that all options with `opt_` prefix will be appended to the name of model file and result file. Options with "opt_T" prefix are considered as hyperparameter for testing phase, which will be only appended to the filenames of translation results.

```python
ap = ArgumentParser()
# Main commands
ap.add_argument("--train", action="store_true", help="training")
ap.add_argument("--resume", action="store_true", help="resume training")
ap.add_argument("--test", action="store_true", help="testing")
ap.add_argument("--evaluate", action="store_true", help="evaluate tokenized BLEU")
ap.add_argument("--all", action="store_true", help="run all phases")
# Model options
ap.add_argument("--opt_hiddensz", type=int, default=256, help="hidden size")
ap.add_argument("--opt_embedsz", type=int, default=256, help="embedding size")
# Training options
ap.add_argument("--opt_gpus", type=int, default=1, help="number of GPU")
ap.add_argument("--opt_batchsz", type=int, default=64, help="batch size")
# Testing options
ap.add_argument("--opt_Tbeam", type=int, default=3, help="beam size")

# Path configs
ap.add_argument("--model_path",
                default="private/example.nmtmodel", help="path of checkpoint")
ap.add_argument("--result_path",
                default="private/example.result", help="path of translation result")
OPTS.parse(ap)
```

Next, we define the dataset.
```python
train_corpus = "private/iwslt15_vien/iwslt15_train.truecased.bpe20k.vien"
test_corpus = "private/iwslt15_vien/iwslt15_tst2013.truecased.bpe20k.vien"
ref_path = "private/iwslt15_vien/iwslt15_tst2013.truecased.en"
src_vocab_path = "private/iwslt15_vien/iwslt15.truecased.bpe20k.vi.vocab"
tgt_vocab_path = "private/iwslt15_vien/iwslt15.truecased.bpe20k.en.vocab"

# Define data set
dataset = MTDataset(
    train_corpus, src_vocab_path, tgt_vocab_path,
    batch_size=OPTS.batchsz * OPTS.gpus)
```

Then we create the NMT model.
```python
nmt = RNMTPlusModel(
    num_encoders=1, num_decoders=2,
    dataset=dataset, hidden_size=OPTS.hiddensz, embed_size=OPTS.embedsz, label_uncertainty=0.1)
```

In the training phase, we choose a PyTorch optimizer and training scheduler, and train the model.
```python
# Training phase
if OPTS.train or OPTS.all:
    # Define optimizer and scheduler
    optimizer = optim.SGD(nmt.parameters(), lr=0.25, momentum=0.99, nesterov=True, weight_decay=1e-5)
    scheduler = AnnealScheduler(patience=3, n_total_anneal=3, anneal_factor=10)
    
    # Define trainer
    trainer = MTTrainer(nmt, dataset, optimizer, scheduler=scheduler, multigpu=OPTS.gpus > 1)
    trainer.configure(save_path=OPTS.model_path, n_valid_per_epoch=1, criteria="bleu", clip_norm=0.1)
    if OPTS.resume:
        trainer.load()
    trainer.run()
```

Next, we translate all the sentences in the test corpus.
```python
    
# Testing phase
if OPTS.test or OPTS.all:
    print("[testing]")
    nmt.load(OPTS.model_path)
    fout = open(OPTS.result_path, "w")
    translator = BeamTranslator(nmt, dataset.src_vocab(), dataset.tgt_vocab(), beam_size=OPTS.Tbeam)
    for line in open(test_corpus):
        src_sent, _ = line.strip().split("\t")
        result, _ = translator.translate("<s> {} </s>".format(src_sent))
        if result is None:
            result = ""
        result = result.replace("@@ ", "")
        fout.write(result + "\n")
        sys.stdout.write("." if result else "x")
        sys.stdout.flush()
    sys.stdout.write("\n")
    fout.close()
    print("[result path]")
    print(OPTS.result_path)
```

Finally, here are the codes for evaluate the BLEU score.
```python
# Evaluation phase
if OPTS.evaluate or OPTS.all:
    evaluator = MosesBLEUEvaluator(ref_path)
    print("[tokenized BLEU]")
    print(evaluator.evaluate(OPTS.result_path))
```

Please check and use the full code in `examples/rnmt_experiment.py`. Note that in this example, we are not using per-gate layer normalization and dynamic Adam Scheduling, which may affect the final model performance.

The code can be run with the following command:
```bash
python ./examples/rnmt_experiment.py --opt_gpus 1 --all
```
or
```bash
mpirun -np 4 -H localhost:4 -bind-to none -map-by slot -x LD_LIBRARY_PATH -x PATH \
python ./examples/rnmt_experiment.py --opt_gpus 4 --all
```

# Design your NMT model

A customized NMT model can be defined in following structure:
```python
from nmtlab.models import EncoderDecoderModel

class ExampleModel(EncoderDecoderModel):
    
    def prepare(self):
        """
        Initalize layers for the NMT models.
        """
        # Set the names of decoder states
        self.set_states(
            ["hidden1", "cell1"],
            [self._hidden_size, self._hidden_size])
        # Choose whether to decode the sequence in stepwise fashion in the training time.
        # This shall be set to False when using Cudnn LSTM API or transformer.
        # When this flag is False you have to implement the 'decode_step' function for both scenarios.
        self.set_stepwise_training(False)
    
    def encode(self, src_seq, src_mask=None):
        """
        Encode the input sequence and output encoder states.
        """
    
    def lookup_feedback(self, feedback):
        """
        Return the embeddings for target-side tokens.
        """
    
    def decode_step(self, context, states, full_sequence=False):
        """
        Produce decoder states given encoder context.
        Args:
            context - encoder states and feedback embeddings
            states - decoder states in previous step.
            full_sequence - whether to produce states for one step or for a full sequence
        """
        if full_sequence:
            # Compute the states in full sequence mode.
        else:
            # Update the decoder states in every step.
    
    def expand(self, states):
        """
        Compute the softmax logits given final decoder states.
        """
```

Please check `examples/example_model.py` for an example model implementation.

Raphael Shu, 2018.7

