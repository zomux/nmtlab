#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import importlib.util
import sys
sys.path.append(".")

from torch import optim

from nmtlab import MTTrainer, MTDataset
from nmtlab.models import RNMTPlusModel
from nmtlab.schedulers import AnnealScheduler
from nmtlab.decoding import BeamTranslator
from nmtlab.evaluation.moses_bleu import MosesBLEUEvaluator
from nmtlab.utils import OPTS

from argparse import ArgumentParser

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

train_corpus = "private/iwslt15_vien/iwslt15_train.truecased.bpe20k.vien"
test_corpus = "private/iwslt15_vien/iwslt15_tst2013.truecased.bpe20k.vien"
ref_path = "private/iwslt15_vien/iwslt15_tst2013.truecased.en"
src_vocab_path = "private/iwslt15_vien/iwslt15.truecased.bpe20k.vi.vocab"
tgt_vocab_path = "private/iwslt15_vien/iwslt15.truecased.bpe20k.en.vocab"

# Define data set
dataset = MTDataset(
    train_corpus, src_vocab_path, tgt_vocab_path,
    batch_size=OPTS.batchsz * OPTS.gpus)

# Define model
nmt = RNMTPlusModel(
    num_encoders=1, num_decoders=2,
    dataset=dataset, hidden_size=OPTS.hiddensz, embed_size=OPTS.embedsz, label_uncertainty=0.1)

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

# Testing and Evaluation phase is only for the master node
if importlib.util.find_spec("horovod") is not None:
    import horovod.torch as hvd
    hvd.init()
    if hvd.local_rank() != 0:
        raise SystemExit

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

# Evaluation phase
if OPTS.evaluate or OPTS.all:
    evaluator = MosesBLEUEvaluator(ref_path)
    print("[tokenized BLEU]")
    print(evaluator.evaluate(OPTS.result_path))

