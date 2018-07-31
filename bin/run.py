#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os, sys
sys.path.append(".")

import json
import importlib.util
from torch import optim

from nmtlab import MTTrainer, MTDataset
from nmtlab.schedulers import AnnealScheduler
from nmtlab.utils import OPTS
from nmtlab.models import DeepLSTMModel, AttentionModel, FastDeepLSTMModel, RNMTPlusModel

from argparse import ArgumentParser

ap = ArgumentParser()
# Main commands
ap.add_argument("--train", action="store_true", help="training")
ap.add_argument("--resume", action="store_true", help="resume training")
ap.add_argument("--test", action="store_true", help="testing")
ap.add_argument("--evaluate", action="store_true", help="evaluate tokenized BLEU")
ap.add_argument("--all", action="store_true", help="run all phases")
# Dataset config path
ap.add_argument("-d", "--dataset", type=str, default="private/dataset.json", help="dataset config")
ap.add_argument("-t", "--opt_tok", default="none", type=str, help="datapair token")
# Model options
ap.add_argument("--opt_model", default="attention", help="model name")
ap.add_argument("--opt_hiddensz", type=int, default=256, help="hidden size")
ap.add_argument("--opt_embedsz", type=int, default=256, help="embedding size")
ap.add_argument("--opt_labelsmooth", type=float, default=0.1, help="uncertainty of label smoothing")
ap.add_argument("--opt_layernorm", action="store_true", help="whether using layer normalization")
ap.add_argument("--opt_weightdecay", action="store_true", help="whether using weight decay")
# Training options
ap.add_argument("--opt_optim", default="nestrov", type=str, help="optimizer")
ap.add_argument("--opt_gpus", type=int, default=1, help="number of GPU, must use mpirun in multi-gpu training")
ap.add_argument("--opt_batchsz", type=int, default=64, help="batch size")
ap.add_argument("--opt_clipnorm", type=float, default=0.1, help="max norm of gradient clipping")
ap.add_argument("--opt_criteria", default="bleu", type=str, help="criteria for model selection, must be 'bleu' or 'loss'")
# Path configs
ap.add_argument("--model_path",
                default="private/nmtlab.nmtmodel", help="path for saving checkpoint")
ap.add_argument("--result_path",
                default="private/nmtlab.result", help="path of translation result")
OPTS.parse(ap)

# Load datatset configuration
if not os.path.exists(OPTS.dataset):
    raise SystemError("Can not find dataset config in {}".format(OPTS.dataset))
dataset_config = json.load(open(OPTS.dataset, encoding="utf-8"))
if OPTS.tok not in dataset_config:
    raise SystemError("Data token {} is not in the dataset".format(OPTS.tok))

datapair_config = dataset_config[OPTS.tok]
train_corpus = datapair_config["train_corpus"]
test_corpus = datapair_config["test_corpus"]
ref_path = datapair_config["reference_path"]
src_vocab_path = datapair_config["src_vocab_path"]
tgt_vocab_path = datapair_config["tgt_vocab_path"]
n_valid_per_epoch = datapair_config["n_valid_per_epoch"]

# Define data set
dataset = MTDataset(
    train_corpus, src_vocab_path, tgt_vocab_path,
    batch_size=OPTS.batchsz * OPTS.gpus)

# Define NMT model
kwargs = dict(
    dataset=dataset, hidden_size=OPTS.hiddensz, embed_size=OPTS.embedsz,
    label_uncertainty=OPTS.labelsmooth
)
if OPTS.model == "attention":
    nmt = AttentionModel(**kwargs)
elif OPTS.model == "deep_lstm":
    nmt = DeepLSTMModel(**kwargs)
elif OPTS.model == "fast_deep_lstm":
    nmt = FastDeepLSTMModel(**kwargs)
elif OPTS.model == "rnmt_plus":
    kwargs["layer_norm"] = OPTS.layernorm
    nmt = RNMTPlusModel(**kwargs)
else:
    raise NotImplementedError

# Training phase
if OPTS.train or OPTS.all:
    # Define optimizer and scheduler
    weight_decay = 1e-5 if OPTS.weightdecay else 0
    scheduler = AnnealScheduler(patience=3, n_total_anneal=3, anneal_factor=10)
    if OPTS.optim == "nestrov":
        optimizer = optim.SGD(nmt.parameters(), lr=0.25, momentum=0.99, nesterov=True, weight_decay=weight_decay)
    elif OPTS.optim == "adam":
        optimizer = optim.Adam(nmt.parameters(), lr=0.0001, weight_decay=weight_decay)
        if OPTS.model == "rnmt_plus":
            from nmtlab.schedulers import RNMTPlusAdamScheduler
            scheduler = RNMTPlusAdamScheduler()
    else:
        raise NotImplementedError
    
    # Define trainer
    trainer = MTTrainer(nmt, dataset, optimizer, scheduler=scheduler, multigpu=OPTS.gpus > 1)
    trainer.configure(save_path=OPTS.model_path, n_valid_per_epoch=n_valid_per_epoch, criteria=OPTS.criteria, clip_norm=OPTS.clipnorm)
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
    from nmtlab.decoding import BeamTranslator
    nmt.load(OPTS.model_path)
    fout = open(OPTS.result_path, "w")
    translator = BeamTranslator(nmt, dataset.src_vocab(), dataset.tgt_vocab(), beam_size=3)
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
    from nmtlab.evaluation.moses_bleu import MosesBLEUEvaluator
    evaluator = MosesBLEUEvaluator(ref_path)
    print("[tokenized BLEU]")
    print(evaluator.evaluate(OPTS.result_path))

