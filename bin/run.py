#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os, sys
from torch import optim

from nmtlab import MTTrainer, MTDataset
from nmtlab.schedulers import AnnealScheduler
from nmtlab.utils import OPTS
from nmtlab.models import DeepLSTMModel, AttentionModel, FastDeepLSTMModel, RNMTPlusModel

from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("--resume", action="store_true")
ap.add_argument("--test", action="store_true")
ap.add_argument("--train", action="store_true")
ap.add_argument("--evaluate", action="store_true")
ap.add_argument("--all", action="store_true")
ap.add_argument("-d", "--dataset", type=str, default="dataset.json")
ap.add_argument("--opt_batchsz", type=int, default=128)
ap.add_argument("--opt_hiddensz", type=int, default=256)
ap.add_argument("--opt_embedsz", type=int, default=256)
ap.add_argument("--opt_clipnorm", type=float, default=0)
ap.add_argument("--opt_labelsmooth", type=float, default=0)
ap.add_argument("--opt_criteria", default="loss", type=str)
ap.add_argument("--opt_datatok", default="iwslt14_de", type=str)
ap.add_argument("--opt_optim", default="nestrov", type=str)
ap.add_argument("--opt_layernorm", action="store_true")
ap.add_argument("--opt_weightdecay", action="store_true")
ap.add_argument("--opt_finetune", action="store_true")
ap.add_argument("--opt_gpus", type=int, default=1)
ap.add_argument("--opt_model", default="attention")
ap.add_argument("--model_path",
                default="{}/data/torch_nmt/models/hv_nmt2.model".format(os.environ["HOME"]))
ap.add_argument("--result_path",
                default="{}/data/torch_nmt/results/hv_nmt2.result".format(os.environ["HOME"]))
OPTS.parse(ap)

if OPTS.dtok == "iwslt14_de":
    DATA_ROOT = "{}/data/torch_nmt/iwslt14".format(os.environ["HOME"])
    train_corpus = "{}/train.de-en.bpe20k".format(DATA_ROOT)
    test_corpus = "{}/test.de-en.bpe20k.tsv".format(DATA_ROOT)
    ref_path = "{}/test.en".format(DATA_ROOT)
    src_vocab_path = "{}/iwslt14.de.bpe20k.vocab".format(DATA_ROOT)
    tgt_vocab_path = "{}/iwslt14.en.bpe20k.vocab".format(DATA_ROOT)
    n_valid_per_epoch = 1
elif OPTS.dtok == "aspec_ej":
    OPTS.hiddensz = 1024
    OPTS.embedsz = 1024
    DATA_ROOT = "{}/data/torch_nmt/aspec".format(os.environ["HOME"])
    train_corpus = "{}/aspec.en-ja.bpe40k.tsv".format(DATA_ROOT)
    ref_path = "{}/aspec_test.case.en".format(DATA_ROOT)
    test_corpus = "{}/aspec_test.en-ja.bpe40k.tsv".format(DATA_ROOT)
    src_vocab_path = "{}/aspec.case.en.bpe40k.vocab".format(DATA_ROOT)
    tgt_vocab_path = "{}/aspec.ja.bpe40k.vocab".format(DATA_ROOT)
    n_valid_per_epoch = 4
else:
    raise NotImplementedError

dataset = MTDataset(
    train_corpus, src_vocab_path, tgt_vocab_path,
    batch_size=OPTS.batchsz * OPTS.gpus)

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

if OPTS.train or OPTS.all:
    # Training code
    scheduler = AnnealScheduler(patience=3, n_total_anneal=3, anneal_factor=OPTS.anneal)
    weight_decay = 1e-5 if OPTS.weightdecay else 0
    if OPTS.optim == "nestrov":
        optimizer = optim.SGD(nmt.parameters(), lr=0.25, momentum=0.99, nesterov=True, weight_decay=weight_decay)
    elif OPTS.optim == "adam":
        optimizer = optim.Adam(nmt.parameters(), lr=0.0001, weight_decay=weight_decay)
        if OPTS.model == "rnmt_plus":
            from nmtlab.schedulers import RNMTPlusAdamScheduler
            scheduler = RNMTPlusAdamScheduler()
    else:
        optimizer = optim.SGD(nmt.parameters(), lr=1.0, weight_decay=weight_decay)
    trainer = MTTrainer(nmt, dataset, optimizer, scheduler=scheduler, multigpu=OPTS.gpus > 1)
    trainer.configure(save_path=OPTS.model_path, n_valid_per_epoch=n_valid_per_epoch, criteria=OPTS.criteria, clip_norm=OPTS.clipnorm)
    if OPTS.resume:
        trainer.load()
    trainer.run()
if OPTS.test or OPTS.all:
    import horovod.torch as hvd
    hvd.init()
    if hvd.local_rank() != 0:
        raise SystemExit
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
    print(OPTS.result_path)
if OPTS.evaluate or OPTS.all:
    from nmtlab.evaluation.moses_bleu import MosesBLEUEvaluator
    import horovod.torch as hvd
    hvd.init()
    if hvd.local_rank() != 0:
        raise SystemExit
    evaluator = MosesBLEUEvaluator(ref_path)
    print(OPTS.result_path)
    print(evaluator.evaluate(OPTS.result_path))

