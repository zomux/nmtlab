#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")

from argparse import ArgumentParser
from nmtlab.utils import Vocab

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("-c", "--corpus", help="corpus path")
    ap.add_argument("-v", "--vocab", help="output path of the vocabulary")
    ap.add_argument("-l", "--limit", type=int, default=0, help="limit of maximum number of vocabulary items")
    args = ap.parse_args()
    
    vocab = Vocab()
    vocab.build(args.corpus, limit=args.limit)
    vocab.save(args.vocab)
