#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchtext.vocab
import pickle
from collections import Counter, defaultdict


class Vocab(torchtext.vocab.Vocab):
    
    def __init__(self, path):
        self.tokens = pickle.load(open(path, "rb"), encoding='latin1')
        self.vectors = None
        self.freqs = Counter(self.tokens)
        self.itos = self.tokens
        self.stoi = defaultdict(lambda: 3)
        self.stoi.update({tok: i for i, tok in enumerate(self.tokens)})
    
    def size(self):
        return len(self.tokens)
