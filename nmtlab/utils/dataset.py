#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchtext
import numpy as np
from nmtlab.utils.vocab import Vocab


class MTDataset(object):
    """Bilingual dataset.
    """
    
    def __init__(self, corpus_path, src_vocab_path, tgt_vocab_path, batch_size=64, max_length=60, n_valid_batch=50):
        self._max_length = max_length
        self._batch_size = batch_size
        self._n_valid_batch = n_valid_batch
        
        src = torchtext.data.Field(pad_token="<null>", preprocessing=lambda seq: ["<s>"] + seq + ["</s>"])
        self._src_vocab = src.vocab = Vocab(src_vocab_path)
        tgt = torchtext.data.Field(pad_token="<null>", preprocessing=lambda seq: ["<s>"] + seq + ["</s>"])
        self._tgt_vocab = tgt.vocab = Vocab(tgt_vocab_path)
        # Make data
        self._data = torchtext.data.TabularDataset(
            path=corpus_path, format='tsv',
            fields=[('src', src), ('tgt', tgt)],
            filter_pred=self._len_filter
        )
        # Create training and valid dataset
        examples = self._data.examples
        n_valid_samples = self._batch_size * n_valid_batch
        n_train_samples = len(examples) - n_valid_samples
        n_train_samples = int(n_train_samples / self._batch_size) * self._batch_size
        np.random.RandomState(3).shuffle(examples)
        self._valid_data = torchtext.data.Dataset(
            examples[:n_valid_samples],
            fields=[('src', src), ('tgt', tgt)],
            filter_pred=self._len_filter
        )
        self._train_data = torchtext.data.Dataset(
            examples[n_valid_samples:n_valid_samples + n_train_samples],
            fields=[('src', src), ('tgt', tgt)],
            filter_pred=self._len_filter
        )

    def set_gpu_scope(self, scope_index, n_scopes):
        """Training a specific part of data for multigpu environment.
        """
        examples = self._train_data.examples
        scope_size = int(float(len(examples)) / n_scopes)
        self._train_data.examples = examples[scope_index * scope_size: (scope_index + 1) * scope_size]
        self._batch_size = self._batch_size / n_scopes
        
    @staticmethod
    def _len_filter(sample):
        return len(sample.src) <= 60 and len(sample.tgt) <= 60
    
    def n_train_batch(self):
        return int(len(self._train_data.examples) / self._batch_size)
    
    def train_set(self):
        batch_iterator = torchtext.data.BucketIterator(
            dataset=self._train_data, batch_size=self._batch_size,
            sort=False, sort_within_batch=True,
            shuffle=True,
            sort_key=lambda x: len(x.src),
            device=None, repeat=False)
        return iter(batch_iterator)
    
    def valid_set(self):
        batch_iterator = torchtext.data.BucketIterator(
            dataset=self._valid_data, batch_size=self._batch_size,
            sort=True, sort_within_batch=True,
            shuffle=False, train=False,
            sort_key=lambda x: len(x.src),
            device=None, repeat=False)
        return batch_iterator
        
    def src_vocab(self):
        return self._src_vocab
    
    def tgt_vocab(self):
        return self._tgt_vocab
    
    def vocab_sizes(self):
        return self._src_vocab.size(), self._tgt_vocab.size()
    
    def batch_size(self):
        return self._batch_size
