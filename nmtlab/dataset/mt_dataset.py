#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchtext
import numpy as np
from nmtlab.utils.vocab import Vocab
from nmtlab.dataset.bilingual_dataset import BilingualDataset
from nmtlab.dataset.base import Dataset


class MTDataset(Dataset):
    """Bilingual dataset.
    """
    
    def __init__(self, corpus_path=None, src_corpus=None, tgt_corpus=None, src_vocab=None, tgt_vocab=None, batch_size=64, max_length=60, n_valid_samples=1000, truncate=None):
        
        assert corpus_path is not None or (src_corpus is not None and tgt_corpus is not None)
        assert src_vocab is not None and tgt_vocab is not None
    
        self._max_length = max_length
        self._n_valid_samples = n_valid_samples
        
        src = torchtext.data.Field(pad_token="<null>", preprocessing=lambda seq: ["<s>"] + seq + ["</s>"])
        self._src_vocab = src.vocab = Vocab(src_vocab)
        tgt = torchtext.data.Field(pad_token="<null>", preprocessing=lambda seq: ["<s>"] + seq + ["</s>"])
        self._tgt_vocab = tgt.vocab = Vocab(tgt_vocab)
        # Make data
        if corpus_path is not None:
            self._data = torchtext.data.TabularDataset(
                path=corpus_path, format='tsv',
                fields=[('src', src), ('tgt', tgt)],
                filter_pred=self._len_filter
            )
        else:
            self._data = BilingualDataset(src_corpus, tgt_corpus, src, tgt, filter_pred=self._len_filter)
        # Create training and valid dataset
        examples = self._data.examples
        if truncate is not None:
            assert type(truncate) == int
            examples = examples[:truncate]
        n_train_samples = len(examples) - n_valid_samples
        n_train_samples = int(n_train_samples / self._batch_size) * self._batch_size
        np.random.RandomState(3).shuffle(examples)
        valid_data = torchtext.data.Dataset(
            examples[:n_valid_samples],
            fields=[('src', src), ('tgt', tgt)],
            filter_pred=self._len_filter
        )
        train_data = torchtext.data.Dataset(
            examples[n_valid_samples:n_valid_samples + n_train_samples],
            fields=[('src', src), ('tgt', tgt)],
            filter_pred=self._len_filter
        )
        super(MTDataset, self).__init__(train_data=train_data, valid_data=valid_data, batch_size=batch_size)

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
    
    def n_train_samples(self):
        return len(self._train_data.examples)
    
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
