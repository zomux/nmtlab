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
from nmtlab.dataset.fixed_iterator import FixedBucketIterator


class MTDataset(Dataset):
    """Bilingual dataset.
    """
    
    def __init__(self, corpus_path=None, src_corpus=None, tgt_corpus=None, src_vocab=None, tgt_vocab=None, batch_size=64, batch_type="sentence", max_length=60, n_valid_samples=1000, truncate=None):
        
        assert corpus_path is not None or (src_corpus is not None and tgt_corpus is not None)
        assert src_vocab is not None and tgt_vocab is not None
    
        self._batch_size = batch_size
        self._fixed_train_batches = None
        self._fixed_valid_batches = None
        self._max_length = max_length
        self._n_valid_samples = n_valid_samples
        
        self._src_field = torchtext.data.Field(pad_token="<null>", preprocessing=lambda seq: ["<s>"] + seq + ["</s>"])
        self._src_vocab = self._src_field.vocab = Vocab(src_vocab)
        self._tgt_field = torchtext.data.Field(pad_token="<null>", preprocessing=lambda seq: ["<s>"] + seq + ["</s>"])
        self._tgt_vocab = self._tgt_field.vocab = Vocab(tgt_vocab)
        # Make data
        if corpus_path is not None:
            self._data = torchtext.data.TabularDataset(
                path=corpus_path, format='tsv',
                fields=[('src', self._src_field), ('tgt', self._tgt_field)],
                filter_pred=self._len_filter
            )
        else:
            self._data = BilingualDataset(src_corpus, tgt_corpus, self._src_field, self._tgt_field, filter_pred=self._len_filter)
        # Create training and valid dataset
        examples = self._data.examples
        if truncate is not None:
            assert type(truncate) == int
            examples = examples[:truncate]
        n_train_samples = len(examples) - n_valid_samples
        n_train_samples = int(n_train_samples / batch_size) * batch_size
        # Shuffle and sort examples
        np.random.RandomState(3).shuffle(examples)
        train_examples = examples[n_valid_samples:n_valid_samples + n_train_samples]
        valid_examples = examples[:n_valid_samples]
        # if batch_type == "token":
            # train_examples.sort(key=lambda ex: len(ex.src))
            # valid_examples.sort(key=lambda ex: len(ex.src))
        # Create data
        valid_data = torchtext.data.Dataset(
            valid_examples,
            fields=[('src', self._src_field), ('tgt', self._tgt_field)],
            filter_pred=self._len_filter
        )
        train_data = torchtext.data.Dataset(
            train_examples,
            fields=[('src', self._src_field), ('tgt', self._tgt_field)],
            filter_pred=self._len_filter
        )
        if batch_type == "token":
            # Precompute the batches
            self._fixed_train_batches = self._make_fixed_batches(train_data, self._batch_size)
            self._fixed_valid_batches = self._make_fixed_batches(valid_data, self._batch_size)

        super(MTDataset, self).__init__(train_data=train_data, valid_data=valid_data, batch_size=batch_size, batch_type=batch_type)

    def use_valid_corpus(self, corpus_path=None, src_corpus=None, tgt_corpus=None):
        if corpus_path is not None:
            data = torchtext.data.TabularDataset(
                path=corpus_path, format='tsv',
                fields=[('src', self._src_field), ('tgt', self._tgt_field)],
                filter_pred=self._len_filter
            )
        else:
            data = BilingualDataset(src_corpus, tgt_corpus, self._src_field, self._tgt_field, filter_pred=self._len_filter)
        examples = data.examples
        # if self._batch_type == "token":
        #     examples.sort(key=lambda ex: len(ex.src))
        self._valid_data = torchtext.data.Dataset(
            examples,
            fields=[('src', self._src_field), ('tgt', self._tgt_field)],
            filter_pred=self._len_filter
        )
        if self._batch_type == "token":
            self._fixed_valid_batches = self._make_fixed_batches(self._valid_data, self._batch_size)
    
    def _make_fixed_batches(self, data, n_max_tokens):
        fixed_batches = [[]]
        cur_max_len = 0
        for example in data:
            new_len = max(map(len, [example.src, example.tgt]))
            new_max_len = max(cur_max_len, new_len)
            if new_max_len * (len(fixed_batches[-1]) + 1) > n_max_tokens:
                # Overflow
                fixed_batches.append([example])
                cur_max_len = new_len
            else:
                # Put in the last batch
                fixed_batches[-1].append(example)
                cur_max_len = new_max_len
        np.random.RandomState(3).shuffle(fixed_batches)
        return fixed_batches
    
    def set_gpu_scope(self, scope_index, n_scopes):
        """Training a specific part of data for multigpu environment.
        """
        if self._batch_type == "token":
            self._batch_size = self._batch_size / n_scopes
            train_batches = self._make_fixed_batches(self._train_data, self._batch_size)
            scope_size = int(float(len(train_batches)) / n_scopes)
            self._fixed_train_batches = train_batches[scope_index * scope_size: (scope_index + 1) * scope_size]
            self._fixed_valid_batches = self._make_fixed_batches(self._valid_data, self._batch_size)
        else:
            examples = self._train_data.examples
            scope_size = int(float(len(examples)) / n_scopes)
            self._train_data.examples = examples[scope_index * scope_size: (scope_index + 1) * scope_size]
            self._batch_size = self._batch_size / n_scopes
        
    def _len_filter(self, sample):
        return (
            len(sample.src) > 2 and len(sample.tgt) > 2 and
            len(sample.src) <= self._max_length and len(sample.tgt) <= self._max_length
        )
    
    def n_train_samples(self):
        return len(self._train_data.examples)

    def n_train_batch(self):
        if self._batch_type == "token":
            return len(self._fixed_train_batches)
        else:
            return int(self.n_train_samples() / self._batch_size)
    
    def train_set(self):
        kwargs = dict(
            dataset=self._train_data, batch_size=self._batch_size,
            sort=False, sort_within_batch=True,
            shuffle=True,
            sort_key=lambda x: len(x.src),
            device=None, repeat=False)
        if self._batch_type == "token":
            iterator_class = FixedBucketIterator
            kwargs["fixed_batches"] = self._fixed_train_batches
        else:
            iterator_class = torchtext.data.BucketIterator
        
        batch_iterator = iterator_class(**kwargs)
        return iter(batch_iterator)
    
    def valid_set(self):
        kwargs = dict(
            dataset=self._valid_data, batch_size=self._batch_size,
            sort=False, sort_within_batch=True,
            shuffle=False,
            sort_key=lambda x: len(x.src),
            device=None, repeat=False)
        if self._batch_type == "token":
            iterator_class = FixedBucketIterator
            kwargs["fixed_batches"] = self._fixed_valid_batches
        else:
            iterator_class = torchtext.data.BucketIterator
    
        batch_iterator = iterator_class(**kwargs)
        return iter(batch_iterator)
    
    def src_vocab(self):
        return self._src_vocab
    
    def tgt_vocab(self):
        return self._tgt_vocab
    
    def vocab_sizes(self):
        return self._src_vocab.size(), self._tgt_vocab.size()
