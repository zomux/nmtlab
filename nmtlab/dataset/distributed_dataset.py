#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torchtext
from torchtext.data.example import Example
import pickle
import numpy as np
from nmtlab.dataset.mt_dataset import MTDataset
from abc import ABCMeta, abstractmethod


class DistributedMTDataset(MTDataset):
    """Picklable dataset for multi-gpu training
    """

    __metaclass__ = ABCMeta

    def __init__(self, corpus_path=None, src_corpus=None, tgt_corpus=None, src_vocab=None, tgt_vocab=None, batch_size=4096, batch_type="sentence", max_length=60, n_valid_samples=1000, truncate=None):
        super(DistributedMTDataset, self).__init__(
            corpus_path=corpus_path, src_corpus=src_corpus,
            tgt_corpus=tgt_corpus, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
            batch_size=batch_size, batch_type=batch_type,
            max_length=max_length, n_valid_samples=n_valid_samples,
            truncate=truncate, load_full_dataset=False
        )
        self._precomputed_batches = None
        self._precomputed_valid_batches = None
        self._fixed_train_batches = None
        self._fixed_valid_batches = None
        self._fields = {
            "src": ("src", self._src_field),
            "tgt": ("tgt", self._tgt_field)
        }
        self._cache_fp = "/tmp/distributed_ds_precompute.{}.pkl".format(os.getppid())

    def precompute_batches(self, generate_valid_batches=True):
        assert self._tgt_corpus is not None
        lenlist = []
        for src, tgt in zip(open(self._src_corpus, encoding="utf-8"), open(self._tgt_corpus, encoding="utf-8")):
            lenlist.append(max(src.count(" ") + 1, tgt.count(" ") + 1))
        lenpairs = [(sent_id, leng) for (sent_id, leng) in zip(range(len(lenlist)), lenlist) if leng <= self._max_length]
        if self._truncate is not None:
            lenpairs = lenpairs[:self._truncate]
        np.random.RandomState(3).shuffle(lenpairs)
        if generate_valid_batches:
            valid_lenpairs = lenpairs[:self._n_valid_samples]
            lenpairs = lenpairs[self._n_valid_samples:]
            self._precomputed_valid_batches = self.create_batches(valid_lenpairs, is_valid=True)
        self._precomputed_batches = self.create_batches(lenpairs, is_valid=False)
        pickle.dump(
            (self._precomputed_batches, self._precomputed_valid_batches),
            open(self._cache_fp, "wb"))

    def load_batches(self, device_rank=0, world_size=1):
        if not self._precomputing_done():
            assert os.path.exists(self._cache_fp)
            self._precomputed_batches, self._precomputed_valid_batches = pickle.load(open(self._cache_fp, "rb"))
        assert self._precomputing_done()
        scope_size = int(float(len(self._precomputed_batches)) / world_size)
        selected_batches = self._precomputed_batches[device_rank * scope_size: (device_rank + 1) * scope_size]
        id_set = set()
        for bat in selected_batches:
            for sent_id in bat:
                id_set.add(sent_id)
        if self._precomputed_valid_batches is not None:
            for bat in self._precomputed_valid_batches:
                for sent_id in bat:
                    id_set.add(sent_id)
        example_map = {}
        sent_id = 0
        for src, tgt in zip(open(self._src_corpus, encoding="utf-8"), open(self._tgt_corpus, encoding="utf-8")):
            if sent_id in id_set:
                example_map[sent_id] = self._make_example(src, tgt)
            sent_id += 1
        self._fixed_train_batches = []
        for batch in selected_batches:
            example_batch = [example_map[i] for i in batch]
            self._fixed_train_batches.append(example_batch)
        self._train_data = torchtext.data.Dataset(
            [],
            fields=[('src', self._src_field), ('tgt', self._tgt_field)],
            filter_pred=self._len_filter
        )
        if self._precomputed_valid_batches is not None:
            self._fixed_valid_batches = []
            valid_examples = []
            for batch in self._precomputed_valid_batches:
                example_batch = [example_map[i] for i in batch]
                valid_examples.extend(example_batch)
                self._fixed_valid_batches.append(example_batch)
            self._valid_data = torchtext.data.Dataset(
                valid_examples,
                fields=[('src', self._src_field), ('tgt', self._tgt_field)],
            )

    def _make_example(self, src, tgt):
        return Example.fromdict(
            {"src": src.strip(), "tgt": tgt.strip()},
            self._fields
        )

    def _precomputing_done(self):
        return self._precomputed_batches is not None

    def n_train_samples(self):
        if self._precomputing_done():
            return sum(map(len, self._precomputed_batches))
        else:
            return 0

    def n_train_batch(self):
        if self._fixed_train_batches is not None:
            return len(self._fixed_train_batches)
        elif self._precomputing_done():
            return len(self._precomputed_batches)
        else:
            return 0

    @abstractmethod
    def create_batches(self, idlen_pairs, is_valid=False):
        pass

    @abstractmethod
    def __reduce__(self):
        pass