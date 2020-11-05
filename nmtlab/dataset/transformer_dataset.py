#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torchtext
from torchtext.data.example import Example
import pickle
import numpy as np
from nmtlab.utils.vocab import Vocab
from nmtlab.dataset.bilingual_dataset import BilingualDataset
from nmtlab.dataset.base import Dataset
from nmtlab.dataset.fixed_iterator import FixedBucketIterator
from nmtlab.dataset.distributed_dataset import DistributedMTDataset


class FastTransformerDataset(DistributedMTDataset):

    def __init__(self, corpus_path=None, src_corpus=None, tgt_corpus=None, src_vocab=None, tgt_vocab=None, batch_size=4096, max_length=60, n_valid_samples=1000, truncate=None, bucketing=False):
        if batch_size > 10000:
            raise SystemError("Batch size in FastTransformerDataset is for single GPU")
        super(FastTransformerDataset, self).__init__(
            corpus_path=corpus_path, src_corpus=src_corpus,
            tgt_corpus=tgt_corpus, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
            batch_size=batch_size, batch_type="token",
            max_length=max_length, n_valid_samples=n_valid_samples,
            truncate=truncate
        )
        self._bucketing = bucketing

    def create_batches(self, idlen_pairs, is_valid=False):
        fixed_batches = [[]]
        cur_max_len = 0
        if self._bucketing and not is_valid:
            idlen_pairs.sort(key=lambda p: p[1])
        for line_id, leng in idlen_pairs:
            new_max_len = max(cur_max_len, leng)
            if new_max_len * (len(fixed_batches[-1]) + 1) > self._batch_size:
                # Overflow
                fixed_batches.append([line_id])
                cur_max_len = leng
            else:
                # Put in the last batch
                fixed_batches[-1].append(line_id)
                cur_max_len = new_max_len
        np.random.RandomState(3).shuffle(fixed_batches)
        return fixed_batches

    def __reduce__(self):
        return (
            FastTransformerDataset,
            (
                self._corpus_path,
                self._src_corpus,
                self._tgt_corpus,
                self._src_vocab_path,
                self._tgt_vocab_path,
                self._batch_size,
                self._max_length,
                self._n_valid_samples,
                self._truncate,
                self._bucketing
            )
        )


if __name__ == '__main__':
    import os
    src_corpus = "../../data/WMT14_processed/train.en"
    tgt_corpus = src_corpus.replace(".en", ".de")
    src_vocab = "../../data/WMT14_processed/wmt14_fair_en.vocab"
    tgt_vocab = src_vocab.replace("_en", "_de")
    assert os.path.exists(src_corpus) and os.path.exists(tgt_corpus)
    ds = FastTransformerDataset(
        src_corpus=src_corpus, tgt_corpus=tgt_corpus,
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        batch_size=8192
    )
    ds.precompute_batches()
    print("size of precomputing batches", len(ds._precomputed_batches))
    print("first batch:")
    print(ds._precomputed_batches[0])
    print("load batch for 1/8")
    ds.load_batches(0, 8)
    print(len(ds._fixed_train_batches))
    print(len(ds._fixed_valid_batches))

