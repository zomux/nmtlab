#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
from nmtlab.dataset.mt_dataset import MTDataset


class FixedBucketIteratorTest(unittest.TestCase):
    
    def test_batch_token_limit(self):
        dataset = MTDataset(
            corpus_path="test/data_examples.txt",
            src_vocab="test/data_example_src.vocab",
            tgt_vocab="test/data_example_tgt.vocab",
            batch_size=300,
            batch_type="token"
        )
        batches = list(dataset.train_set())
        for batch in batches:
            self.assertLessEqual(np.prod(batch.src.shape), 300)
            self.assertLessEqual(np.prod(batch.tgt.shape), 300)
