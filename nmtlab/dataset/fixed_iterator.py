#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from torchtext.data.iterator import BucketIterator


class FixedBucketIterator(BucketIterator):
    """Fix the number of batches then put examples in them."""
    
    def __init__(self, fixed_batches=None, **kwargs):
        super(FixedBucketIterator, self).__init__(**kwargs)
        self.fixed_batches = fixed_batches
        assert self.fixed_batches is not None

    def batch(self):
        yield iter(self.fixed_batches)

    def pool(self):
        batches = self.fixed_batches
        if self.random_shuffler is None:
            random_shuffler = random.shuffle
        else:
            random_shuffler = self.random_shuffler
        if self.shuffle:
            batches = random_shuffler(batches)
        for batch in batches:
            if self.sort_within_batch:
                batch = sorted(batch, key=self.sort_key)
            yield batch
    
    def create_batches(self):
        if self.sort:
            self.batches = self.batch()
        else:
            self.batches = self.pool()
