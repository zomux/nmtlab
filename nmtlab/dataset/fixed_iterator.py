#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torchtext.data.iterator import BucketIterator


class FixedBucketIterator(BucketIterator):
    """Fix the number of batches then put examples in them."""
    
    def __init__(self, n_batches, n_max_tokens, **kwargs):
        super(FixedBucketIterator, self).__init__(**kwargs)
        self.n_batches = n_batches
        self.n_max_tokens = n_max_tokens

    def batch(data, batch_size, batch_size_fn=None):
        """Yield elements from data in chunks of batch_size."""
        if batch_size_fn is None:
            def batch_size_fn(new, count, sofar):
                return count
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
        if minibatch:
            yield minibatch
