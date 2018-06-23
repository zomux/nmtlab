#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchtext

class MTDataset(object):
    """Bilingual dataset.
    """
    
    def __init__(self, corpus_path, src_vocab_path, tgt_vocab_path, max_length=60):
        self._max_length = max_length
        self.train_data = torchtext.data.TabularDataset(
            path=corpus_path, format='tsv',
            fields=[('src', src), ('tgt', tgt)],
            filter_pred=self._len_filter
        )
        
    def _len_filter(self, sample):
        return len(sample.src) <= 60 and len(sample.tgt) <= 60
    

