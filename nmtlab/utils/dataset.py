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
        self.train_data = torchtext.data.TabularDataset(
            path=corpus_path, format='tsv',
            fields=[('src', src), ('tgt', tgt)],
            filter_pred=len_filter
        )
    

