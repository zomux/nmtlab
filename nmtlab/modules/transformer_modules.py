#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from nmtlab.modules.gelu import gelu

class PositionwiseFeedForward(nn.Module):
    """FFN"""
    
    def __init__(self, out_size, hidden_size, dropout=0.1, activation="relu"):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(out_size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self._activate = F.relu
        elif activation == "gelu":
            self._activate = gelu
        else:
            raise NotImplementedError
        
    def forward(self, x):
        return self.w_2(self.dropout(self._activate(self.w_1(x))))
