#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

from nmtlab.functions.gelu import gelu
from nmtlab.modules.multihead_attention import MultiHeadAttention
from nmtlab.functions.residual import residual_connect


class PositionwiseFeedForward(nn.Module):
    """FFN"""
    
    def __init__(self, size, hidden_size, dropout_ratio=0.1, activation="relu"):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.dropout = nn.Dropout(dropout_ratio)
        if activation == "relu":
            self._activate = F.relu
        elif activation == "gelu":
            self._activate = gelu
        else:
            raise NotImplementedError
        
    def forward(self, x):
        return self.w_2(self.dropout(self._activate(self.w_1(x))))


class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, size, ff_size=None, dropout_ratio=0.1):
        super(TransformerEncoderLayer, self).__init__()
        if ff_size is None:
            ff_size = size * 4
        self.dropout = nn.Dropout(dropout_ratio)
        self.attention = MultiHeadAttention(size, dropout_ratio=dropout_ratio)
        self.ff_layer = PositionwiseFeedForward(size, ff_size, dropout_ratio=dropout_ratio)
        self.layer_norm1 = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm()
    
    def forward(self, x, src_mask):
        # Attention layer
        y1 = self.layer_norm1(x)
        y1 = self.attention(y, y, y, mask=src_mask)
        y1 = self.dropout(y)
        y1 = residual_connect(y1, x)
        # Feed-forward layer
        y2 = self.layer_norm2(y1)
        y2 = self.ff_layer(y2)
        y2 = self.dropout(y2)
        y2 = residual_connect(y2, y1)
        return y2
