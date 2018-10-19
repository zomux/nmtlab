#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from nmtlab.modules.kv_attention import KeyValAttention


class MultiHeadAttention(nn.Module):
    """The implementation of multi-head attention.
    
    Following the original description in the transformer paper.
    """
    
    def __init__(self, out_size, num_head=8, hidden_size=None, additive=False, dropout_ratio=0):
        super(MultiHeadAttention, self).__init__()
        if hidden_size is None:
            hidden_size = out_size
        self._num_head = num_head
        self._hidden_size = hidden_size
        self._out_size = out_size
        self._additive = additive
        self._attention = KeyValAttention(scaling=True, dropout_ratio=dropout_ratio)
        if additive:
            # Taken from RNMT+ paper
            raise NotImplementedError
        else:
            self.linear_Q = nn.Linear(out_size, hidden_size)
            self.linear_K = nn.Linear(out_size, hidden_size)
            self.linear_V = nn.Linear(out_size, hidden_size)
        self.linear_O = nn.Linear(hidden_size, out_size)
    
    def forward_2d(self, query, keys, values, mask=None):
        """Compute attention for 2-dimensional queries (batch x hidden).
        """
        query = query.unsqueeze(1)  # (B, 1, H)
        context_vectors, weights = self.forward_3d(query, keys, values, mask=mask)
        context_vectors = context_vectors.squeeze(1)
        weights = weights.squeeze(1)
        return context_vectors, weights
    
    def forward_3d(self, query, keys, values, mask=None):
        """Compute attention for 3-dimensional input (batch x step x hidden).
        """
        B = query.shape[0]
        head_dim = self._hidden_size // self._num_head
        query = self.linear_Q(query).view(B, -1, self._num_head, head_dim).transpose(1, 2)  # (B, 4, T2, H)
        keys = self.linear_K(keys).view(B, -1, self._num_head, head_dim).transpose(1, 2)
        values = self.linear_V(values).view(B, -1, self._num_head, head_dim).transpose(1, 2)
        if mask is not None and mask.dim() < keys.dim():
            mask = mask.unsqueeze(1)
        context_vectors, weights = self._attention(query, keys, values, mask=mask)  # (B, 4, T2, H)
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(B, -1, self._num_head * head_dim)  # (B, T2, H)
        context_vectors = self.linear_O(context_vectors)
        return context_vectors, weights
    
    def forward(self, query, keys, values, mask=None):
        """Compute the context vector with key value attention.
        
        Returns:
            context vector and attention weights.
        """
        if query.dim() == 2:
            return self.forward_2d(query, keys, values, mask)
        elif query.dim() == 3:
            return self.forward_3d(query, keys, values, mask)
        else:
            raise NotImplementedError

