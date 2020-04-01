#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KeyValAttention(nn.Module):
    
    def __init__(self, scaling=False, dropout_ratio=0):
        """Initialize a key-value attention class.
        Args:
            scaling - Whether normalize the attention weights by sqrt(size)
            dropout_ratio - The probability of dropout on the logits
        """
        super(KeyValAttention, self).__init__()
        self._scaling = scaling
        self._dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else None

    def forward_2d(self, query, keys, values, mask=None, additional_logits=None):
        """Compute attention for 2-dimensional queries (batch x hidden).
        """
        context_vector, weights = self.forward_3d(query.unsqueeze(-2), keys, values, mask, additional_logits)
        return context_vector.squeeze(-2), weights.squeeze(-2)
    
    def forward_3d(self, query, keys, values, mask=None, additional_logits=None):
        """Compute attention for 3-dimensional input (batch x step x hidden).
        """
        logits = torch.matmul(query, keys.transpose(-2, -1))
        if additional_logits is not None:
            logits += additional_logits
        if self._scaling:
            logits /= math.sqrt(query.shape[-1])
        if mask is not None:
            if self._dropout is not None:
                mask = self._dropout(mask)
            if mask.dim() < logits.dim():
                mask = mask.unsqueeze(-2)
            logits = logits.masked_fill(mask == 0, -1e3)
        elif self._dropout is not None:
            # Using dropout but no mask
            mask = self._dropout(logits.new_ones(logits.shape))
            logits = logits.masked_fill(mask == 0, -1e3)
        weights = F.softmax(logits, dim=-1)
        context_vector = torch.matmul(weights, values)
        return context_vector, weights
    
    def forward(self, query, keys, values, mask=None, additional_logits=None):
        """Compute the context vector with key value attention.
        
        Returns:
            context vector and attention weights.
        """
        if query.dim() == keys.dim() - 1:
            return self.forward_2d(query, keys, values, mask, additional_logits=additional_logits)
        else:
            return self.forward_3d(query, keys, values, mask, additional_logits=additional_logits)

