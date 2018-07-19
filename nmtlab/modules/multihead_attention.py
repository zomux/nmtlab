#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    
    def forward_2d(self, query, keys, values, mask=None):
        """Compute attention for 2-dimensional queries (batch x hidden).
        """
        logits = (query[:, None, :] * keys).sum(dim=2)
        if mask is not None:
            penalty = (1 - mask.float()) * 99.
            logits -= penalty
        weights = F.softmax(logits, dim=1)
        if weights.shape[0] != values.shape[0]:
            values = values.expand(
                [weights.shape[0]] + list(values.shape)[1:])
        context_vector = torch.bmm(weights[:, None, :], values).squeeze(1)
        return context_vector, weights
    
    def forward_3d(self, query, keys, values, mask=None):
        """Compute attention for 3-dimensional input (batch x step x hidden).
        """
        logits = (query[:, :, None, :] * keys[:, None, :, :]).sum(dim=3)
        if mask is not None:
            penalty = (1 - mask.float()) * 99.
            logits -= penalty[:, None, :]
        weights = F.softmax(logits, dim=2)
        context_vector = torch.bmm(weights, values)
        return context_vector, weights
    
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

