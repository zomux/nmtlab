#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyValAttention(nn.Module):
    
    def forward(self, query, keys, values, mask=None):
        """Compute the context vector with key value attention.
        
        Returns:
            context vector and attention weights.
        """
        attention_logits = (query[:, None, :] * keys).sum(dim=2)
        if mask is not None:
            penalty = (1 - mask.float()) * 99.
            attention_logits -= penalty
        attention_weights = F.softmax(attention_logits, dim=1)
        if attention_weights.shape[0] != values.shape[0]:
            values = values.expand(
                [attention_weights.shape[0]] + list(values.shape)[1:])
        context_vector = torch.bmm(attention_weights[:, None, :], values).squeeze(1)
        return context_vector, attention_weights
