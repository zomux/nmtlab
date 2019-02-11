#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from nmtlab.modules.kv_attention import KeyValAttention


class MultiHeadAttention(nn.Module):
    """The implementation of multi-head attention.
    
    Following the original description in the transformer paper.
    """

    _RELATIVE_POS_CLIP = 2
    
    def __init__(self, out_size, num_head=8, hidden_size=None, additive=False, dropout_ratio=0, relative_pos=False):
        super(MultiHeadAttention, self).__init__()
        if hidden_size is None:
            hidden_size = out_size
        self._num_head = num_head
        self._hidden_size = hidden_size
        self._out_size = out_size
        self._additive = additive
        if relative_pos:
            self.relative_posmatrix = nn.Embedding(self._RELATIVE_POS_CLIP * 2 + 1, hidden_size)
        else:
            self.relative_posmatrix = None
        self._attention = KeyValAttention(scaling=True, dropout_ratio=dropout_ratio, )
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
        transformed_query = self.linear_Q(query)
        if self.relative_posmatrix is not None:
            T2 = query.shape[1]
            T1 = keys.shape[1]
            pos = torch.arange(T1).repeat(T2, 1)
            relpos = pos - torch.arange(T2)[:, None]
            relpos = torch.clamp(relpos, -self._RELATIVE_POS_CLIP, self._RELATIVE_POS_CLIP)
            relpos += self._RELATIVE_POS_CLIP
            if torch.cuda.is_available():
                relpos = relpos.cuda()
            relpos_embed = self.relative_posmatrix(relpos)
            relpos_logits = (transformed_query.unsqueeze(-2) * relpos_embed.unsqueeze(0)).sum(-1)
            relpos_logits = relpos_logits.unsqueeze(1)
        else:
            relpos_logits = None
        query = transformed_query.view(B, -1, self._num_head, head_dim).transpose(1, 2)  # (B, 4, T2, H)
        keys = self.linear_K(keys).view(keys.shape[0], -1, self._num_head, head_dim).transpose(1, 2)
        values = self.linear_V(values).view(values.shape[0], -1, self._num_head, head_dim).transpose(1, 2)
        if mask is not None and mask.dim() < keys.dim():
            mask = mask.unsqueeze(1)
        context_vectors, weights = self._attention(query, keys, values, mask=mask, additional_logits=relpos_logits)  # (B, 4, T2, H)
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

