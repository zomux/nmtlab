#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_head=4, hidden_size=None, additive=False):
        super(MultiHeadAttention, self).__init__()
        self._num_head = num_head
        self._hidden_size = hidden_size
        self._additive = additive
        if additive and hidden_size is None:
            raise Exception("hidden_size can not be None for additive attention.")
        if additive:
            self.W_q = nn.Parameter(torch.randn((hidden_size, hidden_size)))
            self.W_k = nn.Parameter(torch.randn((hidden_size, hidden_size)))
            self.V_a = nn.Parameter(torch.randn(hidden_size))
    
    def compute_logits(self, query, keys):
        if self._additive:
            h_q = torch.matmul(query, self.W_q)
            h_k = torch.matmul(keys, self.W_k)
            if query.dim() == 2:
                h = h_q[:, None, :] + h_k
                h = torch.tanh(h)
                h = h * self.V_a[None, None, :]
                new_size = list(h.shape[:2]) + [self._num_head, -1]
                logits = h.view(new_size).sum(-1)
                logits = logits.permute(0, 2, 1)  # ~ B x head N, enc N
            else:
                h = h_q[:, :, None, :] + h_k[:, None, :, :]
                h = torch.tanh(h)
                h = h * self.V_a[None, None, None, :]
                new_size = list(h.shape[:3]) + [self._num_head, -1]
                logits = h.view(new_size).sum(-1)  # ~ B x dec N x enc N x head N
                logits = logits.permute(0, 3, 1, 2)  # ~ B x head N x dec N x enc N
        else:
            raise NotImplementedError
        return logits
    
    def forward_2d(self, query, keys, values, mask=None):
        """Compute attention for 2-dimensional queries (batch x hidden).
        """
        logits = self.compute_logits(query, keys)
        if mask is not None:
            penalty = (1 - mask.float()) * 99.
            logits -= penalty[:, None, :]
        weights = F.softmax(logits, dim=1)
        if weights.shape[0] != values.shape[0]:
            values = values.expand(
                [weights.shape[0]] + list(values.shape)[1:])
        n_batch, n_head, _ = list(weights.shape)
        _, n_enc, n_hidden = list(values.shape)
        new_values = values.view(n_batch, n_enc, n_head, -1).permute(0, 2, 1, 3).contiguous().view(n_batch * n_head, n_enc, -1)
        context_vector = torch.bmm(weights.view(n_batch * n_head, 1, n_enc), new_values).squeeze(1)
        context_vector = context_vector.view(n_batch, n_head, -1).view(n_batch, -1)
        return context_vector, weights
    
    def forward_3d(self, query, keys, values, mask=None):
        """Compute attention for 3-dimensional input (batch x step x hidden).
        """
        logits = self.compute_logits(query, keys)
        if mask is not None:
            penalty = (1 - mask.float()) * 99.
            logits -= penalty[:, None, None, :]
        weights = F.softmax(logits, dim=-1)
        n_batch, n_head, n_dec, _ = list(weights.shape)
        _, n_enc, n_hidden = list(values.shape)
        new_values = values.view(n_batch, n_enc, n_head, -1).permute(0, 2, 1, 3).contiguous().view(n_batch * n_head, n_enc, -1)
        context_vector = torch.bmm(weights.view(n_batch * n_head, n_dec, n_enc), new_values)
        context_vector = context_vector.view(n_batch, n_head, n_dec, -1).permute(0, 2, 1, 3).contiguous().view(n_batch, n_dec, -1)
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

