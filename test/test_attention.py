#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionTest(unittest.TestCase):
    
    def test_multihead_attention_dot(self):
        from nmtlab.modules.multihead_attention import MultiHeadAttention
        attention = MultiHeadAttention(16, additive=False)
        enc_states = torch.randn((4, 8, 16))
        dec_states = torch.randn((4, 10, 16))
        enc_mask = torch.randn((4, 8))
        context_vector, _ = attention(dec_states, enc_states, enc_states, mask=enc_mask)
        self.assertEqual(tuple(context_vector.shape), (4, 10, 16))
    
    def test_multihead_attention_2d(self):
        from nmtlab.modules.multihead_attention import MultiHeadAttention
        attention = MultiHeadAttention(16, additive=False)
        enc_states = torch.randn((4, 8, 16))
        dec_states = torch.randn((4, 16))
        enc_mask = torch.randn((4, 8))
        context_vector, _ = attention(dec_states, enc_states, enc_states, mask=enc_mask)
        self.assertEqual(tuple(context_vector.shape), (4, 16))
        context_vector_3d, _ = attention(dec_states[:, None, :], enc_states, enc_states, mask=enc_mask)
        self.assertEqual(torch.eq(context_vector_3d.squeeze(1), context_vector).prod().numpy(), 1)
