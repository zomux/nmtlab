#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import torch

from nmtlab.modules.transformer_modules import PositionalEmbedding


class TransformerModulesTest(unittest.TestCase):
    
    def test_positional_embedding(self):
        embed_layer = PositionalEmbedding(50)
        x = torch.randn((3, 30, 50))
        embed = embed_layer(x)
        shape = list(x.shape)
        shape[0] = 1
        self.assertEqual(tuple(shape), tuple(embed.shape))
        offset_embed = embed_layer(x, start=1)
        self.assertEqual(tuple(embed[0, 1].numpy()), tuple(offset_embed[0, 0].numpy()))
