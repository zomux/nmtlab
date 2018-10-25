#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nmtlab.models.transformer import Transformer
from nmtlab.utils import MapDict


class TransformerTest(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TransformerTest, self).__init__(*args, **kwargs)
        self.model = Transformer(
            num_encoders=2, num_decoders=2,
            src_vocab_size=100, tgt_vocab_size=100,
            hidden_size=64, embed_size=64,
        )
        self.model.train(False)
    
    def test_stepwise_graph(self):
        input_seq = torch.randint(0, 100, (3, 5)).long()
        target_seq = torch.randint(0, 100, (3, 6)).long()
        src_mask = input_seq.clone().fill_(1)
        context = self.model.encode(input_seq, src_mask)
        self.model.set_stepwise_training(False)
        context, states = self.model.pre_decode(context, target_seq, src_mask=src_mask)
        context = MapDict(context)
        full_states = self.model.decode(context, states, False)
        self.model.set_stepwise_training(True)
        context, states = self.model.pre_decode(context, target_seq, src_mask=src_mask)
        context = MapDict(context)
        stepwise_states = self.model.decode(context, states, False)
        self.assertAlmostEqual(float((full_states.final_states - stepwise_states.final_states).sum()), 0, delta=0.0001)


if __name__ == '__main__':
    test = TransformerTest()
    test.test_stepwise_graph()
