#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from nmtlab.evaluation.sacre_bleu import SacreBLEUEvaluator


class SacreBleuTest(unittest.TestCase):

    def test_dataset_token(self):
        ev = SacreBLEUEvaluator(dataset_token="wmt14", langpair_token="de-en")
        bleu = ev.evaluate("test/data_wmt14_deen.hyp")
        assert 29.38 < bleu < 29.39

    def test_ref_lines(self):
        ev = SacreBLEUEvaluator(ref_path="test/data_wmt14_deen.ref")
        bleu = ev.evaluate("test/data_wmt14_deen.hyp")
        assert 29.38 < bleu < 29.39

