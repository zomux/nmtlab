#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.evaluation.base import EvaluationKit
from nmtlab.utils import bleu


class TokenizedBLEUEvaluator(EvaluationKit):
    """Evaluate tokenized BLEU."""
    
    def evaluate_line(self, result_line, ref_line):
        result_tokens = result_line.split()
        ref_tokens = ref_line.split()
        return bleu(result_tokens, ref_tokens)
