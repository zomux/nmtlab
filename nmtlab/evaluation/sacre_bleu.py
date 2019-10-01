#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
from nmtlab.evaluation.base import EvaluationKit


class SacreBLEUEvaluator(EvaluationKit):

    def __init__(self, dataset_token=None, langpair_token=None, ref_path=None, ref_field=None, ref_delim="\t",
                 tokenizer="13a", lowercase=False):
        if dataset_token is None and ref_path is None:
            raise SystemError("Either ref_token or ref_path is required.")
        super(SacreBLEUEvaluator, self).__init__(ref_path, ref_field, ref_delim)
        self.dataset_token = dataset_token
        self.langpair_token = langpair_token
        self.tokenizer = tokenizer
        self.lowercase = lowercase

    def evaluate(self, result_path):
        from sacrebleu import download_test_set, corpus_bleu, smart_open
        assert os.path.exists(result_path)
        if self.dataset_token is not None:
            _, *refs = download_test_set(self.dataset_token, self.langpair_token)
            if not refs:
                raise SystemError("Error with dataset_token and langpair_token: {} {}".format(
                    self.dataset_token, self.langpair_token))
            refs = [smart_open(x, encoding="utf-8").readlines() for x in refs]
        else:
            refs = [self.ref_lines]
        hyp_lines = open(result_path).readlines()
        bleu = corpus_bleu(hyp_lines, refs, tokenize=self.tokenizer, lowercase=self.lowercase)
        return float(bleu.score)

    def evaluate_line(self, result_line, ref_line):
        raise NotImplementedError
