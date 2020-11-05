#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from shutil import copyfile
from abc import ABCMeta, abstractmethod
import numpy as np


class EvaluationKit(object):
    """Class of evaluating translation results
    """

    __metaclass__ = ABCMeta
    
    def __init__(self, ref_path=None, ref_field=None, ref_delim="\t"):
        if ref_field is None:
            ref_field = 0
        self.ref_lines = []
        self.ref_path = ref_path
        if ref_path is not None:
            for ref_line in map(str.strip, open(ref_path)):
                if ref_field is not None:
                    fields = ref_line.split(ref_delim)
                    assert len(fields) > ref_field
                    ref_line = fields[ref_field]
                self.ref_lines.append(ref_line)
        self.prepare()
    
    def prepare(self):
        """Preparation function, which will be called after initialization.
        """
    
    def evaluate(self, result_path):
        """Evaluate the given result file.
        """
        result_lines = list(map(str.strip, open(result_path)))
        scores = []
        for result, ref in zip(result_lines, self.ref_lines):
            scores.append(self.evaluate_line(result, ref))
        return np.mean(scores)

    def post_process(self, path, out_path="/tmp/preprocessed_results.txt",
                     recover_subwords=True, detokenize=True):
        if recover_subwords:
            self.recover_subwords(path, out_path)
        if detokenize:
            script = "scripts/detokenizer.perl"
            os.system("perl {} < {} > {}.detok".format(script, out_path, out_path))
            copyfile("{}.detok".format(out_path), out_path)
        return out_path

    def recover_subwords(self, path, out_path):
        with open(out_path, "w") as outf:
            for line in open(path):
                # Remove duplicated tokens
                tokens = line.strip().split()
                new_tokens = []
                for tok in tokens:
                    if len(new_tokens) > 0 and tok != new_tokens[-1]:
                        new_tokens.append(tok)
                    elif len(new_tokens) == 0:
                        new_tokens.append(tok)
                new_line = " ".join(new_tokens) + "\n"
                line = new_line
                # Remove sub-word indicator in sentencepiece and BPE
                line = line.replace("@@ ", "")
                if "▁" in line:
                    line = line.strip()
                    line = "".join(line.split())
                    line = line.replace("▁", " ").strip() + "\n"
                outf.write(line)
    
    @abstractmethod
    def evaluate_line(self, result_line, ref_line):
        """Evaluate one line in the result."""
