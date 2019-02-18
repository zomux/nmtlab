#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
    
    @abstractmethod
    def evaluate_line(self, result_line, ref_line):
        """Evaluate one line in the result."""
