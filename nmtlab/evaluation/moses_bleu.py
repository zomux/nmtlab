#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
from nmtlab.evaluation.base import EvaluationKit


class MosesBLEUEvaluator(EvaluationKit):
    
    def prepare(self):
        dir_path = os.path.dirname(__file__)
        self._script_path = os.path.join(dir_path, "..", "..", "scripts", "multi-bleu.perl")
        self._script_path = os.path.abspath(self._script_path)
        
    def evaluate(self, result_path):
        cmd = ("{} {} < {}".format(self._script_path, self.ref_path, result_path))
        # print(cmd)
        pipe = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        bleu = pipe.stdout.decode("utf-8").replace("BLEU", "").replace("=", "").split(",")[0].strip()
        return float(bleu)
    
    def evaluate_line(self, result_line, ref_line):
        raise NotImplementedError
