#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.schedulers.base import Scheduler


class AnnealScheduler(Scheduler):
    """Scheduler for annealing learning rate.
    """
    
    def __init__(self, patience=5, n_total_anneal=3, anneal_factor=2.):
        super(AnnealScheduler, self).__init__()
        self._patience = patience
        self._fail_count = 0
        self._anneal_count = 0
        self._n_total_anneal = n_total_anneal
        self._anneal_factor = anneal_factor
        self._finished = False
        
    def after_valid(self, is_improved, score_map):
        if not is_improved:
            self._fail_count += 1
        else:
            self._fail_count = 0
        if self._fail_count >= self._patience:
            if self._anneal_count >= self._n_total_anneal:
                self._finished = True
            else:
                new_lr = self._trainer.learning_rate() / self._anneal_factor
                self._trainer.set_learning_rate(new_lr)
                self._fail_count = 0
                self._anneal_count += 1
    
    def is_finished(self):
        return self._finished
