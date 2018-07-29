#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.schedulers.base import Scheduler


class RNMTPlusAdamScheduler(Scheduler):
    """Scheduler for Adam training when using RNMT+ model.
    """
    
    def __init__(self, warm_steps=500, decay_start=60000, decay_end=120000, max_steps=200000, min_lr=0.00005):
        self._warm_steps = warm_steps
        self._decay_start = decay_start
        self._decay_end = decay_end
        self._max_steps = max_steps
        self._min_lr = min_lr
        super(RNMTPlusAdamScheduler, self).__init__()
    
    def before_step(self):
        t = self._trainer.global_step()
        if t % 100 == 0:
            n = self._trainer.devices()
            lr = 0.0001 * min(
                1 + t * (n - 1) / (n * self._warm_steps),
                n,
                n * (2 * n) ** ((self._decay_start - n * t) / (self._decay_end - self._decay_start))
            )
            if lr < self._min_lr:
                lr = self._min_lr
            if lr != self._trainer.learning_rate():
                self._trainer.set_learning_rate(lr)
    
    def before_epoch(self):
        # Report learning rate.
        self._trainer.log("RNMTPlusAdamScheduler", "lr={}".format(self._trainer.learning_rate()))
        
    def is_finished(self):
        return self._trainer.global_step() > self._max_steps
