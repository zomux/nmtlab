#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.schedulers.base import Scheduler


class RNMTPlusScheduler(Scheduler):
    """Scheduler for Adam training when using RNMT+ model.
    """
    
    def __init__(self, warm_steps=500, decay_start=15000, decay_end=30000, max_steps=50000):
        self._warm_steps = warm_steps
        self._decay_start = decay_start
        self._decay_end = decay_end
        self._max_steps = max_steps
        super(RNMTPlusScheduler, self).__init__()
    
    def
    
    def is_finished(self):
        return self._trainer.global_step() > self._max_steps
