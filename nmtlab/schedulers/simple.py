#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.schedulers.base import Scheduler


class SimpleScheduler(Scheduler):
    """Simply run the trainer for N epoches.
    """
    
    def __init__(self, max_epoch=10):
        super(SimpleScheduler, self).__init__()
        self._max_epoch = max_epoch
    
    def is_finished(self):
        return self._trainer.epoch() >= self._max_epoch - 1
