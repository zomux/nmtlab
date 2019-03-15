#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.schedulers.base import Scheduler
import torch.nn as nn


class TransformerScheduler(Scheduler):
    """Scheduler for Adam training when using RNMT+ model.
    """
    
    def __init__(self, factor=2, warm_steps=5000, max_steps=20000, min_lr=0.00005):
        self._factor = factor
        self._warm_steps = warm_steps
        self._max_steps = max_steps
        self._min_lr = min_lr
        self._size_factor = None
        super(TransformerScheduler, self).__init__()
    
    def bind(self, trainer):
        super(TransformerScheduler, self).bind(trainer)
        model = self._trainer.model()
        if isinstance(model, nn.DataParallel):
            size = model.module.hidden_size
        else:
            size = model.hidden_size
        self._devices = self._trainer.devices()
        self._size_factor = size ** -0.5
    
    def _learning_rate(self, t):
        t = float(t + 1)
        lr = self._factor * self._size_factor * min(
            t ** -0.5,
            t * (self._warm_steps ** -1.5)
        )
        if lr < self._min_lr:
            lr = self._min_lr
        return lr
        
    def before_step(self):
        assert self._factor is not None
        t = self._trainer.global_step()
        if t % 2 == 0:
            lr = self._learning_rate(t)
            self._trainer.set_learning_rate(lr, silent=True)
    
    def before_epoch(self):
        # Report learning rate.
        self._trainer.log("TransformerScheduler", "lr={:.4f}".format(self._trainer.learning_rate()))
        
    def is_finished(self):
        return self._trainer.global_step() > self._max_steps
