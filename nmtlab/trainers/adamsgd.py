#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
from torch.optim.adam import Adam

class AdamSGD(Adam):
    """Two-stage optimizer with Adam and SGD
    """
    
    def __init__(self, params, adam_lr=1e-3, sgd_lr=1.0):
        super(AdamSGD, self).__init__(params, lr=adam_lr)
        self._sgd_mode = False
        self._sgd_lr = sgd_lr
        self._adam_lr = adam_lr
    
    def switch_to_sgd(self):
        self._sgd_mode = True
        for group in self.param_groups:
            group["lr"] = self._sgd_lr
    
    def switch_to_adam(self):
        self._sgd_mode = False
        for group in self.param_groups:
            group["lr"] = self._adam_lr
    
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self._sgd_mode:
            return super(AdamSGD, self).step(closure=closure)
        else:
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data

                    p.data.add_(-group['lr'], grad)
            return loss
        


