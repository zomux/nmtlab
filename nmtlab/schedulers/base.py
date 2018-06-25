#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Scheduler(object):
    
    def __init__(self):
        self._binded = False
        self._trainer = None
    
    def bind(self, trainer):
        self._binded = True
        self._trainer = trainer
        
    def before_epoch(self, epoch):
        pass
    
    def after_epoch(self, epoch):
        pass
    
    def after_valid(self, epoch, step, is_improved, score_map):
        pass
    
    def is_finished(self, epoch):
        pass
