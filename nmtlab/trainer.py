#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

class MTTrainer(object):
    
    def __init__(self, model, dataset, optimizer, scheduler=None, multigpu=False):
        self._model = model
        self._dataset = dataset
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._multigpu = multigpu
        # Setup horovod
        if multigpu:
            import horovod.torch as hvd
            # Initialize Horovod
            hvd.init()
            # Pin GPU to be used to process local rank (one GPU per process)
            torch.cuda.set_device(hvd.local_rank())
        # Move model to GPU
        self._model.cuda()
    
        
    
