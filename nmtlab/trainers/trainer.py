#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.trainers.base import TrainerKit
from six.moves import xrange

MAX_EPOCH = 10000


class MTTrainer(TrainerKit):
    
    def run(self):
        """Run the training from begining to end.
        """
        self.valid(force=True)
        self._model.train(True)
        for epoch in xrange(MAX_EPOCH):
            self.begin_epoch(epoch)
            for step, batch in enumerate(self._dataset.train_set()):
                self.begin_step(step)
                self.train(batch)
                self.valid()
            self.end_epoch()
            # Check if finished
            if self.is_finished():
                break
