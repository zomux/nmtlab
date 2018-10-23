#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod


class Dataset(object):
    """Base class of nmtlab dataset.
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, train_data=None, valid_data=None, batch_size=32, batch_type="sentence"):
        self._train_data = train_data
        self._valid_data = valid_data
        self._batch_size = batch_size
        self._batch_type = batch_type
    
    @abstractmethod
    def set_gpu_scope(self, scope_index, n_scopes):
        """Training a specific part of data for multigpu environment.
        """
    
    @abstractmethod
    def n_train_samples(self):
        """Return the number of training samples.
        """
    
    def n_train_batch(self):
        return int(self.n_train_samples() / self._batch_size)
    
    @abstractmethod
    def train_set(self):
        """
        Return an iterator of the training set.
        """
    
    @abstractmethod
    def valid_set(self):
        """
        Return an iterator of the validation set.
        """
    
    def raw_train_data(self):
        if hasattr(self, "_train_data"):
            return self._train_data
        else:
            return None
    
    def raw_valid_data(self):
        if hasattr(self, "_valid_data"):
            return self._valid_data
        else:
            return None
    
    def batch_size(self):
        return self._batch_size
    
    def batch_type(self):
        return self._batch_type
    
    def set_batch_size(self, batch_size):
        """Change the batch size of the dataset.
        """
        self._batch_size = batch_size
