#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Mapping


class TensorMap(Mapping):
    """A map for contianing tensors.
    """
    
    def __init__(self, *args, **kwargs):
        self._selected_batch = None
        self._detach = False
        self._detach_map = {}
        self._raw_dict = dict(*args, **kwargs)
    
    def get_raw_item(self, item):
        return self._raw_dict.get(item)
    
    def __getattr__(self, attr):
        return self.__getitem__(attr)
    
    def __getitem__(self, item):
        if self._detach and item in self._detach_map:
            ret = self._detach_map[item][0]
        else:
            ret = self.get_raw_item(item)
            if self._detach:
                detached_item = ret.detach()
                detached_item.requires_grad = True
                self._detach_map[item] = (detached_item, ret)
                ret = detached_item
        if self._selected_batch is not None:
            start, end = self._selected_batch
            ret = ret[start:end]
        return ret
    
    def __setitem__(self, key, item):
        self._raw_dict.update({key: item})
    
    def __delattr__(self, item):
        self._raw_dict.__delitem__(item)
    
    def __delitem__(self, key):
        self._raw_dict.__delitem__(key)
    
    def __iter__(self):
        return iter(self._raw_dict)
    
    def __len__(self):
        return len(self._raw_dict)
    
    def update(self, m):
        for k, v in m.items():
            self[k] = v
    
    def select_batch(self, start, end, detach=False):
        """Let the lazy dict return only the batches in the selected range.
        """
        self._selected_batch = (start, end)
        self._detach = detach
    
    def unselect_batch(self):
        self._selected_batch = None
        self._detach = False
        self._detach_map.clear()
    
    def get_detached_items(self):
        return self._detach_map


class LazyTensorMap(TensorMap):
    """Lazily evaluated map
    """
    
    def get_raw_item(self, item):
        return self._raw_dict.get(item)(item)
