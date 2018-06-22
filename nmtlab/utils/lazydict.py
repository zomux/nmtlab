#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Mapping


class LazyDict(Mapping):
    """Lazily evaluated map
    """
    
    def __init__(self, *args, **kwargs):
        self._raw_dict = dict(*args, **kwargs)
    
    def __getattr__(self, attr):
        return self._raw_dict.get(attr)()
    
    def __getitem__(self, item):
        return self._raw_dict.get(item)()
    
    def __setitem__(self, key, func):
        self._raw_dict.update({key: func})
    
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
