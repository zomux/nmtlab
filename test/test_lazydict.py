#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import unittest


class LazyDictTest(unittest.TestCase):
    
    def test_lazydict(self):
        from nmtlab.utils.tensormap import LazyTensorMap
        ldict = LazyTensorMap()
        k = 2
        ldict["item"] = lambda name: k + 3
        retrieved_item = ldict.item
        self.assertEqual(retrieved_item, 5)

