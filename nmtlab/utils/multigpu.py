#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def is_root_node():
    try:
        import horovod.torch as hvd
        hvd.init()
        return hvd.rank() == 0
    except ImportError:
        return True
