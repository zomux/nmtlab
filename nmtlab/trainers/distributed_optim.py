#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from nmtlab.utils import OPTS
from horovod.torch import _DistributedOptimizer
from horovod.torch.compression import Compression

METHOD_KEYS = ["__init__",
               "_register_hooks", "_allreduce_grad_async",
               "synchronize", "step"]


class _FlexibleDistributedOptimizer(_DistributedOptimizer):

    def _make_hook(self, p):
        def hook(*ignore):
            if OPTS.disable_backward_hooks:
                return
            assert not p.grad.requires_grad
            handle, ctx = None, None
            handle, ctx = self._allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)

        return hook


def FlexibleDistributedOptimizer(optimizer, named_parameters=None, compression=Compression.none):
    methods = dict()
    methods["_make_hook"] = _FlexibleDistributedOptimizer._make_hook
    for key in METHOD_KEYS:
        methods[key] = eval("_DistributedOptimizer.{}".format(key))
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               methods)
    return cls(optimizer.param_groups, named_parameters, compression)


