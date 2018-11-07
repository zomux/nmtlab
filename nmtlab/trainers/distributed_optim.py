#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from horovod.torch.compression import Compression
from horovod.torch.mpi_ops import allreduce, allreduce_async, allreduce_, allreduce_async_
from horovod.torch.mpi_ops import poll, synchronize
from horovod.torch.mpi_ops import size, local_size, rank, local_rank
from nmtlab.utils import OPTS


class _FlexibleDistributedOptimizer(torch.optim.Optimizer):
    
    def __init__(self, params, named_parameters, compression):
        super(self.__class__, self).__init__(params)
        self._compression = compression
        
        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []
        
        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')
        
        if len(named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                     in sorted(named_parameters)}
        else:
            self._parameter_names = {v: 'allreduce.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}
        
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        if size() > 1:
            self._register_hooks()
    
    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)
    
    def _allreduce_grad(self, p):
        name = self._parameter_names.get(p)
        tensor = p.grad.data
        tensor_compressed, ctx = self._compression.compress(tensor)
        
        handle = allreduce_async_(tensor_compressed, average=True, name=name)
        return handle, ctx
    
    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            self._allreduce_grad(p)
        
        for p, value in self._handles.items():
            handle, ctx = value
            output = synchronize(handle)
            p.grad.data.set_(self._compression.decompress(output, ctx))
        self._handles.clear()
    
    def step(self, closure=None):
        self.synchronize()
        return super(self.__class__, self).step(closure)
    
    def _make_hook(self, p):
        def hook(*ignore):
            if OPTS.disable_backward_hooks:
                return
            assert p not in self._handles
            assert not p.grad.requires_grad
            handle, ctx = self._allreduce_grad(p)
            self._handles[p] = (handle, ctx)
        return hook


def FlexibleDistributedOptimizer(optimizer, named_parameters=None, compression=Compression.none):
    methods = dict(_FlexibleDistributedOptimizer.__dict__)
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               methods)
    return cls(optimizer.param_groups, named_parameters, compression)


