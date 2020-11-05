#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import torch
import torch.distributed as dist
import importlib

from nmtlab.utils import OPTS


def execution_env():
    if not torch.cuda.is_available():
        return ""
    nsml_installed = importlib.util.find_spec("nsml") is not None
    hvd_installed = importlib.util.find_spec("horovod") is not None
    if nsml_installed:
        return "nsml"
    elif hvd_installed:
        raise SystemError("horovod is deprecated!")
        return "horovod"
    else:
        return ""


def world_size():
    env = execution_env()
    if env == "nsml":
        from nsml import GPU_NUM, PARALLEL_WORLD
        n_world = max(len(PARALLEL_WORLD), 1)
        return int(GPU_NUM) * n_world
    elif env == "horovod":
        import horovod.torch as hvd
        torch.init()
        return hvd.size()
    elif torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


def distributed_init(local_rank, local_size=1, master="127.0.0.1", port="12355"):
    if dist.is_nccl_available():
        backend = "nccl"
    else:
        backend = "gloo"
    if local_rank == 0:
        print("[DISTRIBUTED] using {} backend".format(backend))
        sys.stdout.flush()
    init_method = None
    node_rank = 0
    node_size = 1
    if execution_env() == "nsml":
        from nsml import PARALLEL_WORLD, PARALLEL_PORTS, MY_RANK
        if len(PARALLEL_WORLD) > 1:
            master = PARALLEL_WORLD[0]
            port = PARALLEL_PORTS[0]
            init_method = "tcp://{}:{}".format(master, port)
            node_rank = MY_RANK
            node_size = len(PARALLEL_WORLD)
    # print("[nmtlab] Backend {} is used for Data Distributed Parallel".format(backend))
    os.environ['MASTER_ADDR'] = master
    os.environ['MASTER_PORT'] = str(port)
    rank = node_rank * local_size + local_rank
    world_sz = node_size * local_size
    dist.init_process_group(backend, rank=rank, world_size=world_sz, init_method=init_method)
    OPTS.dist_local_rank = local_rank
    OPTS.dist_local_size = local_size


def global_rank():
    return node_rank() * local_size() + local_rank()


def global_size():
    return local_size() * node_size()


def local_rank():
    if "dist_local_rank" in OPTS and OPTS.dist_local_rank is not None:
        return OPTS.dist_local_rank
    else:
        return 0


def local_size():
    if "dist_local_size" in OPTS and OPTS.dist_local_size is not None:
        return OPTS.dist_local_size
    elif torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


def node_rank():
    node_rank = 0
    if execution_env() == "nsml":
        from nsml import PARALLEL_WORLD, MY_RANK
        if len(PARALLEL_WORLD) > 1:
            node_rank = MY_RANK
    return node_rank


def node_size():
    node_size = 1
    if execution_env() == "nsml":
        from nsml import PARALLEL_WORLD
        if len(PARALLEL_WORLD) > 1:
            node_size = len(PARALLEL_WORLD)
    return node_size


def distributed_cleanup():
    dist.destroy_process_group()
