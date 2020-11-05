#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from nmtlab.utils import OPTS
from nmtlab.dataset.distributed_dataset import DistributedMTDataset


def distributed_process(local_rank, local_size, dataset, trainer):
    from nmtlab.utils.distributed import distributed_init, distributed_cleanup, global_rank, global_size
    import torch.distributed as dist
    distributed_init(local_rank, local_size)

    assert isinstance(dataset, DistributedMTDataset)
    if local_rank == 0:
        dataset.precompute_batches()
    dist.barrier()
    dataset.load_batches(global_rank(), global_size())

    torch.cuda.set_device(local_rank)
    trainer.set_dataset(dataset)
    OPTS.trainer = trainer
    trainer.prepare_distributed_training(local_rank, local_size)

    trainer.run()
    distributed_cleanup()
