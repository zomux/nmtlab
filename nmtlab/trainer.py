#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
from collections import defaultdict
from six.moves import xrange

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

from nmtlab.models import EncoderDecoderModel
from nmtlab.utils import MTDataset, smoothed_bleu
from nmtlab.schedulers import Scheduler

MAX_EPOCH = 10000


class MTTrainer(object):
    """Training NMT models.
    """
    
    def __init__(self, model, dataset, optimizer, scheduler=None, multigpu=False):
        """Create a trainer.
        Args:
            model (EncoderDecoderModel): The model to train.
            dataset (MTDataset): Bilingual dataset.
            optimizer (Optimizer): Torch optimizer.
            scheduler (Scheduler): Training scheduler.
        """
        self._model = model
        self._dataset = dataset
        self._optimizer = optimizer
        self._scheduler = scheduler if scheduler is not None else Scheduler()
        self._multigpu = multigpu
        self._n_devices = 1
        # Setup horovod1i
        if multigpu:
            try:
                import horovod.torch as hvd
            except ImportError:
                raise SystemError("nmtlab requires horovod to run multigpu training.")
            # Initialize Horovod
            hvd.init()
            # Pin GPU to be used to process local rank (one GPU per process)
            torch.cuda.set_device(hvd.local_rank())
            self._model.cuda()
            self._optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=self._model.named_parameters())
            hvd.broadcast_parameters(self._model.state_dict(), root_rank=0)
            # Set the scope of training data
            self._dataset.set_gpu_scope(hvd.rank(), hvd.size())
            self._n_devices = hvd.size()
            import pdb;pdb.set_trace()
        else:
            self._model.cuda()
        # Initialize common variables
        self._log_lines = []
        self._scheduler.bind(self._model, self._optimizer)
        self._best_criteria = 65535
        self._n_train_batch = self._dataset.n_train_batch()
        self._batch_size = self._dataset.batch_size()
        self.configure()
        self._begin_time = 0
        # Print information
        self._log("nmtlab", "Training {} with {} parameters".format(
            self._model.__class__.__name__, len(list(self._model.named_parameters()))
        ))
        self._log("nmtlab", "with {} and {}".format(
            self._optimizer.__class__.__name__, self._scheduler.__class__.__name__
        ))
        self._log("nmtlab", "Training data has {} batches".format(self._dataset.n_train_batch()))
        self._log("nmtlab", "Running with {} GPUs".format(
            hvd.size() if multigpu else 1
        ))
    
    def configure(self, save_path=None, clip_norm=5, n_valid_per_epoch=10, criteria="bleu"):
        """Configure the hyperparameters of the trainer.
        """
        self._save_path = save_path
        self._clip_norm = clip_norm
        self._n_valid_per_epoch = n_valid_per_epoch
        self._criteria = criteria
        self._valid_freq = int(self._n_train_batch / self._n_valid_per_epoch)
        assert self._criteria in ("bleu", "loss")
        
    def run(self):
        """Run the training from begining to end.
        """
        for epoch in xrange(MAX_EPOCH):
            self._scheduler.before_epoch(epoch)
            self._begin_time = time.time()
            self._model.train(True)
            for step, batch in enumerate(self._dataset.train_set()):
                val_map = self.step(batch)
                self.check_valid(epoch, step)
                self.print_progress(epoch, step, val_map)
            self._scheduler.after_epoch(epoch)
            if self._scheduler.is_finished(epoch):
                break
            self._log("nmtlab", "Ending epoch {}, spent {} minutes  ".format(
                epoch + 1, int((time.time() - self._begin_time) / 60.)
            ))
    
    def step(self, batch):
        """Run one forward and backward step with given batch.
        """
        src_seq = batch.src.transpose(0, 1)
        tgt_seq = batch.tgt.transpose(0, 1)
        val_map = self._model(src_seq, tgt_seq)
        self._optimizer.zero_grad()
        val_map["loss"].backward()
        if self._clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_norm)
        self._optimizer.step()
        return val_map
        
    def check_valid(self, epoch, step):
        """Validate the model every few steps.
        """
        if step % self._valid_freq == 0 and self._is_root_node():
            self._model.train(False)
            score_map = self.run_valid()
            is_improved = self.check_improvement(score_map)
            self._scheduler.after_valid(epoch, step, is_improved, score_map)
            self._model.train(True)
            self._log("valid", "{}{} (epoch {}, step {})".format(
                self._dict_str(score_map), " *" if is_improved else "",
                epoch + 1, step + 1
            ))
    
    def run_valid(self):
        """Run the model on the validation set and report loss.
        """
        score_map = defaultdict(list)
        for batch in self._dataset.valid_set():
            src_seq = batch.src.transpose(0, 1)
            tgt_seq = batch.tgt.transpose(0, 1)
            with torch.no_grad():
                val_map = self._model(src_seq, tgt_seq, sampling=True)
                # Estimate BLEU
                if "sampled_tokens" in val_map:
                    bleu = self._compute_bleu(val_map["sampled_tokens"], tgt_seq)
                    score_map["bleu"].append(- bleu)
                    del val_map["sampled_tokens"]
                for k, v in val_map.items():
                    score_map[k].append(v)
        for key, vals in score_map.items():
            score_map[key] = np.mean(vals)
        return score_map
    
    def check_improvement(self, score_map):
        cri = score_map[self._criteria]
        if cri < self._best_criteria:
            self._best_criteria = cri
            self.save(0, 0)
            return True
        else:
            return False
    
    def print_progress(self, epoch, step, val_map):
        progress = int(float(step) / self._n_train_batch * 100)
        speed = float(step * self._batch_size) / (time.time() - self._begin_time) * self._n_devices
        sys.stdout.write("[epoch {}|{}%] loss={:.2f} | {:.1f} sample/s   \r".format(
            epoch + 1, progress, val_map["loss"], speed
        ))
        sys.stdout.flush()

    def _log(self, who, msg):
        line = "[{}] {}".format(who, msg)
        self._log_lines.append(line)
        if self._is_root_node():
            print(line)
    
    def save(self, epoch, step):
        state_dict = {
            "epoch": epoch,
            "step": step,
            "model_state": self._model.state_dict(),
            "optimizer_state": self._optimizer.state_dict()
        }
        if self._save_path is not None:
            torch.save(state_dict, self._save_path)
            open(self._save_path + ".log", "w").writelines([l + "\n" for l in self._log_lines])
    
    def load(self):
        state_dict = torch.load(self._save_path)
        self._model.load_state_dict(state_dict["model_state"])
        self._optimizer.load_state_dict(state_dict["optimizer_state"])
    
    @staticmethod
    def _compute_bleu(sampled_tokens, tgt_seq):
        """Compute smoothed BLEU of sampled tokens
        """
        bleus = []
        tgt_seq = tgt_seq.cpu().numpy()
        sampled_tokens = sampled_tokens.cpu().numpy()
        tgt_mask = np.greater(tgt_seq,  0)
        for i in xrange(tgt_seq.shape[0]):
            target_len = int(tgt_mask[i].sum())
            ref_tokens = tgt_seq[i, :target_len]
            out_tokens = sampled_tokens[i, :target_len]
            bleus.append(smoothed_bleu(out_tokens, ref_tokens))
        return np.mean(bleus)
    
    @staticmethod
    def _dict_str(rmap):
        return " ".join(
            ["{}={:.2f}".format(n, v) for n, v in rmap.items()]
        )
    
    def _is_root_node(self):
        if self._multigpu:
            import horovod.torch as hvd
            return hvd.local_rank() == 0
        else:
            return True

