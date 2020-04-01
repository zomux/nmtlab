#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip

from abc import abstractmethod, ABCMeta
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from nmtlab.utils import MapDict, LazyTensorMap, TensorMap
from nmtlab.utils import OPTS


class EncoderDecoderModel(nn.Module):
    
    __metaclass__ = ABCMeta

    def __init__(self, hidden_size=512, embed_size=512,
                 src_vocab_size=None, tgt_vocab_size=None,
                 dataset=None,
                 state_names=None, state_sizes=None,
                 shard_size=32,
                 label_uncertainty=0,
                 fp16=False,
                 enable_valid_grad=False,
                 seed=3):
        super(EncoderDecoderModel, self).__init__()
        if dataset is None and (src_vocab_size is None or tgt_vocab_size is None):
            raise SystemError("src_vocab_size and tgt_vocab_size must be specified.")
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self._shard_size = shard_size
        self._stepwise_training = True
        self._label_uncertainty = label_uncertainty
        self._fp16 = fp16
        self.enable_valid_grad = enable_valid_grad
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if dataset is not None:
            self._src_vocab_size, self._tgt_vocab_size = dataset.vocab_sizes()
        else:
            self._src_vocab_size = src_vocab_size
            self._tgt_vocab_size = tgt_vocab_size
        self._state_names = state_names if state_names else ["hidden", "cell"]
        self._state_sizes = state_sizes if state_sizes else [self.hidden_size] * len(
            self._state_names)
        self._monitors = {}
        self._layers = []
        self.prepare()
        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize the parameters in the model."""
        # Initialize weights
        def get_fans(shape):
            fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
            fan_out = shape[1] if len(shape) == 2 else shape[0]
            return fan_in, fan_out
        for param in self.parameters():
            shape = param.shape
            if len(shape) > 1:
                nn.init.xavier_uniform_(param)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
            # Initilalize LSTM
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "bias" in name:
                        nn.init.constant_(param, 0.0)
                        n = param.size(0)
                        param.data[n//4: n//2].fill_(1.)
        if self._fp16:
            self.half()

    def set_states(self, state_names, state_sizes=None):
        """Set state names and sizes for the decoder.
        """
        self._state_names = state_names
        if state_sizes is not None:
            self._state_sizes = state_sizes
        else:
            self._state_sizes = [self.hidden_size] * len(state_names)
        
    def set_stepwise_training(self, flag=True):
        """Set whether the model is autoregressive when training.
        """
        self._stepwise_training = flag

    @abstractmethod
    def prepare(self):
        """Create layers.
        """

    @abstractmethod
    def encode(self, src_seq, src_mask=None):
        """Encode input sequence and return a value map.
        """

    @abstractmethod
    def lookup_feedback(self, feedback):
        """Get the word embeddings of feedback tokens.
        """

    @abstractmethod
    def decode_step(self, context, states, full_sequence=False):
        """Computations of each decoding step.
        """

    def decode(self, context, states, sampling=False):
        """Decode the output states.
        """
        if not self._stepwise_training and not sampling:
            states.feedback_embed = self.lookup_feedback(context.feedbacks)
            self.decode_step(context, states, full_sequence=True)
            return TensorMap(states)
        else:
            T = context.feedbacks.shape[1]
            state_stack = []
            steps = T + 9 if sampling else T - 1
            for t in range(steps):
                states = states.copy()
                states.t = t
                if sampling:
                    if t == 0:
                        feedback = context.feedbacks[:, 0].unsqueeze(0)
                    else:
                        logits = self.expand(states)
                        feedback = logits.argmax(-1)
                    states.prev_token = feedback
                    states.feedback_embed = self.lookup_feedback(feedback.squeeze(0))
                else:
                    states.prev_token = context.feedbacks[:, t].unsqueeze(0)
                    states.feedback_embed = context.feedback_embeds[:, t].unsqueeze(0)
                self.decode_step(context, states)
                state_stack.append(states)
            return self.combine_states(state_stack)

    def combine_states(self, state_stack):
        lazydict = LazyTensorMap()
        for state_name in state_stack[0]:
            tensor = state_stack[0][state_name]
            if hasattr(tensor, "shape") and len(tensor.shape) >= 2:
                lazydict[state_name] = lambda name: torch.cat([m[name] for m in state_stack], 0).transpose(1, 0)
        return lazydict
    
    def pre_decode(self, encoder_outputs, tgt_seq, extra_states=None, src_mask=None, tgt_mask=None):
        """Prepare the context and initial states for decoding.
        """
        feedback_embeds = self.lookup_feedback(tgt_seq)

        B = tgt_seq.shape[0]
        context = encoder_outputs
        states = MapDict({"t": 0})
        context["feedbacks"] = tgt_seq
        context["feedback_embeds"] = feedback_embeds
        # Process initial states
        for state_name, size in zip(self._state_names, self._state_sizes):
            if "init_{}".format(state_name) in context:
                states[state_name] = context["init_{}".format(state_name)]
                if len(states[state_name].shape) == 2:
                    states[state_name] = states[state_name].unsqueeze(0)
                del context["init_{}".format(state_name)]
            else:
                states[state_name] = Variable(torch.zeros((1, B, size)))
                if torch.cuda.is_available():
                    states[state_name] = states[state_name].cuda()
        if extra_states is not None:
            extra_states.update(extra_states)
        # Process mask
        context["src_mask"] = src_mask
        context["tgt_mask"] = tgt_mask
        return context, states
        
    @abstractmethod
    def expand(self, states):
        """
        Expand decoder outputs to a vocab-size tensor.
        """
    
    def compute_loss(self, logits, tgt_seq, tgt_mask, denominator=None, ignore_first_token=True):
        if self._label_uncertainty > 0 and self.training and not OPTS.marginloss:
            uniform_seq = self.to_float(tgt_seq).uniform_(0, self._tgt_vocab_size)
            smooth_mask = self.to_float(tgt_seq).bernoulli_(self._label_uncertainty)
            tgt_seq = (1 - smooth_mask) * self.to_float(tgt_seq) + smooth_mask * uniform_seq
            tgt_seq = tgt_seq.long()
        B, T, _ = logits.shape
        logits = F.log_softmax(logits, dim=2)
        flat_logits = logits.contiguous().view(B * T, self._tgt_vocab_size)
        if ignore_first_token:
            tgt_seq = tgt_seq[:, 1:]
            tgt_mask = tgt_mask[:, 1:]
        flat_targets = tgt_seq.contiguous().view(B * T)
        loss = nn.NLLLoss(ignore_index=0, reduce=False).forward(flat_logits, flat_targets)
        if OPTS.marginloss:
            correct_mask = self.to_float(flat_logits.argmax(1) == flat_targets)
            loss = (1 - correct_mask) * loss
        if denominator is None:
            loss = loss.sum() / self.to_float(tgt_mask.sum())
        else:
            loss = loss.sum() / denominator
        return loss
    
    def monitor(self, key, value):
        """Monitor a value with the key.
        """
        self._monitors[key] = value
    
    def forward(self, src_seq, tgt_seq, sampling=False):
        """Forward to compute the loss.
        """
        sampling = False
        src_mask = self.to_float(torch.ne(src_seq, 0))
        tgt_mask = self.to_float(torch.ne(tgt_seq, 0))
        encoder_outputs = MapDict(self.encode(src_seq, src_mask))
        context, states = self.pre_decode(encoder_outputs, tgt_seq, src_mask=src_mask, tgt_mask=tgt_mask)
        decoder_outputs = self.decode(context, states)
        if self._shard_size is not None and self._shard_size > 0:
            self.compute_shard_loss(decoder_outputs, tgt_seq, tgt_mask)
        else:
            logits = self.expand(decoder_outputs)
            loss = self.compute_loss(logits, tgt_seq, tgt_mask)
            acc = self.compute_word_accuracy(logits, tgt_seq, tgt_mask)
            self.monitor("loss", loss)
            self.monitor("word_acc", acc)
        if sampling:
            context, states = self.pre_decode(encoder_outputs, tgt_seq, src_mask=src_mask, tgt_mask=tgt_mask)
            sample_outputs = self.decode(context, states, sampling=True)
            self.monitor("sampled_tokens", sample_outputs.prev_token)
        return self._monitors
    
    def compute_word_accuracy(self, logits, tgt_seq, tgt_mask, denominator=None, ignore_first_token=True):
        """Compute per-word accuracy."""
        preds = logits.argmax(2)
        if ignore_first_token:
            tgt_seq = tgt_seq[:, 1:]
            tgt_mask = tgt_mask[:, 1:]
        if denominator is None:
            denominator = self.to_float(tgt_mask.sum())
        word_acc = (self.to_float(preds.eq(tgt_seq)) * tgt_mask).sum() / denominator
        return word_acc
    
    def compute_shard_loss(self, decoder_outputs, tgt_seq, tgt_mask, denominator=None, ignore_first_token=True,
                           backward=True):
        assert isinstance(decoder_outputs, TensorMap)
        is_grad_enabled = torch.is_grad_enabled()
        B = tgt_seq.shape[0]
        score_map = defaultdict(list)
        if denominator is None:
            if ignore_first_token:
                denom = tgt_mask[:, 1:].sum()
            else:
                denom = tgt_mask.sum()
        else:
            denom = denominator
        # Compute loss for each shard
        # The computation is performed on detached decoder states
        # Backpropagate the gradients to the deocder states
        OPTS.disable_backward_hooks = True
        for i in range(0, B, self._shard_size):
            j = i + self._shard_size
            decoder_outputs.select_batch(i, j, detach=True)
            logits = self.expand(decoder_outputs)
            loss = self.compute_loss(logits, tgt_seq[i:j], tgt_mask[i:j], denominator=denom,
                                     ignore_first_token=ignore_first_token)
            word_acc = self.compute_word_accuracy(logits, tgt_seq[i:j], tgt_mask[i:j], denominator=denom,
                                                  ignore_first_token=ignore_first_token)
            score_map["loss"].append(loss)
            score_map["word_acc"].append(word_acc)
            if i >= B - self._shard_size:
                # Enable the backward hooks to gather the gradients
                OPTS.disable_backward_hooks = False
            if is_grad_enabled:
                loss.backward()
        OPTS.disable_backward_hooks = False
        # Monitor scores
        monitors = {}
        for k in score_map:
            val = sum(score_map[k])
            self.monitor(k, val)
            monitors[k] = val
        # Backpropagate the gradients to all the parameters
        if is_grad_enabled:
            detached_items = list(decoder_outputs.get_detached_items().values())
            state_tensors = [x[1] for x in detached_items]
            grads = [x[0].grad for x in detached_items]
            if backward:
                torch.autograd.backward(state_tensors, grads)
        else:
            state_tensors, grads = None, None
        return monitors, state_tensors, grads

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        state_keys = list(state_dict.keys())
        for key in state_keys:
            if key.startswith("module."):
                new_key = key[7:]
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.load_state_dict(state_dict)
        
    def state_names(self):
        return self._state_names
    
    def state_sizes(self):
        return self._state_sizes

    def to_float(self, x):
        if self._fp16:
            return x.half()
        else:
            return x.float()
