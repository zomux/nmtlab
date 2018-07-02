#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip

from abc import abstractmethod, ABCMeta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from nmtlab.utils import MapDict, LazyDict


class EncoderDecoderModel(nn.Module):
    
    __metaclass__ = ABCMeta

    def __init__(self, hidden_size=512, embed_size=512,
                 src_vocab_size=None, tgt_vocab_size=None,
                 dataset=None,
                 state_names=None, state_sizes=None):
        super(EncoderDecoderModel, self).__init__()
        if dataset is None and (src_vocab_size is None or tgt_vocab_size is None):
            raise SystemError("src_vocab_size and tgt_vocab_size must be specified.")
        self._hidden_size = hidden_size
        self._embed_size = embed_size
        self._autoregressive = True
        if dataset is not None:
            self._src_vocab_size, self._tgt_vocab_size = dataset.vocab_sizes()
        else:
            self._src_vocab_size = src_vocab_size
            self._tgt_vocab_size = tgt_vocab_size
        self._state_names = state_names if state_names else ["hidden", "cell"]
        self._state_sizes = state_sizes if state_sizes else [self._hidden_size] * len(
            self._state_names)
        self._monitors = {}
        self._layers = []
        self.prepare()
    
    def set_states(self, state_names, state_sizes):
        """Set state names and sizes for the decoder.
        """
        self._state_names = state_names
        self._state_sizes = state_sizes
        
    def set_autoregressive(self, flag=True):
        """Set whether the model is autoregressive when training.
        """
        self._autoregressive = flag

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
        if not self._autoregressive and not sampling:
            states.feedback_embed = self.lookup_feedback(context.feedback)
            return self.decode_step(context, states, full_sequence=True)
        else:
            T = context.feedbacks.shape[1]
            state_stack = []
            for t in range(T - 1):
                states = states.copy()
                states[t] = t
                if sampling:
                    if t == 0:
                        feedback = context.feedbacks[:, 0].unsqueeze(0)
                    else:
                        logits = self.expand(states)
                        feedback = logits.argmax(-1)
                    states.sampled_token = feedback
                    states.feedback_embed = self.lookup_feedback(feedback.squeeze(0))
                else:
                    states.feedback_embed = context.feedback_embeds[:, t]
                states = self.decode_step(context, states)
                state_stack.append(states)
            return self.combine_states(state_stack)

    def combine_states(self, state_stack):
        lazydict = LazyDict()
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
                states[state_name] = Variable(torch.zeros((1, B, size))).cuda()
        if extra_states is not None:
            extra_states.update(extra_states)
        # Process mask
        if src_mask is not None:
            context["src_mask"] = src_mask
        return context, states
        
    @abstractmethod
    def expand(self, decoder_outputs):
        """
        Expand decoder outputs to a vocab-size tensor.
        """
    
    def compute_loss(self, logits, tgt_seq, tgt_mask):
        B, T, _ = logits.shape
        logits = F.log_softmax(logits, dim=2)
        flat_logits = logits.contiguous().view(B * T, self._tgt_vocab_size)
        flat_targets = tgt_seq[:, 1:].contiguous().view(B * T)
        flat_mask = tgt_mask[:, 1:].contiguous().view(B * T)
        loss = nn.NLLLoss(ignore_index=0, reduce=False).forward(flat_logits, flat_targets)
        loss = (loss.view(B, T).sum(1) / (tgt_mask.sum(1) - 1).float()).mean()
        word_acc = (flat_logits.argmax(1).eq(flat_targets) * flat_mask).view(B, T).sum(1).float() / tgt_mask[:, 1:].sum(1).float()
        word_acc = word_acc.mean()
        self.monitor("word_acc", word_acc)
        return loss
    
    def monitor(self, key, value):
        """Monitor a value with the key.
        """
        self._monitors[key] = value
    
    def forward(self, src_seq, tgt_seq, sampling=False):
        """
        Forward to compute the loss.
        """
        src_mask = torch.ne(src_seq, 0)
        tgt_mask = torch.ne(tgt_seq, 0)
        encoder_outputs = MapDict(self.encode(src_seq, src_mask))
        context, states = self.pre_decode(encoder_outputs, tgt_seq, src_mask=src_mask, tgt_mask=tgt_mask)
        decoder_outputs = self.decode(context, states)
        logits = self.expand(decoder_outputs)
        if sampling:
            sample_outputs = self.decode(context, states, sampling=True)
            self.monitor("sampled_tokens", sample_outputs.sampled_token)
        loss = self.compute_loss(logits, tgt_seq, tgt_mask)
        self.monitor("loss", loss)
        return self._monitors

    def load(self, path):
        state_dict = torch.load(path)
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        self.load_state_dict(state_dict)
        
    def state_names(self):
        return self._state_names
    
    def state_sizes(self):
        return self._state_sizes
