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

    def __init__(self, hidden_size=512, embed_size=512, src_vocab_size=40000, tgt_vocab_size=40000,
                 dataset=None,
                 decoder_states=None, decoder_state_sizes=None):
        super(EncoderDecoderModel, self).__init__()
        self._hidden_size = hidden_size
        self._embed_size = embed_size
        self._src_vocab_size = src_vocab_size
        self._tgt_vocab_size = tgt_vocab_size
        self._decoder_states = decoder_states if decoder_states else ["hidden", "cell"]
        self._decoder_state_sizes = decoder_state_sizes if decoder_state_sizes else [self._hidden_size] * len(
            self._decoder_states)
        self._layers = []
        self.prepare()

    @abstractmethod
    def prepare(self):
        """
        Create layers.
        """

    @abstractmethod
    def encode(self, src_seq, src_mask=None):
        """
        Encode input sequence and return a value map.
        """

    @abstractmethod
    def lookup_feedback(self, feedback):
        """
        Get the word embeddings of feedback tokens.
        """

    @abstractmethod
    def decode_step(self, context, states):
        """
        Computations of each decoding step.
        """

    def decode(self, context, states):
        """Decode the output states.
        """
        T = context.feedbacks.shape[1]
        state_stack = []
        for t in range(T - 1):
            states[t] = t
            states = self.decode_step(context, states)
            state_stack.append(states)
        return self.post_decode(state_stack)

    def post_decode(self, state_stack):
        lazydict = LazyDict()
        for state_name in self._decoder_states:
            lazydict[state_name] = lambda name: torch.cat([m[name] for m in state_stack], 0).permute(1, 0, 2)
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
        for state_name, size in zip(self._decoder_states, self._decoder_state_sizes):
            if "init_{}".format(state_name) in context:
                states[state_name] = context["init_{}".format(state_name)]
                if len(states[state_name].shape) == 2:
                    states[state_name] = states[state_name].unsqueeze(0)
                del context["init_{}".format(state_name)]
            else:
                states[state_name] = Variable(torch.zeros((1, B, self._hidden_size))).cuda()
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
        flat_logits = logits.resize(B * T, self._tgt_vocab_size)
        flat_targets = tgt_seq[:, 1:].resize(B * T)
        loss = nn.NLLLoss(ignore_index=0).forward(flat_logits, flat_targets)
        return loss
    
    def forward(self, src_seq, tgt_seq):
        """
        Forward to compute the loss.
        """
        src_mask = torch.ne(src_seq, 0)
        tgt_mask = torch.ne(tgt_seq, 0)
        encoder_outputs = MapDict(self.encode(src_seq, src_mask))
        context, states = self.pre_decode(encoder_outputs, tgt_seq, src_mask=src_mask, tgt_mask=tgt_mask)
        decoder_outputs = self.decode(context, states)
        logits = self.expand(decoder_outputs)

        loss = self.compute_loss(logits, tgt_seq, tgt_mask)
        return loss
