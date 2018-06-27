#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.models import EncoderDecoderModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModel(EncoderDecoderModel):
    """Attention-based NMT with dot attention.
    """
    
    def prepare(self):
        self.src_embed_layer = nn.Embedding(self._src_vocab_size, self._embed_size)
        self.tgt_embed_layer = nn.Embedding(self._tgt_vocab_size, self._embed_size)
        self.encoder_rnn = nn.LSTM(self._embed_size, self._hidden_size, batch_first=True, bidirectional=True)
        self.decoder_rnn = nn.LSTM(self._hidden_size* 2 + self._embed_size, self._hidden_size, batch_first=True)
        self.init_hidden_nn = nn.Linear(self._hidden_size, self._hidden_size)
        self.attention_key_nn = nn.Linear(self._hidden_size * 2, self._hidden_size)
        self.expander_nn = nn.Sequential(
            nn.Linear(self._hidden_size, int(self._hidden_size / 2)),
            nn.Linear(int(self._hidden_size / 2), self._tgt_vocab_size))

    def encode(self, src_seq, src_mask=None):
        src_embed = self.src_embed_layer(src_seq)
        encoder_states, (encoder_last_states, _) = self.encoder_rnn(src_embed)  # - B x N x s
        attention_keys = self.attention_key_nn(encoder_states)
        dec_init_hidden = self.init_hidden_nn(encoder_last_states[1])
        encoder_outputs = {
            "encoder_states": encoder_states,
            "keys": attention_keys,
            "init_hidden": dec_init_hidden,
            "attention_penalty":  (1 - src_mask.float()) * 99
        }
        return encoder_outputs

    def lookup_feedback(self, feedback):
        return self.tgt_embed_layer(feedback)

    def decode_step(self, context, states, full_sequence=False):
        feedback_embed = states.feedback_embed
        last_dec_hidden = states.hidden.squeeze(0)
        # Attention
        attention_query = last_dec_hidden + feedback_embed
        attention_logits = (attention_query[:, None, :] * context.keys).sum(dim=2)
        attention_logits -= context.attention_penalty
        attention_weights = F.softmax(attention_logits, dim=1)
        encoder_states = context.encoder_states.expand([attention_weights.shape[0]] + list(context.encoder_states.shape)[1:])
        context_vector = torch.bmm(attention_weights[:, None, :], encoder_states).squeeze(1)
        # Decode
        dec_input = torch.cat((context_vector, feedback_embed), 1)
        _, (states.hidden, states.cell) = self.decoder_rnn(dec_input[:, None, :], (states.hidden, states.cell))
        return states

    def expand(self, decoder_outputs):
        decoder_hiddens = decoder_outputs["hidden"]
        logits = self.expander_nn(decoder_hiddens)
        return logits
