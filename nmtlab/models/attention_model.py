#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.models import EncoderDecoderModel
from nmtlab.modules import KeyValAttention

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModel(EncoderDecoderModel):
    """Attention-based NMT with dot attention.
    
    Encoder: bidirectional LSTM
    Decoder: one-layer forward LSTM
    Attention: KeyValue Dot Attention
    """
    
    def prepare(self):
        self.src_embed_layer = nn.Embedding(self._src_vocab_size, self.embed_size)
        self.tgt_embed_layer = nn.Embedding(self._tgt_vocab_size, self.embed_size)
        self.encoder_rnn = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True, bidirectional=True)
        # self.decoder_rnn = nn.LSTM(self._hidden_size * 2 + self._embed_size, self._hidden_size, batch_first=True)
        self.decoder_rnn = nn.LSTMCell(self.hidden_size * 2 + self.embed_size, self.hidden_size)
        self.init_hidden_nn = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_key_nn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attention = KeyValAttention()
        self.expander_nn = nn.Sequential(
            nn.Linear(self.hidden_size, 600),
            nn.Linear(600, self._tgt_vocab_size))

    def encode(self, src_seq, src_mask=None):
        src_embed = self.src_embed_layer(src_seq)
        encoder_states, (encoder_last_states, _) = self.encoder_rnn(src_embed)  # - B x N x s
        attention_keys = self.attention_key_nn(encoder_states)
        dec_init_hidden = F.tanh(self.init_hidden_nn(encoder_last_states[1]))
        encoder_outputs = {
            "encoder_states": encoder_states,
            "keys": attention_keys,
            "init_hidden": dec_init_hidden,
            "src_mask": src_mask
        }
        return encoder_outputs

    def lookup_feedback(self, feedback):
        return self.tgt_embed_layer(feedback)

    def decode_step(self, context, states, full_sequence=False):
        feedback_embed = states.feedback_embed
        last_dec_hidden = states.hidden.squeeze(0)
        # Attention
        attention_query = last_dec_hidden + feedback_embed
        context_vector, _ = self.attention(
            attention_query, context.keys, context.encoder_states,
            mask=context.src_mask)
        # Decode
        dec_input = torch.cat((context_vector, feedback_embed), 1)
        states.hidden, states.cell = self.decoder_rnn(dec_input, (states.hidden[0], states.cell[0]))
        states.hidden = states.hidden.unsqueeze(0)
        states.cell= states.cell.unsqueeze(0)
        return states

    def expand(self, decoder_outputs):
        decoder_hiddens = decoder_outputs["hidden"]
        logits = self.expander_nn(decoder_hiddens)
        return logits
