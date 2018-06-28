#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.models import EncoderDecoderModel

import torch
import torch.nn as nn
import torch.nn.functionational as F

from nmtlab.modules import KeyValAttention


class DeepLSTMModel(EncoderDecoderModel):
    """Deep LSTM model with attention.
    
    Encoder: bidirectional LSTM
    Decoder: two-layer forward LSTM
    Attention: KeyValue Dot Attention
    Other tricks: dropout, residual connection
    """
    
    def prepare(self):
        self.src_embed_layer = nn.Embedding(self._src_vocab_size, self._embed_size)
        self.tgt_embed_layer = nn.Embedding(self._tgt_vocab_size, self._embed_size)
        self.encoder_rnn = nn.LSTM(self._embed_size, self._hidden_size, batch_first=True, bidirectional=True)
        self.decoder_rnn_1 = nn.LSTM(self._embed_size, self._hidden_size, batch_first=True)
        self.decoder_rnn_2 = nn.LSTM(self._hidden_size * 3, self._hidden_size, batch_first=True)
        self.init_hidden_nn_1 = nn.Linear(self._hidden_size, self._hidden_size)
        self.init_hidden_nn_2 = nn.Linear(self._hidden_size, self._hidden_size)
        self.attention_key_nn = nn.Linear(self._hidden_size * 2, self._hidden_size)
        self.attention = KeyValAttention()
        self.expander_nn = nn.Sequential(
            nn.Linear(self._hidden_size, int(self._hidden_size / 2)),
            nn.Linear(int(self._hidden_size / 2), self._tgt_vocab_size))
        self.set_states(["hidden1", "cell1", "hidden2", "cell2"], [self._hidden_size] * 4)
        self.set_autoregressive(False)
    
    def encode(self, src_seq, src_mask=None):
        src_embed = self.src_embed_layer(src_seq)
        encoder_states, (encoder_last_states, _) = self.encoder_rnn(src_embed)  # - B x N x s
        attention_keys = self.attention_key_nn(encoder_states)
        dec_init_hidden_1 = self.init_hidden_nn_1(encoder_last_states[1])
        dec_init_hidden_2 = self.init_hidden_nn_2(encoder_last_states[1])
        encoder_outputs = {
            "encoder_states": encoder_states,
            "keys": attention_keys,
            "init_hidden1": dec_init_hidden_1,
            "init_hidden2": dec_init_hidden_2,
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
        _, (states.hidden, states.cell) = self.decoder_rnn(dec_input[:, None, :], (states.hidden, states.cell))
        return states
    
    def expand(self, decoder_outputs):
        decoder_hiddens = decoder_outputs["hidden2"]
        logits = self.expander_nn(decoder_hiddens)
        return logits
