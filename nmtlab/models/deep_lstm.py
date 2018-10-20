#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.models import EncoderDecoderModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nmtlab.modules import KeyValAttention


class DeepLSTMModel(EncoderDecoderModel):
    """Deep LSTM model with attention.

    Encoder: bidirectional LSTM
    Decoder: two-layer forward LSTM
    Attention: KeyValue Dot Attention
    Other tricks: dropout, residual connection
    """

    def prepare(self):
        self.src_embed_layer = nn.Embedding(self._src_vocab_size, self.embed_size)
        self.tgt_embed_layer = nn.Embedding(self._tgt_vocab_size, self.embed_size)
        self.encoder_rnn = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.decoder_rnn_1 = nn.LSTM(self.embed_size + self.hidden_size * 2, self.hidden_size, batch_first=True)
        self.decoder_rnn_2 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.init_hidden_nn_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_hidden_nn_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_key_nn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attention = KeyValAttention()
        self.dropout = nn.Dropout(0.2)
        self.expander_nn = nn.Sequential(
            nn.Linear(self.hidden_size, 600),
            nn.Linear(600, self._tgt_vocab_size))
        self.residual_scaler = torch.sqrt(torch.from_numpy(np.array(0.5, dtype="float32")))
        self.set_states(["hidden1", "cell1", "hidden2", "cell2"])

    def encode(self, src_seq, src_mask=None):
        src_embed = self.src_embed_layer(src_seq)
        src_embed = self.dropout(src_embed)
        if src_mask is not None:
            src_embed = pack_padded_sequence(src_embed, lengths=src_mask.sum(1), batch_first=True)
        encoder_states, (encoder_last_states, _) = self.encoder_rnn(src_embed)  # - B x N x s
        if src_mask is not None:
            encoder_states, _ = pad_packed_sequence(encoder_states, batch_first=True)
        encoder_states = self.dropout(encoder_states)
        attention_keys = self.attention_key_nn(encoder_states)
        dec_init_hidden_1 = F.tanh(self.init_hidden_nn_1(encoder_last_states[1]))
        dec_init_hidden_2 = F.tanh(self.init_hidden_nn_2(encoder_last_states[1]))
        encoder_outputs = {
            "encoder_states": encoder_states,
            "keys": attention_keys,
            "init_hidden1": dec_init_hidden_1,
            "init_hidden2": dec_init_hidden_2,
            "src_mask": src_mask
        }
        return encoder_outputs

    def lookup_feedback(self, feedback):
        tgt_embed = self.tgt_embed_layer(feedback)
        tgt_embed = self.dropout(tgt_embed)
        return tgt_embed

    def decode_step(self, context, states, full_sequence=False):
        assert not full_sequence
        feedback_embed = states.feedback_embed
        last_dec_hidden = states.hidden1.squeeze(0)
        # Attention
        attention_query = last_dec_hidden + feedback_embed
        context_vector, _ = self.attention(
            attention_query, context.keys, context.encoder_states,
            mask=context.src_mask)
        # Decode
        dec_input = torch.cat((context_vector, feedback_embed), 1)
        _, (states.hidden1, states.cell1) = self.decoder_rnn_1(dec_input[:, None, :], (states.hidden1, states.cell1))
        dec2_input = self.dropout(states.hidden1.transpose(1, 0))
        _, (states.hidden2, states.cell2) = self.decoder_rnn_2(dec2_input, (states.hidden2, states.cell2))
        return states

    def expand(self, decoder_outputs):
        residual_hidden = self.residual_scaler * (decoder_outputs.hidden1 + decoder_outputs.hidden2)
        residual_hidden = self.dropout(residual_hidden)
        logits = self.expander_nn(residual_hidden)
        return logits

    def cuda(self, device=None):
        super(DeepLSTMModel, self).cuda(device)
        self.residual_scaler = self.residual_scaler.cuda()
