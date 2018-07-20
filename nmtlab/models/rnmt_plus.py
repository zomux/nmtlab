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


class RNMTPlusModel(EncoderDecoderModel):
    """RNMT+ Model.
    
    Encoder: Deep bidirectional LSTM
    Decoder: Deep forward LSTM
    Attention: Multihead Attention
    Other tricks: dropout, residual connection
    """

    def __init__(self, num_encoders=1, num_decoders=2,
                 hidden_size=512, embed_size=512,
                 src_vocab_size=None, tgt_vocab_size=None,
                 dataset=None,
                 state_names=None, state_sizes=None):
        """Create a RNMT+ Model.
        Args:
            num_encoders - Number of bidirectional encoders.
            num_decoders - Number of forward decoders.
        """
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        super(RNMTPlusModel, self).__init__(
            hidden_size, embed_size, src_vocab_size, tgt_vocab_size, dataset,
            state_names, state_sizes)
    
    def prepare(self):
        self.src_embed_layer = nn.Embedding(self._src_vocab_size, self._embed_size)
        self.tgt_embed_layer = nn.Embedding(self._tgt_vocab_size, self._embed_size)
        self.encoder_rnns = []
        for l in range(self.num_encoders):
            if l == 0:
                encoder_lstm = nn.LSTM(self._embed_size, self._hidden_size, batch_first=True, bidirectional=True)
            else:
                encoder_lstm = nn.LSTM(self._hidden_size, self._hidden_size, batch_first=True, bidirectional=True)
            self.encoder_rnns.append(encoder_lstm)
        self.decoder_rnns = []
        for l in range(self.num_decoders):
            decoder_lstm  = nn.LSTM(self._embed_size, self._hidden_size, batch_first=True)
            self.decoder_rnns.append(decoder_lstm)
        self.init_hidden_nn_1 = nn.Linear(self._hidden_size, self._hidden_size)
        self.init_hidden_nn_2 = nn.Linear(self._hidden_size, self._hidden_size)
        self.attention_key_nn = nn.Linear(self._hidden_size * 2, self._hidden_size)
        self.attention = KeyValAttention()
        self.dropout = nn.Dropout(0.2)
        self.expander_nn = nn.Sequential(
            nn.Linear(self._hidden_size, 600),
            nn.Linear(600, self._tgt_vocab_size))
        self.residual_scaler = torch.sqrt(torch.from_numpy(np.array(0.5, dtype="float32")))
        self.set_states(["hidden1", "cell1", "hidden2", "cell2"], [self._hidden_size] * 4)
        self.set_autoregressive(False)
        
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
        if full_sequence:
            feedback_embeds = states.feedback_embed[:, :-1]
            states.hidden1, _ = self.decoder_rnn_1(feedback_embeds)
            # Attention
            query = states.hidden1
            context_vector, _ = self.attention(
                query, context.keys, context.encoder_states,
                mask=context.mask
            )
            # Decoder layer 2
            decoder_input_2 = torch.cat([states.hidden1, context_vector], 2)
            states.hidden2, _ = self.decoder_rnn_2(decoder_input_2)
        else:
            feedback_embed = states.feedback_embed
            _, (states.hidden1, states.cell1) = self.decoder_rnn_1(feedback_embed[:, None, :], (states.hidden1, states.cell1))
            query = states.hidden1.squeeze(0)
            context_vector, _ = self.attention(
                query, context.keys, context.encoder_states,
                mask=context.mask
            )
            decoder_input_2 = torch.cat([query, context_vector], 1)
            _, (states.hidden2, states.cell2) = self.decoder_rnn_2(decoder_input_2[:, None, :], (states.hidden2, states.cell2))
    
    def expand(self, decoder_outputs):
        residual_hidden = self.residual_scaler * (decoder_outputs.hidden1 + decoder_outputs.hidden2)
        residual_hidden = self.dropout(residual_hidden)
        logits = self.expander_nn(residual_hidden)
        return logits
    
    def cuda(self, device=None):
        super(RNMTPlusModel, self).cuda(device)
        self.residual_scaler = self.residual_scaler.cuda()
