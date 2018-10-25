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

from nmtlab.modules import MultiHeadAttention


class RNMTPlusModel(EncoderDecoderModel):
    """RNMT+ Model.
    
    Encoder: Deep bidirectional LSTM
    Decoder: Deep forward LSTM
    Attention: Multihead Attention
    Other tricks: dropout, residual connection, layer normalization
    TODO: per gate layer normlaization
    """

    def __init__(self, num_encoders=1, num_decoders=2, layer_norm=False, **kwargs):
        """Create a RNMT+ Model.
        Args:
            num_encoders - Number of bidirectional encoders.
            num_decoders - Number of forward decoders.
            layer_norm - Using normal layer normalization.
        """
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.layer_norm = layer_norm
        super(RNMTPlusModel, self).__init__(**kwargs)
    
    def prepare(self):
        # Embedding layers
        self.src_embed_layer = nn.Embedding(self._src_vocab_size, self.embed_size)
        self.tgt_embed_layer = nn.Embedding(self._tgt_vocab_size, self.embed_size)
        # Encoder
        self.encoder_rnns = []
        for l in range(self.num_encoders):
            if l == 0:
                encoder_lstm = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True, bidirectional=True)
            else:
                encoder_lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, batch_first=True, bidirectional=True)
            setattr(self, "encoder_rnn{}".format(l + 1), encoder_lstm)
            self.encoder_rnns.append(encoder_lstm)
        self.project_nn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # Decoder
        self.decoder_rnns = []
        for l in range(self.num_decoders):
            if l == 0:
                decoder_lstm = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True)
            else:
                decoder_lstm = nn.LSTM(self.embed_size + self.hidden_size, self.hidden_size, batch_first=True)
            setattr(self, "decoder_rnn{}".format(l + 1), decoder_lstm)
            self.decoder_rnns.append(decoder_lstm)
        self.attention = MultiHeadAttention(self.hidden_size, num_head=4, additive=False)
        self.dropout = nn.Dropout(0.2)
        self.expander_nn = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 600),
            nn.Linear(600, self._tgt_vocab_size))
        self.residual_scaler = torch.sqrt(torch.from_numpy(np.array(0.5, dtype="float32")))
        state_names = ["context", "final_hidden"]
        for i in range(self.num_decoders):
            state_names.append("hidden{}".format(i + 1))
            state_names.append("cell{}".format(i + 1))
        self.set_states(state_names, [self.hidden_size] * (self.num_decoders * 2 + 2))
        self.set_stepwise_training(False)
        
    def encode(self, src_seq, src_mask=None):
        src_embed = self.src_embed_layer(src_seq)
        src_embed = self.dropout(src_embed)
        enc_states = src_embed
        for l, rnn in enumerate(self.encoder_rnns):
            prev_states = enc_states
            if src_mask is not None:
                prev_states = pack_padded_sequence(prev_states, lengths=src_mask.sum(1), batch_first=True)
            enc_states, (enc_last_hidden, _) = rnn(prev_states)
            if src_mask is not None:
                enc_states, _ = pad_packed_sequence(enc_states, batch_first=True)
            enc_states = self.dropout(enc_states)
            if l >= 2:
                enc_states = self.residual_scaler * (enc_states + prev_states)
            if self.layer_norm:
                enc_states = F.layer_norm(enc_states, (self.hidden_size * 2,))
        enc_states = self.project_nn(enc_states)
        if self.layer_norm:
            enc_states = F.layer_norm(enc_states, (self.hidden_size,))
        encoder_outputs = {
            "encoder_states": enc_states,
            "keys": enc_states,
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
            dec_states = None
            for l, rnn in enumerate(self.decoder_rnns):
                if l == 0:
                    dec_states, _ = rnn(feedback_embeds)
                    if self.layer_norm:
                        dec_states = F.layer_norm(dec_states, (self.hidden_size,))
                    # Attention
                    states.context, _ = self.attention(dec_states, context.keys, context.encoder_states, mask=context.src_mask)
                else:
                    prev_states = dec_states
                    dec_input = torch.cat([prev_states, states.context], 2)
                    dec_states, _ = rnn(dec_input)
                    dec_states = self.dropout(dec_states)
                    if l >= 2:
                        dec_states = self.residual_scaler * (dec_states + prev_states)
                    if self.layer_norm:
                        dec_states = F.layer_norm(dec_states, (self.hidden_size,))
        else:
            feedback_embed = states.feedback_embed
            dec_states = None
            for l, rnn in enumerate(self.decoder_rnns):
                lstm_state = (getattr(states, "hidden{}".format(l + 1)), getattr(states, "cell{}".format(l + 1)))
                if l == 0:
                    _, (states.hidden1, states.cell1) = rnn(feedback_embed.transpose(0, 1), lstm_state)
                    dec_states = states.hidden1
                    if self.layer_norm:
                        dec_states = F.layer_norm(dec_states, (self.hidden_size,))
                    # Attention
                    states.context, _ = self.attention(dec_states.squeeze(0), context.keys, context.encoder_states, mask=context.src_mask)
                    states.context = states.context.unsqueeze(0)
                else:
                    prev_states = dec_states
                    dec_input = torch.cat([prev_states, states.context], 2)
                    _, (hidden, cell) = rnn(dec_input.transpose(1, 0), lstm_state)
                    dec_states = self.dropout(hidden)
                    if l >= 2:
                        dec_states = self.residual_scaler * (dec_states + prev_states)
                    if self.layer_norm:
                        dec_states = F.layer_norm(dec_states, (self.hidden_size,))
                    states["hidden{}".format(l + 1)] = hidden
                    states["cell{}".format(l + 1)] = cell
        states["final_hidden"] = dec_states

    def expand(self, states):
        last_dec_states = states.final_hidden
        softmax_input = torch.cat([last_dec_states, states.context], -1)
        logits = self.expander_nn(softmax_input)
        return logits
    
    def cuda(self, device=None):
        super(RNMTPlusModel, self).cuda(device)
        self.residual_scaler = self.residual_scaler.cuda()
