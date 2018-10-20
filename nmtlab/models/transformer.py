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

from nmtlab.modules.transformer_modules import TransformerEmbedding
from nmtlab.modules.transformer_modules import TemporalMasking
from nmtlab.modules.transformer_modules import TransformerEncoderLayer
from nmtlab.modules.transformer_modules import TransformerDecoderLayer


class Transformer(EncoderDecoderModel):
    """RNMT+ Model.

    Encoder: Transformer Encoder
    Decoder: Transformer Decoder
    Attention: Multihead Attention
    Other tricks: dropout, residual connection, layer normalization
    """
    
    def __init__(self, num_encoders=2, num_decoders=2, ff_size=None, dropout_ratio=0.1, **kwargs):
        """Create a RNMT+ Model.
        Args:
            num_encoders - Number of bidirectional encoders.
            num_decoders - Number of forward decoders.
            layer_norm - Using normal layer normalization.
        """
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self._ff_size = ff_size
        self._dropout_ratio = dropout_ratio
        super(Transformer, self).__init__(**kwargs)
        if self._src_vocab_size != self._tgt_vocab_size:
            raise ValueError("The vocabulary size shall be identical in both sides for transformer.")
    
    def prepare(self):
        # Layer Norm
        self.encoder_norm = nn.LayerNorm(self.hidden_size)
        self.decoder_norm = nn.LayerNorm(self.hidden_size)
        # Shared embedding layer
        self.src_embed_layer = TransformerEmbedding(self._src_vocab_size, self.embed_size)
        self.tgt_embed_layer = TransformerEmbedding(self._tgt_vocab_size, self.embed_size)
        self.temporal_mask = TemporalMasking()
        # Encoder
        self.encoder_layers = nn.ModuleList()
        for _ in range(self.num_encoders):
            layer = TransformerEncoderLayer(self.hidden_size, self._ff_size, dropout_ratio=self._dropout_ratio)
            self.encoder_layers.append(layer)
        # Decoder
        self.decoder_layers = nn.ModuleList()
        for _ in range(self.num_decoders):
            layer = TransformerDecoderLayer(self.hidden_size, self._ff_size, dropout_ratio=self._dropout_ratio)
            self.decoder_layers.append(layer)
        # Expander
        self.expander_nn = nn.Linear(self.hidden_size, self._tgt_vocab_size)
        # Decoding states need to be remembered for beam search
        state_names = ["embeddings"]
        for i in range(self.num_decoders):
            state_names.append("layer{}".format(i))
        self.set_states(state_names)
        self.set_stepwise_training(False)
    
    def encode(self, src_seq, src_mask=None):
        x = self.src_embed_layer(src_seq)
        for l, layer in enumerate(self.encoder_layers):
            x = layer(x, src_mask)
        encoder_states = self.encoder_norm(x)
        encoder_outputs = {
            "encoder_states": encoder_states,
            "src_mask": src_mask
        }
        return encoder_outputs
    
    def lookup_feedback(self, feedback):
        return self.tgt_embed_layer(feedback)
    
    def decode_step(self, context, states, full_sequence=False):
        if full_sequence:
            x = states.feedback_embed[:, :-1]
            temporal_mask = self.temporal_mask(x)
            for l, layer in enumerate(self.decoder_layers):
                x = layer(context.encoder_states, x, context.src_mask, temporal_mask)
            states["final_states"] = self.decoder_norm(x)
        else:
            feedback_embed = self.tgt_embed_layer(states.prev_token, start=states.t)  # ~ (batch, size)
            if states.t == 0:
                states.embeddings = feedback_embed.unsqueeze(1)
            else:
                states.embeddings = torch.cat([states.embeddings, feedback_embed.unsqueeze(1)], 1)
            x = states.embeddings
            for l, layer in enumerate(self.decoder_layers):
                x = layer(context.encoder_states, x, last_only=True)  # ~ (batch, 1, size)
                if states.t == 0:
                    states["layer{}".format(l)] = x
                else:
                    old_states = states["layer{}".format(l)]
                    states["layer{}".format(l)] = torch.cat([old_states, x], 1)
                    x = states["layer{}".format(l)]
            states["final_states"] = self.decoder_norm(x[:, -1].unsqueeze(0))  # ~ (1, batch ,size)
    
    def expand(self, states):
        return self.expander_nn(states.final_states)
