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
    
    def __init__(self, num_encoders=3, num_decoders=3, ff_size=None, n_att_heads=2, dropout_ratio=0.1, relative_pos=False, **kwargs):
        """Create a RNMT+ Model.
        Args:
            num_encoders - Number of bidirectional encoders.
            num_decoders - Number of forward decoders.
            layer_norm - Using normal layer normalization.
            relative_pos - Computing relative positional representations in attention
        """
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self._ff_size = ff_size
        self._n_att_heads = n_att_heads
        self._dropout_ratio = dropout_ratio
        self._relative_pos = relative_pos
        super(Transformer, self).__init__(**kwargs)
        # if self._src_vocab_size != self._tgt_vocab_size:
        #     raise ValueError("The vocabulary size shall be identical in both sides for transformer.")
        if ff_size is None:
            self._ff_size = self.hidden_size * 4
    
    def prepare(self):
        from nmtlab.modules.transformer_modules import LabelSmoothingKLDivLoss
        self.label_smooth = LabelSmoothingKLDivLoss(0.1, self._tgt_vocab_size, 0)
        # Layer Norm
        self.encoder_norm = nn.LayerNorm(self.hidden_size)
        self.decoder_norm = nn.LayerNorm(self.hidden_size)
        # Shared embedding layer
        self.src_embed_layer = TransformerEmbedding(self._src_vocab_size, self.embed_size, dropout_ratio=self._dropout_ratio)
        self.tgt_embed_layer = TransformerEmbedding(self._tgt_vocab_size, self.embed_size, dropout_ratio=self._dropout_ratio)
        self.temporal_mask = TemporalMasking()
        # Encoder
        self.encoder_layers = nn.ModuleList()
        for _ in range(self.num_encoders):
            layer = TransformerEncoderLayer(self.hidden_size, self._ff_size,
                                            n_att_head=self._n_att_heads, dropout_ratio=self._dropout_ratio,
                                            relative_pos=self._relative_pos)
            self.encoder_layers.append(layer)
        # Decoder
        self.decoder_layers = nn.ModuleList()
        for _ in range(self.num_decoders):
            layer = TransformerDecoderLayer(self.hidden_size, self._ff_size,
                                            n_att_head=self._n_att_heads, dropout_ratio=self._dropout_ratio,
                                            relative_pos=self._relative_pos)
            self.decoder_layers.append(layer)
        # Expander
        self.expander_nn = nn.Linear(self.hidden_size, self._tgt_vocab_size)
        # Decoding states need to be remembered for beam search
        state_names = ["embeddings"]
        for i in range(self.num_decoders):
            state_names.append("layer{}".format(i))
        self.set_states(state_names)
        self.set_stepwise_training(False)
    
    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
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
    
    def compute_loss(self, logits, tgt_seq, tgt_mask, denominator=None, ignore_first_token=True):
        if self._label_uncertainty > 0:
            return super(Transformer, self).compute_loss(logits, tgt_seq, tgt_mask, denominator)
        B, T, _ = logits.shape
        logits = F.log_softmax(logits, dim=2)
        flat_logits = logits.contiguous().view(B * T, self._tgt_vocab_size)
        if ignore_first_token:
            tgt_seq = tgt_seq[:, 1:]
            tgt_mask = tgt_mask[:, 1:]
        flat_targets = tgt_seq.contiguous().view(B * T)
        flat_mask = tgt_mask.contiguous().view(B * T)
        if denominator is None:
            denominator = flat_mask.sum()
        loss = self.label_smooth(flat_logits, flat_targets) / denominator
        return loss
    
    def lookup_feedback(self, feedback):
        return self.tgt_embed_layer(feedback)
    
    def decode_step(self, context, states, full_sequence=False):
        if full_sequence:
            # During training: full sequence mode
            x = states.feedback_embed[:, :-1]
            temporal_mask = self.temporal_mask(x)
            # print("full embed", x[1, :, :2])
            for l, layer in enumerate(self.decoder_layers):
                x = layer(context.encoder_states, x, context.src_mask, temporal_mask)
                # print("full {}".format(l), x[1, :, :2])
            states["final_states"] = self.decoder_norm(x)
        else:
            # During beam search: stepwise mode
            feedback_embed = self.tgt_embed_layer(states.prev_token.transpose(0, 1), start=states.t).transpose(0, 1)  # ~ (batch, size)
            # print("embed", feedback_embed[0, 1, :2])
            if states.t == 0:
                states.embeddings = feedback_embed
            else:
                states.embeddings = torch.cat([states.embeddings, feedback_embed], 0)
            x = states.embeddings.transpose(0, 1)
            for l, layer in enumerate(self.decoder_layers):
                x = layer(context.encoder_states, x, last_only=True)  # ~ (batch, 1, size)
                if states.t == 0:
                    states["layer{}".format(l)] = x.transpose(0, 1)
                else:
                    old_states = states["layer{}".format(l)]
                    states["layer{}".format(l)] = torch.cat([old_states, x.transpose(0, 1)], 0)
                    x = states["layer{}".format(l)].transpose(0, 1)
            states["final_states"] = self.decoder_norm(x[:, -1].unsqueeze(0))  # ~ (1, batch ,size)
    
    def expand(self, states):
        return self.expander_nn(states.final_states)
