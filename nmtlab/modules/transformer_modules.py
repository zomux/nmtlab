#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from nmtlab.functions.gelu import gelu
from nmtlab.modules.multihead_attention import MultiHeadAttention
from nmtlab.functions.residual import residual_connect

class LabelSmoothingKLDivLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        self.padding_idx = ignore_index
        super(LabelSmoothingKLDivLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction="sum")


class TransformerEmbedding(nn.Embedding):
    """
    Rescale the embeddings.
    TODO: share the weight with pre-softmax linear transformation
    """
    
    def __init__(self, num_embeddings, embedding_dim, dropout_ratio=0.1):
        super(TransformerEmbedding, self).__init__(num_embeddings, embedding_dim)
        self.pos_layer = PositionalEmbedding(embedding_dim)
        self.dropout = nn.Dropout(dropout_ratio)
    
    def forward(self, x, start=None, positional_encoding=True):
        """
        Compute the embeddings with positional encoderi
        Args:
            x - input sequence ~ (batch, len)
            start - the begining position (option)
            positional_encoding - whether using positional encoding
        """
        embed = super(TransformerEmbedding, self).forward(x)
        embed = embed * math.sqrt(self.embedding_dim)
        if positional_encoding:
            if embed.dim() == 2:
                # Collapse one dimension of positional embedding
                pos_embed = self.pos_layer(embed.unsqueeze(1), start=start)
                pos_embed = pos_embed.squeeze(1)
            else:
                pos_embed = self.pos_layer(embed, start=start)
            embed += pos_embed
        return self.dropout(embed)
        

class TemporalMasking(nn.Module):
    """
    Produce (1, size, size) mask for masking out previous positions.
    """
    
    def __init__(self, max_len=1000):
        super(TemporalMasking, self).__init__()
        shape = (1, max_len, max_len)
        subsequent_mask = np.triu(np.ones(shape), k=1).astype('uint8')
        mask = (torch.from_numpy(subsequent_mask) == 0).float()
        self.register_buffer("mask", mask)
    
    def forward(self, x):
        """Compute the temporal mask on given embeddings
        
        Args:
            x - embedding ~ (batch, len, size)
        """
        if type(x) == int:
            seq_len = x
        else:
            seq_len = x.shape[-2]
        return self.mask[:, :seq_len, :seq_len]
        

class PositionalEmbedding(nn.Module):
    """
    This function is stealed from The Annotated Transformer (same as openNMT implementation).
    http://nlp.seas.harvard.edu/2018/04/03/attention.html#embeddings-and-softmax
    """
    
    def __init__(self, size, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, size, 2).float() *
                             -(math.log(10000.0) / size)).float())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x, start=None):
        """
        Return 3d tensor with shape (1, len, size).
        """
        if start is None:
            start = 0
        if type(x) == int:
            length = x
        else:
            length = x.shape[1]
        return Variable(self.pe[:, start:start + length], requires_grad=False)


class TransformerFeedForward(nn.Module):
    """The common feed-forward layer."""
    
    def __init__(self, size, hidden_size, dropout_ratio=0.1, activation="relu"):
        super(TransformerFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.dropout = nn.Dropout(dropout_ratio)
        if activation == "relu":
            self._activate = F.relu
        elif activation == "gelu":
            self._activate = gelu
        else:
            raise NotImplementedError
        
    def forward(self, x):
        return self.w_2(self.dropout(self._activate(self.w_1(x))))


class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, size, ff_size=None, n_att_head=8, dropout_ratio=0.1, relative_pos=False):
        super(TransformerEncoderLayer, self).__init__()
        if ff_size is None:
            ff_size = size * 4
        self.dropout = nn.Dropout(dropout_ratio)
        self.attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio, relative_pos=relative_pos)
        self.ff_layer = TransformerFeedForward(size, ff_size, dropout_ratio=dropout_ratio)
        self.layer_norm1 = nn.LayerNorm(size)
        self.layer_norm2 = nn.LayerNorm(size)

    def forward(self, x, src_mask=None):
        # Attention layer
        y1 = self.layer_norm1(x)
        y1, _ = self.attention(y1, y1, y1, mask=src_mask)
        y1 = self.dropout(y1)
        y1 = residual_connect(y1, x)
        # Feed-forward layer
        y2 = self.layer_norm2(y1)
        y2 = self.ff_layer(y2)
        y2 = self.dropout(y2)
        y2 = residual_connect(y2, y1)
        return y2


class TransformerDecoderLayer(nn.Module):
    
    def __init__(self, size, ff_size=None, n_att_head=8, dropout_ratio=0.1, relative_pos=False):
        super(TransformerDecoderLayer, self).__init__()
        if ff_size is None:
            ff_size = size * 4
        self.dropout = nn.Dropout(dropout_ratio)
        self.attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio, relative_pos=relative_pos)
        self.cross_attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio, relative_pos=relative_pos)
        self.ff_layer = TransformerFeedForward(size, ff_size, dropout_ratio=dropout_ratio)
        self.layer_norm1 = nn.LayerNorm(size)
        self.layer_norm2 = nn.LayerNorm(size)
        self.layer_norm3 = nn.LayerNorm(size)
    
    def forward(self, encoder_states, decoder_states, src_mask=None, tgt_mask=None, last_only=False):
        """
        Args:
            last_only - only compute the states for the last position
        """
        # Self-attention layer
        y1 = self.layer_norm1(decoder_states)
        if last_only:
            y1, _ = self.attention(y1[:, -1].unsqueeze(1), y1, y1, mask=tgt_mask)
            y1 = self.dropout(y1)
            y1 = residual_connect(y1, decoder_states[:, -1].unsqueeze(1))
        else:
            y1, _ = self.attention(y1, y1, y1, mask=tgt_mask)
            y1 = self.dropout(y1)
            y1 = residual_connect(y1, decoder_states)
        # Cross-attention layer
        y2 = self.layer_norm2(y1)
        y2, _ = self.attention(y2, encoder_states, encoder_states, mask=src_mask)
        y2 = self.dropout(y2)
        y2 = residual_connect(y2, y1)
        # Feed-forward layer
        y3 = self.layer_norm3(y2)
        y3 = self.ff_layer(y3)
        y3 = self.dropout(y3)
        y3 = residual_connect(y3, y2)
        return y3
