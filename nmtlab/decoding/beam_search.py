#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from nmtlab.models import EncoderDecoderModel
from nmtlab.utils import MapDict
import copy

import torch

class BeamSearchKit(object):
    
    def __init__(self, model, source_vocab, target_vocab, start_token="<s>", end_token="</s>", beam_size=5, opts=None):
        assert isinstance(model, EncoderDecoderModel)
        if torch.cuda.is_available():
            model.cuda()
        self.model = model
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.start_token = start_token
        self.end_token = end_token
        self.start_token_id = self.source_vocab.encode_token(start_token)
        self.end_token_id = self.target_vocab.encode_token(end_token)
        self.opts = MapDict(opts) if opts else opts
        self.beam_size = beam_size
        self.prepare()
    
    def prepare(self):
        """
        A prepraration function.
        """
    
    def preprocess(self, sentence):
        return self.source_vocab.encode(sentence.split())
    
    def postprocess(self, input, raw_result):
        return self.target_vocab.decode(raw_result)
    
    def translate(self, sentence):
        """
        Translate one sentence.
        :return: result, score
        """
        
        input_tokens = self.preprocess(sentence)
        result, score = self.beam_search(input_tokens)
        # Special case
        if result:
            result_words = self.postprocess(sentence, result)
            if not result_words:
                result_words.append("EMPTY")
            output_line = " ".join(result_words)
            return output_line, score
        else:
            return None, None
    
    def smart_init_hyps(self, encoder_outputs=None, items=None):
        final_hyps = []
        # hyp: state, tokens, sum of -log
        state = np.zeros((self.model.decoder_hidden_size(),), dtype="float32")
        for key, val in encoder_outputs.items():
            if key.startswith("init_"):
                real_key = key.replace("init_", "")
                idx = self.model._decoder_states.index(real_key)
                begin = sum(self.model._decoder_state_sizes[:idx], 0)
                end = sum(self.model._decoder_state_sizes[:idx + 1], 0)
                state[begin:end] = val[0]
        first_hyp = {
            "state": state,
            "tokens": [self.start_token_id],
            "logp": 0.
        }
        if items:
            first_hyp.update(items)
        hyps = [first_hyp]
        return hyps, final_hyps
    
    def init_hyps(self, init_state=None, items=None):
        final_hyps = []
        # hyp: state, tokens, sum of -log
        state = np.zeros((self.model.decoder_hidden_size(),), dtype="float32")
        if init_state is not None:
            state[:init_state.shape[0]] = init_state
        first_hyp = {
            "state": state,
            "tokens": [self.start_token_id],
            "logp": 0.
        }
        if items:
            first_hyp.update(items)
        hyps = [first_hyp]
        return hyps, final_hyps
    
    def fix_new_hyp(self, i, hyp, new_hyp):
        """
        Modify new hyp in the expansion.
        """
        return new_hyp
    
    def expand_hyps(self, hyps, batch_new_states, batch_scores, sort=True, expand_num=None):
        """
        Create B x B new hypotheses
        """
        if not expand_num:
            expand_num = self.beam_size
        new_hyps = []
        for i, hyp in enumerate(hyps):
            new_state = batch_new_states[i]
            logprob = batch_scores[i] + hyp["logp"]
            best_indices = sorted(
                np.argpartition(logprob, expand_num)[:expand_num], key=lambda x: logprob[x])
            for idx in best_indices:
                new_hyp = {
                    "state": new_state,
                    "tokens": hyp["tokens"] + [idx],
                    "logp": logprob[idx],
                    "last_token_logp": batch_scores[i][idx],
                    "old_state": hyp["state"]
                }
                new_hyp = self.fix_new_hyp(i, hyp, new_hyp)
                # Keep old information
                for key in hyp:
                    if key not in new_hyp:
                        new_hyp[key] = copy.copy(hyp[key])
                new_hyps.append(new_hyp)
        if sort:
            new_hyps.sort(key=lambda h: h["logp"])
        return new_hyps
    
    def truncate_hyps(self, new_hyps, final_hyps=None):
        """
        Collect finished hyps and truncate.
        """
        # Get final hyps
        if final_hyps is not None:
            for i in range(len(new_hyps)):
                hyp = new_hyps[i]
                if hyp["tokens"][-1] == self.end_token_id:
                    tokens = hyp["tokens"][1:-1]
                    final_hyps.append({
                        "tokens": tokens,
                        "logp": hyp["logp"] / len(tokens),
                        "raw": hyp
                    })
        # Update hyps
        hyps = [h for h in new_hyps if h["tokens"][-1] != self.end_token_id][:self.beam_size]
        return hyps, final_hyps
    
    def update_hyps(self, hyps, final_hyps, batch_new_states, batch_scores):
        """
        Expand and Truncate hypotheses.
        """
        new_hyps = self.expand_hyps(hyps, batch_new_states, batch_scores)
        hyps, final_hyps = self.truncate_hyps(new_hyps, final_hyps)
        return hyps, final_hyps
    
    def beam_search(self, input_tokens):
        raise NotImplementedError
        return None, None


