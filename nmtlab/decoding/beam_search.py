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
    
    def init_hyps(self, encoder_outputs, items=None):
        final_hyps = []
        # hyp: state, tokens, sum of -log
        states = MapDict()
        for name, size in zip(self.model.state_names(), self.model.state_sizes()):
            if "init_{}".format(name) in encoder_outputs:
                states[name] = encoder_outputs["init_{}".format(name)]
                if len(states[name].shape) == 2:
                    states[name] = states[name].unsqueeze(0)
            else:
                states[name] = torch.zeros((1, 1, size))
                if torch.cuda.is_available():
                    states[name] = states[name].cuda()
        first_hyp = {
            "state": states,
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
    
    def expand_hyps(self, hyps, new_states, batch_scores, sort=True, expand_num=None):
        """
        Create B x B new hypotheses
        """
        if not expand_num:
            expand_num = self.beam_size
        new_hyps = []
        best_scores, best_tokens = batch_scores.topk(expand_num)
        for i, hyp in enumerate(hyps):
            new_hyp_state = MapDict()
            for sname in self.model.state_names():
                new_hyp_state[sname] = new_states[sname][:, i, :].unsqueeze(1)
            new_scores = best_scores[i].cpu().detach().numpy().tolist()
            new_tokens = best_tokens[i].cpu().detach().numpy().tolist()
            for new_token, new_score in zip(new_tokens, new_scores):
                new_hyp = {
                    "state": new_hyp_state,
                    "tokens": hyp["tokens"] + [new_token],
                    "logp": new_score + hyp["logp"],
                    "last_token_logp": new_score,
                    "old_state": hyp["state"]
                }
                new_hyp = self.fix_new_hyp(i, hyp, new_hyp)
                # Keep old information
                for key in hyp:
                    if key not in new_hyp:
                        new_hyp[key] = copy.copy(hyp[key])
                new_hyps.append(new_hyp)
        if sort:
            new_hyps.sort(key=lambda h: h["logp"], reverse=True)
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


