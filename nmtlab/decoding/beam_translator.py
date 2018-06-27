#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nmtlab.decoding.beam_search import BeamSearchKit
from nmtlab.utils import MapDict

import torch
import torch.nn.functional as F

MAX_STEPS = 300


class BeamTranslator(BeamSearchKit):
    
    def beam_search(self, input_tokens, nbest=False):
        self.model.train(False)
        encoder_outputs = self.encode(input_tokens)
        # Beam search for the tokens
        hyps, final_hyps = self.init_hyps(encoder_outputs)
        
        for t in range(MAX_STEPS):
            # Run in batch mode
            states = self.combine_states(hyps)
            states.t = t
            
            # Decode
            batch_new_states = self.decode_step(encoder_outputs, states)
            
            # Prob
            batch_logprobs = self.expand(batch_new_states)
            
            new_hyps = self.expand_hyps(hyps, batch_new_states, batch_logprobs, expand_num=self.beam_size)
            
            new_hyps = new_hyps[:self.beam_size]
            # Get final hyps
            for i in range(len(new_hyps)):
                hyp = new_hyps[i]
                if hyp["tokens"][-1] == self.end_token_id:
                    tokens = hyp["tokens"][1:-1]
                    final_hyps.append({
                        "tokens": tokens,
                        "logp": hyp["logp"] / (len(tokens) + 1),
                        "raw": hyp
                    })
            # Update hyps
            hyps = [h for h in new_hyps if h["tokens"][-1] != self.end_token_id][:self.beam_size - len(final_hyps)]
            if len(final_hyps) == self.beam_size:
                break
        
        final_hyps.sort(key=lambda h: h["logp"], reverse=True)

        if not final_hyps:
            return None, None
        
        best_hyp = final_hyps[0]
        if nbest:
            return final_hyps
        score = best_hyp["logp"]
        tokens = best_hyp["tokens"]
        return tokens, score
    
    def combine_states(self, hyps):
        states = MapDict()
        # Combine states
        for name in self.model.state_names():
            states[name] = torch.cat([h["state"][name] for h in hyps], 1)
        # Combine last tokens
        last_tokens = torch.tensor([h["tokens"][-1] for h in hyps])
        if torch.cuda.is_available():
            last_tokens = last_tokens.cuda()
            states.feedback_embed = self.model.lookup_feedback(last_tokens)
        return states
        
    def encode(self, input_tokens):
        input_tensor = torch.tensor([input_tokens])
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        input_mask = torch.gt(input_tensor, 0)
        encoder_outputs = self.model.encode(input_tensor, input_mask)
        return MapDict(encoder_outputs)
    
    def decode_step(self, context, states):
        return self.model.decode_step(context, states)

    def expand(self, states):
        logits = self.model.expand(states).squeeze(0)
        logp = - F.log_softmax(logits, -1)
        return logp
