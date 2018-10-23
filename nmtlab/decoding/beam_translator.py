#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.decoding.beam_search import BeamSearchKit

MAX_STEPS = 300


class BeamTranslator(BeamSearchKit):
    """Beam search translator with length normalization.
    """
    
    def beam_search(self, input_tokens, nbest=False):
        self.model.train(False)
        encoder_outputs = self.encode(input_tokens)
        # Create initial hyptheses
        hyps, final_hyps = self.initialize_hyps(encoder_outputs)
        
        for t in range(MAX_STEPS):
            # Make batch of states
            states = self.combine_states(t, hyps)
            
            # Decode one step
            new_states = self.decode_step(encoder_outputs, states)
            
            # Compute log probabilities
            new_scores = self.expand(new_states)
            
            # Expand to get new hypotheses for beam search
            hyps, final_hyps = self.get_new_hyps(
                hyps, final_hyps, new_states, batch_scores=new_scores)
            if len(final_hyps) == self.beam_size:
                break
        
        final_hyps.sort(key=lambda h: h["score"], reverse=True)
        
        if nbest:
            return final_hyps
        elif not final_hyps:
            return None, None
        else:
            best_hyp = final_hyps[0]
            score = best_hyp["score"]
            tokens = best_hyp["tokens"]
            return tokens, score
