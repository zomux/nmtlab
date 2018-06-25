#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nmtlab.decoding.beam_search import BeamSearchKit
from nmtlab.models import EncoderDecoderModel


class BeamTranslator(BeamSearchKit):
    
    def _encode(self, input_tokens):
    
    
    def beam_search(self, input_tokens, nbest=False):
        encoder_outputs = self.encoder_graph.compute([input_tokens])
        # Beam search for the tokens
        max_steps = 150
        hyps, final_hyps = self.init_hyps(
            init_state=encoder_outputs.init_state[0])
        
        for t in range(max_steps):
            # Run in batch mode
            batch_states = [hyp["state"] for hyp in hyps]
            batch_last_token = [hyp["tokens"][-1] for hyp in hyps]
            
            # Decode
            decoder_inputs = [t, batch_states, batch_last_token]
            decoder_outputs = self.decoder_graph.compute(*decoder_inputs)
            batch_new_states = decoder_outputs
            
            # Prob
            batch_logprobs = - np.log(self.expander_graph.compute(batch_new_states))
            
            new_hyps = self.expand_hyps(hyps, batch_new_states, batch_logprobs, expand_num=self.beam_size)
            
            new_hyps = new_hyps[:self.beam_size]
            # Get final hyps
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
            hyps = [h for h in new_hyps if h["tokens"][-1] != self.end_token_id][:self.beam_size - len(final_hyps)]
            if len(final_hyps) == self.beam_size:
                break
        
        final_hyps.sort(key=lambda h: h["logp"])
        
        if not final_hyps:
            return None, None
        
        best_hyp = final_hyps[0]
        if nbest:
            return final_hyps
        score = best_hyp["logp"]
        tokens = best_hyp["tokens"]
        return tokens, score
