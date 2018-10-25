#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F

from nmtlab.models import EncoderDecoderModel
from nmtlab.utils import MapDict
import copy


class BeamSearchKit(object):
    
    __metaclass__ = ABCMeta
    
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
        """Prepare callback.
        
        This function will be execute at the begining of beam search.
        """
    
    def preprocess(self, sentence):
        """Preprocessing callback.
        """
        return self.source_vocab.encode(sentence.split())
    
    def postprocess(self, input, raw_result):
        """Post-processing callback.
        """
        return self.target_vocab.decode(raw_result)
    
    def translate(self, sentence, greedy=False):
        """Translate one sentence.
        """
        self.model.train(False)
        input_tokens = self.preprocess(sentence)
        with torch.no_grad():
            if greedy:
                result, score = self.greedy_search(input_tokens)
            else:
                result, score = self.beam_search(input_tokens)
        if result:
            result_words = self.postprocess(sentence, result)
            output_line = " ".join(result_words)
            return output_line, score
        else:
            # When failed
            return None, None
    
    def greedy_search(self, input_tokens):
        """Bypass beam search, as a way to verify the output.
        """
        
        src_seq = torch.tensor(input_tokens).unsqueeze(0)
        if torch.cuda.is_available():
            src_seq = src_seq.cuda()
        self.model.train(False)
        with torch.no_grad():
            val_map = self.model(src_seq, torch.ones_like(src_seq), sampling=True)
        sampled_tokens = val_map["sampled_tokens"]
        score = float(val_map["loss"].cpu())
        sampled_tokens = sampled_tokens[0].cpu().numpy().tolist()[1:]
        if self.end_token_id in sampled_tokens:
            sampled_tokens = sampled_tokens[:sampled_tokens.index(self.end_token_id)]
        return sampled_tokens, score
        
    def initialize_hyps(self, encoder_outputs, items=None):
        """Initialize the first hypothesis for beam search.
        """
        final_hyps = []
        states = MapDict()
        # Create initial states
        for name, size in zip(self.model.state_names(), self.model.state_sizes()):
            if "init_{}".format(name) in encoder_outputs:
                states[name] = encoder_outputs["init_{}".format(name)]
                if len(states[name].shape) == 2:
                    states[name] = states[name].unsqueeze(0)
            else:
                states[name] = torch.zeros((1, 1, size))
                if torch.cuda.is_available():
                    states[name] = states[name].cuda()
        # Create first hypthesis
        first_hyp = {
            "state": states,
            "tokens": [self.start_token_id],
            "score": 0.
        }
        if items:
            first_hyp.update(items)
        hyps = [first_hyp]
        return hyps, final_hyps
    
    def fix_new_hyp(self, i, hyp, new_hyp):
        """Modify a created hypothesis in the expansion.
        """
        return new_hyp
    
    def expand_hyps(self, hyps, new_states, batch_scores, sort=True, expand_num=None):
        """Create B x B new hypotheses
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
                    "score": new_score + hyp["score"],
                    "last_token_score": new_score,
                    "old_state": hyp["state"]
                }
                new_hyp = self.fix_new_hyp(i, hyp, new_hyp)
                # Keep old information
                for key in hyp:
                    if key not in new_hyp:
                        new_hyp[key] = copy.copy(hyp[key])
                new_hyps.append(new_hyp)
        if sort:
            new_hyps.sort(key=lambda h: h["score"], reverse=True)
        return new_hyps
    
    def collect_finished_hyps(self, new_hyps, final_hyps=None):
        """
        Collect finished hyps and truncate.
        """
        # Get final hyps
        for i in range(len(new_hyps)):
            hyp = new_hyps[i]
            if hyp["tokens"][-1] == self.end_token_id:
                # This hypothesis is finished, remove <s> and </s>
                tokens = hyp["tokens"][1:-1]
                final_hyps.append({
                    "tokens": tokens,
                    "score": hyp["score"] / (len(tokens) + 1),
                    "raw": hyp
                })
        # Remove finished hypotheses
        hyps = [
            h for h in new_hyps
            if h["tokens"][-1] != self.end_token_id][:self.beam_size - len(final_hyps)]
        return hyps, final_hyps
    
    def get_new_hyps(self, hyps, final_hyps, batch_new_states, batch_scores):
        """Expand hypothesis to get new hypotheses.
        
        Returns:
            New hypotheses and finished hypotheses.
        """
        new_hyps = self.expand_hyps(
            hyps, batch_new_states, batch_scores, expand_num=self.beam_size)
        new_hyps = new_hyps[:self.beam_size]
        hyps, final_hyps = self.collect_finished_hyps(new_hyps, final_hyps)
        return hyps, final_hyps

    def combine_states(self, t, hyps):
        """Batch all states in different hyptheses.
        Args:
            t - time step
            hyps - hypotheses
        """
        states = MapDict({"t": t})
        # Combine states
        for name in self.model.state_names():
            states[name] = torch.cat([h["state"][name] for h in hyps], 1)
        # Combine last tokens
        last_tokens = torch.tensor([h["tokens"][-1] for h in hyps])
        if torch.cuda.is_available():
            last_tokens = last_tokens.cuda()
            states.prev_token = last_tokens.unsqueeze(0)
            states.feedback_embed = self.model.lookup_feedback(last_tokens)
        return states

    def encode(self, input_tokens):
        """Run the encoder to get context.
        """
        input_tensor = torch.tensor([input_tokens])
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        input_mask = torch.gt(input_tensor, 0)
        encoder_outputs = self.model.encode(input_tensor, input_mask)
        return MapDict(encoder_outputs)

    def decode_step(self, context, states):
        """Run one-step in decoder.
        """
        self.model.decode_step(context, states)
        return states

    def expand(self, states):
        """Expand the decoder states to get log probabilities.
        """
        logits = self.model.expand(states).squeeze(0)
        logp = F.log_softmax(logits, -1)
        return logp
    
    @abstractmethod
    def beam_search(self, input_tokens):
        """An abstract beam search method to be implement.
        """
