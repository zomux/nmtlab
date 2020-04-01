#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import sys
import os

import torch
import torch.nn.functional as F

from nmtlab.models import EncoderDecoderModel
from nmtlab.utils import MapDict, is_root_node, OPTS
import copy


class BeamSearchKit(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, model, source_vocab, target_vocab, start_token="<s>", end_token="</s>", beam_size=5, length_norm=False, opts=None, device=None):
        assert isinstance(model, EncoderDecoderModel)
        # Iniliatize horovod for multigpu translate
        self._is_multigpu = False
        try:
            import horovod.torch as hvd
            hvd.init()
            if torch.cuda.is_available():
                torch.cuda.set_device(hvd.local_rank())
                self._is_multigpu = True
        except ImportError:
            pass
        if torch.cuda.is_available():
            model.cuda(device)
        self.length_norm = length_norm
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
                if new_hyp is None:
                    continue
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
                if self.length_norm:
                    lp = (float(5 + len(tokens)) ** 0.6) / (6. ** 0.6)
                    score = hyp["score"] / lp
                else:
                    score = hyp["score"] / (len(tokens) + 1),
                final_hyps.append({
                    "tokens": tokens,
                    "score": score,
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
    
    def load(self, model_path):
        """Load NMT model.
        """
        self.model.load(model_path)
    
    def batch_translate(self, input_path, output_path, field=0, remove_subword_tokens=True, max_length=100, resume=False):
        """Translate a file."""
        # Check whether using multiple GPUs
        try:
            import horovod.torch as hvd
        except ImportError:
            pass
        # If using multigpu, then separate the input file
        if self._is_multigpu:
            sync_tensor = torch.tensor(0)
            tmp_output_path = "/tmp/{}.{}".format(os.path.basename(output_path), hvd.local_rank())
        else:
            sync_tensor = None
            tmp_output_path = output_path
        result_map = {}
        if self._is_multigpu and resume and os.path.exists(tmp_output_path):
            for line in open(tmp_output_path):
                pair = line.strip("\n").split("\t")
                if len(pair) != 2:
                    print(line)
                id, line = pair
                result_map[int(id)] = line
            print("loaded {} computed results".format(len(result_map)))
        fout = open(tmp_output_path, "w")
        test_lines = list(open(input_path))
        err = 0
        for i, line in enumerate(test_lines):
            # Gather error counts in multigpu mode
            if self._is_multigpu:
                if i % (10 * hvd.size()) == 0:
                    sync_tensor.fill_(err)
                    hvd.allreduce_(sync_tensor, average=False)
                if i % hvd.size() != hvd.local_rank():
                    continue
            # Translate
            pair = line.strip().split("\t")
            src_sent = pair[field]
            if len(src_sent.split()) > max_length:
                result = "x"
            else:
                if i in result_map:
                    result = result_map[i]
                else:
                    result, _ = self.translate("<s> {} </s>".format(src_sent))

            if result is None:
                result = ""
            if remove_subword_tokens:
                if "▁" in result:
                    result = "".join(result.split()).replace("▁", " ").strip()
                else:
                    result = result.replace("@@ ", "")
            if not result:
                err += 1
            # Write the results and print progress
            if self._is_multigpu:
                fout.write("{}\t{}\n".format(i, result))
            else:
                fout.write("{}\n".format(result))
            fout.flush()
            if self._is_multigpu and hvd.local_rank() == 0:
                sys.stdout.write("translating: {:.0f}%  err: {}    \r".format(float(i + 1) * 100 / len(test_lines),
                                                                              int(sync_tensor)))
            elif not self._is_multigpu:
                sys.stdout.write("translating: {:.0f}%  err: {}    \r".format(float(i + 1) * 100 / len(test_lines), err))
            sys.stdout.flush()
        if is_root_node():
            sys.stdout.write("\n")
        fout.close()
        if self._is_multigpu:
            # Wait for all process to end
            hvd.allreduce_(sync_tensor, average=False)
            # Concatenate all separated translation results
            if hvd.local_rank() == 0:
                results = []
                for i in range(hvd.size()):
                    for line in open("/tmp/{}.{}".format(os.path.basename(output_path), i)):
                        id, result = line.strip("\n").split("\t")
                        results.append((int(id), result))
                results.sort()
                with open(output_path, "w") as fout:
                    for _, result in results:
                        fout.write(result + "\n")
