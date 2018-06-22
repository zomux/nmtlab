#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from six.moves import zip

import torchtext
import torchtext.vocab
import pickle
from collections import Counter, defaultdict

DEFAULT_SPECIAL_TOKENS = ["<null>", "<s>", "</s>", "UNK"]

class Vocab(torchtext.vocab.Vocab):
    
    def __init__(self, path):
        self.tokens = pickle.load(open(path, "rb"), encoding='latin1')
        self.vectors = None
        self.freqs = Counter(self.tokens)
        self.itos = self.tokens
        self.stoi = defaultdict(lambda: 3)
        self.stoi.update({tok: i for i, tok in enumerate(self.tokens)})
    
    def size(self):
        return len(self.tokens)
    
    def build(self, txt_path, limit=None, special_tokens=None, char_level=False):
        vocab_counter = Counter()
        for line in open(txt_path).xreadlines():
            line = line.strip()
            if char_level:
                words = [w.encode("utf-8") for w in line.decode("utf-8")]
            else:
                words = line.split(" ")
            vocab_counter.update(words)
        if special_tokens is None:
            special_tokens = DEFAULT_SPECIAL_TOKENS
        if limit is not None:
            final_items = vocab_counter.most_common()[:limit - len(special_tokens)]
        else:
            final_items = vocab_counter.most_common()
        final_items.sort(key=lambda x: (-x[1], x[0]))
        final_words = [x[0] for x in final_items]
        self._vocab = special_tokens + final_words
        self._build_vocab_map()

    def add(self, token):
        self._vocab.append(token)
        self._vocab_map[token] = self._vocab.index(token)

    def save(self, path):
        pickle.dump(self._vocab, open(path, "wb"))

    def load(self, path):
        u = pickle._Unpickler(open(path, "rb"))
        u.encoding = 'latin1'
        self._vocab = u.load()
        self._build_vocab_map()

    def _build_vocab_map(self):
        self._vocab_map = {}
        for i, tok in enumerate(self._vocab):
            self._vocab_map[tok] = i

    def encode(self, tokens):
        return map(self.encode_token, tokens)

    def encode_token(self, token):
        if token in self._vocab_map:
            return self._vocab_map[token]
        else:
            return self._vocab_map[self._unk_token]
    
    def get_index_table(self):
        import tempfile
        assert len(self._vocab) > 0
        path = tempfile.mkstemp()[1]
        open(path, "w").write("\n".join(self._vocab))
        return lookup.index_table_from_file(
            vocabulary_file=path, vocab_size=self.size(),
            default_value=self.encode_token("UNK"))
    
    def decode(self, indexes):
        return map(self.decode_token, indexes)

    def decode_token(self, index):
        return self._vocab[index] if index < len(self._vocab) else self._unk_token

    def contains(self, token):
        return token in self._vocab_map

    def size(self):
        return len(self._vocab)

    def get_list(self):
        return self._vocab
