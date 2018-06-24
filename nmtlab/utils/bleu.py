#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
BLEU functions are copied from groundhog.
"""

from collections import Counter
import numpy
from six.moves import xrange


def bleu_stats(hypothesis, reference):
    yield len(hypothesis)
    yield len(reference)
    for n in xrange(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i + n]) for i in xrange(len(hypothesis) + 1 - n)])
        r_ngrams = Counter([tuple(reference[i:i + n]) for i in xrange(len(reference) + 1 - n)])
        yield sum((s_ngrams & r_ngrams).values())
        yield max(len(hypothesis) + 1 - n, 0)


def bleu(hypothesis, reference):
    stats = list(bleu_stats(hypothesis, reference))
    stats = numpy.atleast_2d(numpy.asarray(stats))[:, :10].sum(axis=0)
    if not all(stats):
        return 0
    c, r = stats[:2]
    log_bleu_prec = sum([numpy.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]) / 4.
    return numpy.exp(min(0, 1 - float(r) / c) + log_bleu_prec) * 100


def smoothed_bleu(hypothesis, reference):
    stats = list(bleu_stats(hypothesis, reference))
    c, r = stats[:2]
    if c == 0:
        return 0
    log_bleu_prec = sum([numpy.log((1 + float(x)) / (1 + y)) for x, y in zip(stats[2::2], stats[3::2])]) / 4.
    return numpy.exp(min(0, 1 - float(r) / c) + log_bleu_prec) * 100
