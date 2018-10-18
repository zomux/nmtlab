#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import math

def gelu(x):
    """
    ﻿Hendrycks, D., & Gimpel, K. (2016) Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units.
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
