# -*- coding: utf-8 -*-
r"""
    malgan.discriminator
    ~~~~~~~~~~~~~~~~~

    Discriminator (i.e., substitute detector) block for MalGAN.

    Based on the paper: "Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN"
    By Weiwei Hu and Ying Tan.

    :version: 0.1.0
    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""
from typing import List

import torch
from torch import Tensor
import torch.nn as nn


# noinspection PyPep8Naming
class Discriminator(nn.Module):
    r""" MalGAN discriminator (substitute detector).  Simple feed forward network. """
    EPS = 1e-7

    def __init__(self, M: int, hidden_size: List[int], g: nn.Module):
        r"""Discriminator Constructor

        Builds the discriminator block.

        :param M: Width of the malware feature vector
        :param hidden_size: Width of the hidden layer(s).
        :param g: Activation function
        """
        super().__init__()

        # Build the feed forward layers.
        self._layers = nn.Sequential()
        for i, (in_w, out_w) in enumerate(zip([M] + hidden_size[:-1], hidden_size)):
            layer = nn.Sequential(nn.Linear(in_w, out_w), g)
            self._layers.add_module("FF%02d" % i, layer)

        layer = nn.Sequential(nn.Linear(hidden_size[-1], 1), nn.Sigmoid())
        self._layers.add_module("FF%02d" % len(hidden_size), layer)

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Forward path through the discriminator.

        :param X: Input example tensor
        :return: :math:`D_{sigma}(x)` -- Value predicted by the discriminator.
        """
        d_theta = self._layers(X)
        return torch.clamp(d_theta, self.EPS, 1. - self.EPS).view(-1)
