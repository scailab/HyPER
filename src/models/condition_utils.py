# This code was based on and modified from https://github.com/pdearena/pdearena/blob/main/pdearena/modules/conditioned/condition_utils.py

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math
from abc import abstractmethod

import torch
from torch import nn

def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module

# def fourier_embedding(timesteps: torch.Tensor, dim, max_period=10000):
#     r"""Create sinusoidal timestep embeddings.

#     Args:
#         timesteps: a 1-D Tensor of N indices, one per batch element.
#                       These may be fractional.
#         dim (int): the dimension of the output.
#         max_period (int): controls the minimum frequency of the embeddings.
#     Returns:
#         embedding (torch.Tensor): [N $\times$ dim] Tensor of positional embeddings.
#     """
#     half = dim // 2
#     freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
#         device=timesteps.device
#     )
#     args = timesteps[:, None].float() * freqs[None]
#     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#     if dim % 2:
#         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#     return embedding

# fixed embedding function
def fourier_embedding(timesteps: torch.Tensor, dim, max_period=10000.0):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
    pe = torch.zeros(timesteps.shape[0], dim).to(timesteps.device)
    position = torch.arange(0, timesteps.shape[0]).unsqueeze(1).to(timesteps.device)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                         -(math.log(max_period) / dim))).to(timesteps.device)
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class ConditionedBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` embdding of time or others."""

# class EmbedSequential(nn.Sequential, ConditionedBlock):
#     def forward(self, x, emb):
#         for layer in self:
#             if isinstance(layer, ConditionedBlock):
#                 x = layer(x, emb)
#             else:
#                 x = layer(x)

#         return x

class EmbedSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb)
        return x
