# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pytorch implementation of Vision Transformer (ViT)

Based on the timm library by Ross Wightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import math

import torch
import torch.nn as nn
import warnings


class PosEmbedder(nn.Module):

    max_position_index = 200
    repeat_section_embedding = True

    def __init__(
        self, 
        embed_dim: int, 
        dropout_prob: float,
    ) -> None:
        """
        Initialize class for generating predefined positional embeddings.

        Args:
            embed_dim:  Dimensionality of embeddings.
            dropout_prob:  Probablity of dropout.
            learnable_weight:  Indicates whether the positional embedding and
                section embedding are first multiplied with a learnable weight
                before being added to the feature vectors.
        """
        super().__init__()
        # initialize instance attributes
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(
            data=torch.zeros((self.max_position_index*2+1, self.embed_dim//2)),
            requires_grad=False,
        )
        X = torch.arange(-self.max_position_index, self.max_position_index+1, dtype=torch.float32).reshape(-1, 1)
        X = X / torch.pow(10000, torch.arange(0, self.embed_dim//2, 2, dtype=torch.float32) / (self.embed_dim//2))

        self.pos_embed[:, 0::2] = torch.sin(X)
        self.pos_embed[:, 1::2] = torch.cos(X)

        # initialize dropout layer
        self.pos_drop = nn.Dropout(p=dropout_prob)

    def forward(
        self, 
        x: torch.Tensor, 
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add positional embedding to sequence of embeddings.

        Args:
            x:  Batch of embeddings as (batch, sequence, channels).
            shape:  Actual spatial shape of the input as (height, width).
        
        Returns:
            x:  Batch of embeddings as (batch, sequence, channels).

        Note: Interpolation of positional embeddings not implemented.
        """
        pos = torch.round(pos).to(int)
        if torch.max(pos[:, :, 1:]) > self.max_position_index:
            raise ValueError(
                'Maximum requested position index exceeds the prepared position indices.'
            )
        # get the number of items in the batch and the number of tokens in the sequence
        B, S, _ = pos.shape
        device = self.pos_embed.get_device()
        if device == -1:
            device = 'cpu'

        # define embeddings for x and y dimension
        embeddings = [self.pos_embed[pos[:, :, 0]+self.max_position_index, :],
                      self.pos_embed[pos[:, :, 1]+self.max_position_index, :]]
        # add a row of zeros as padding in case the embedding dimension has an odd length
        if self.embed_dim % 2 == 1:
            embeddings.append(torch.zeros((B, S, 1), device=device))

        # prepare positional embedding
        pos_embedding = torch.concat(embeddings, dim=-1)
        
        # check if the shape of the features and positional embeddings match
        if x.shape != pos_embedding.shape:
            raise ValueError(
                'Shape of features and positional embedding tensors do not match.',
            )
        # add the combined embedding to each element in the sequence
        x = self.pos_drop(x+pos_embedding)
        
        return x
    

class PytorchMultiheadAttention(nn.MultiheadAttention):
    """ 
    Wrapper around nn.MultiheadAttention for making the key, value, and query 
    all equal to the input x of the forward method and set average_attn_weights=False.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, 
                 add_zero_attn=False, kdim=None, vdim=None, batch_first=False, 
                 device=None, dtype=None) -> None:
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, 
                         add_zero_attn, kdim, vdim, batch_first, device, dtype)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(x, x, x, need_weights=True, average_attn_weights=False)


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Initialize multi-head self-attention layer.

        Args:
            embed_dim:  Dimensionality of embedding used throughout the model.
            num_heads:  Number of heads in each multi-head attention sub-layer.
            qkv_bias:  Indicates whether biases are added to the queries, keys, 
                and values.
            attn_drop:  Probablity of dropout for multi-head attention.
            proj_drop:  Probablity of dropout for fully-connected layer.
        """
        super().__init__()
        # initialize instance attributes
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads)**-0.5
        
        # initialize layers
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        # get the queries, keys, and values
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # compute attention matrix
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # get attention weighted value embeddings and project them
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn
        

class MultiHeadAttentionFactory():

    def __init__(
        self, 
        embed_dim: int, 
        n_heads: int, 
        attn_dropout_prob: float,
        pytorch_imp: bool,
    ) -> None:
        """
        Returns multi-head attention implementation of the specified type with the
        correct configuration of input arguments.

        Notes:
        (1) The naive implementation has an option to enable or disable the bias 
            of the first linear layer, which by default was disabled. The Pytorch
            implementation only has an option to enable or disable the bias of
            both linear layers, which by default is enabled.
        
        Args:
            embed_dim:  Dimensionality of embeddings.
            n_heads:  Number of heads in multi-head attention.
            attn_dropout_prob:  Probability of dropping values from attention map.
            pytorch_imp:  Indicates if the Pytorch multi-head self-attention is used.
        """
        # select the correct implementation based on the requirements
        if pytorch_imp:                    
            self.multi_head_attention = PytorchMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=n_heads,
                add_bias_kv=False,
                dropout=attn_dropout_prob,
                batch_first=True,
            )
        else:
            self.multi_head_attention = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=n_heads,
                qkv_bias=True,
                attn_drop=attn_dropout_prob,
                proj_drop=0,
            )
    
    def get_multi_head_attention(self):
        return self.multi_head_attention


def _no_grad_trunc_normal_(tensor: torch.Tensor, mean: float, std: float, 
                           a: float, b: float) -> torch.Tensor:
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, 
                  a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)