# This file is modified from https://github.com/JinluZhang1126/MixSTE/blob/main/common/model_cross.py
# which is originally from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Used under the Apache-2.0 license: https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE
#
# All modifications by CSIRO:
# Copyright (c) 2024-present, CSIRO
# All rights reserved.
# Licensed under the license found in the LICENSE file in the root directory of this source tree.

import math
from functools import partial
from einops import rearrange

import torch
import torch.nn as nn

from timm.models.layers import DropPath


def exists(x):
    return x is not None


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Two-layer MLP
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # Modified Attention based on GRAND: https://openreview.net/forum?id=_1fu_cjsaRE
        I = torch.eye(N, device=attn.device, dtype=attn.dtype).view(1, 1, N, N).repeat(B, self.num_heads, 1, 1)
        x = ((attn - I) @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Transformer Block
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, time_emb_dim = None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # time_emb
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, is_spatial, time_emb = None):
        b, f, p, c = x.shape
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b 1 1 c')
            x = x + time_emb

        if is_spatial:
            x = rearrange(x, 'b f p c -> (b f) p c')
        else:
            x = rearrange(x, 'b f p c -> (b p) f c')

        if self.training and self.drop_path_rate > 0.:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))

        if is_spatial:
            x = rearrange(x, '(b f) p c -> b f p c', b=b, f=f)
        else:
            x = rearrange(x, '(b p) f c -> b f p c', b=b, p=p)

        return x


# Seq2frame MixSTE
class ConditionalDiffusionMixSTES2FGRANDLinLift(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, with_time_emb=True, **kwargs):
        """
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            with_time_emb (bool): whether to use time embeddings
        """
        super().__init__()

        # time embeddings
        if with_time_emb:
            time_dim = embed_dim * 2

            sinu_pos_emb = SinusoidalPosEmb(embed_dim)
            fourier_dim = embed_dim

            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        in_3D_pose_chans = 3
        out_dim = 3

        # Linear layer to fuse noisy 3D pose and 2D pose
        self.fusion_layer = nn.Linear(in_3D_pose_chans + in_chans, embed_dim)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)


        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        ### Spatial pos embedding
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))

        # Spatial Transformer Blocks
        self.STEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, time_emb_dim = time_dim)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim)

        ### Temporal pos embedding
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))

        # Temporal Transformer Blocks
        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, time_emb_dim = time_dim)
            for i in range(depth)])

        self.Temporal_norm = norm_layer(embed_dim)


        # ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        # Regression Head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )

    def ST_foward(self, x, t=None):
        assert len(x.shape) == 4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(self.block_depth):
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]

            if i == 0:
                x = rearrange(x, 'b f n cw -> (b f) n cw')
                x += self.Spatial_pos_embed
                x = self.pos_drop(x)
                x = rearrange(x, '(b f) p cw  -> b f p cw', b=b, f=f)

            x = steblock(x, is_spatial=True, time_emb=t)
            x = self.Spatial_norm(x)

            if i == 0:
                x = rearrange(x, 'b f n cw -> (b n) f cw')
                x += self.Temporal_pos_embed
                x = self.pos_drop(x)
                x = rearrange(x, '(b n) f cw -> b f n cw', b=b, n=n)

            x = tteblock(x, is_spatial=False, time_emb=t)
            x = self.Temporal_norm(x)

        return x

    def forward_denoise(self, x, time):
        b, f, n, c = x.shape
        x = self.fusion_layer(x)
        # x size [batch, frame, part, embed_dim]

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        x = self.ST_foward(x, t)

        x = x.view(b, f, -1)
        x = self.weighted_mean(x)
        x = x.view(b, 1, n, -1)
        x = self.head(x)
        # x size [batch, 1, part, embed_dim]
        return x