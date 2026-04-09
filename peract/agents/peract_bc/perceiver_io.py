# PerceiverIO adapted for 6-DoF manipulation WITHOUT language input
# Modified from perceiver_lang_io.py for the YOLH project

from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops.layers.torch import Reduce

from helpers.network_utils import DenseBlock, SpatialSoftmax3D, Conv3DBlock, Conv3DUpsampleBlock


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn


def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class PerceiverVoxelEncoder(nn.Module):
    """PerceiverIO encoder for voxel grids WITHOUT language conditioning."""

    def __init__(
            self,
            depth,
            iterations,
            voxel_size,
            initial_dim,
            low_dim_size,
            layer=0,
            num_rotation_classes=72,
            num_collision_classes=2,
            input_axis=3,
            num_latents=512,
            im_channels=64,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            activation='relu',
            weight_tie_layers=False,
            input_dropout=0.1,
            attn_dropout=0.1,
            decoder_dropout=0.0,
            voxel_patch_size=9,
            voxel_patch_stride=8,
            no_skip_connection=False,
            no_perceiver=False,
            final_dim=64,
    ):
        super().__init__()
        self.depth = depth
        self.layer = layer
        self.init_dim = int(initial_dim)
        self.iterations = iterations
        self.input_axis = input_axis
        self.voxel_size = voxel_size
        self.low_dim_size = low_dim_size
        self.im_channels = im_channels
        self.voxel_patch_size = voxel_patch_size
        self.voxel_patch_stride = voxel_patch_stride
        self.num_rotation_classes = num_rotation_classes
        self.num_collision_classes = num_collision_classes
        self.final_dim = final_dim
        self.input_dropout = input_dropout
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.no_skip_connection = no_skip_connection
        self.no_perceiver = no_perceiver

        # patchified spatial dimensions
        spatial_size = voxel_size // self.voxel_patch_stride  # 100/5 = 20

        # No language: input dim = voxel_features (+ proprio_features if available)
        if self.low_dim_size > 0:
            self.input_dim_before_seq = self.im_channels * 2
        else:
            self.input_dim_before_seq = self.im_channels

        # positional encoding on the flattened voxel grid only (no language tokens)
        self.pos_encoding = nn.Parameter(torch.randn(1,
                                                     spatial_size ** 3,
                                                     self.input_dim_before_seq))

        # voxel input preprocessing
        self.input_preprocess = Conv3DBlock(
            self.init_dim, self.im_channels, kernel_sizes=1, strides=1,
            norm=None, activation=activation,
        )

        # patchify conv
        self.patchify = Conv3DBlock(
            self.input_preprocess.out_channels, self.im_channels,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation)

        # proprioception
        if self.low_dim_size > 0:
            self.proprio_preprocess = DenseBlock(
                self.low_dim_size, self.im_channels, norm=None, activation=activation,
            )

        # pooling functions
        self.local_maxp = nn.MaxPool3d(3, 2, padding=1)
        self.global_maxp = nn.AdaptiveMaxPool3d(1)

        # 1st 3D softmax
        self.ss0 = SpatialSoftmax3D(
            self.voxel_size, self.voxel_size, self.voxel_size, self.im_channels)
        flat_size = self.im_channels * 4

        # latent vectors
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # encoder cross attention
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim,
                                          self.input_dim_before_seq,
                                          heads=cross_heads,
                                          dim_head=cross_dim_head,
                                          dropout=input_dropout),
                    context_dim=self.input_dim_before_seq),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads,
                                                    dim_head=latent_dim_head, dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # self attention layers
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # decoder cross attention
        self.decoder_cross_attn = PreNorm(self.input_dim_before_seq, Attention(self.input_dim_before_seq,
                                                                               latent_dim,
                                                                               heads=cross_heads,
                                                                               dim_head=cross_dim_head,
                                                                               dropout=decoder_dropout),
                                          context_dim=latent_dim)

        # upsample conv
        self.up0 = Conv3DUpsampleBlock(
            self.input_dim_before_seq, self.final_dim,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation,
        )

        # 2nd 3D softmax
        self.ss1 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self.input_dim_before_seq)

        flat_size += self.input_dim_before_seq * 4

        # final conv
        self.final = Conv3DBlock(
            self.im_channels if (self.no_perceiver or self.no_skip_connection) else self.im_channels * 2,
            self.im_channels,
            kernel_sizes=3,
            strides=1, norm=None, activation=activation)

        self.trans_decoder = Conv3DBlock(
            self.final_dim, 1, kernel_sizes=3, strides=1,
            norm=None, activation=None,
        )

        # rotation, gripper, and collision MLPs
        if self.num_rotation_classes > 0:
            self.ss_final = SpatialSoftmax3D(
                self.voxel_size, self.voxel_size, self.voxel_size,
                self.im_channels)
            flat_size += self.im_channels * 4

            self.dense0 = DenseBlock(flat_size, 256, None, activation)
            self.dense1 = DenseBlock(256, self.final_dim, None, activation)
            self.rot_grip_collision_ff = DenseBlock(self.final_dim,
                                                    self.num_rotation_classes * 3 + \
                                                    1 + \
                                                    self.num_collision_classes,
                                                    None, None)

    def forward(
            self,
            ins,
            proprio,
            prev_layer_voxel_grid,
            bounds,
            prev_layer_bounds,
            mask=None,
    ):
        # Note: no lang_goal_emb or lang_token_embs arguments

        # preprocess input
        d0 = self.input_preprocess(ins)

        # aggregated features
        feats = [self.ss0(d0.contiguous()), self.global_maxp(d0).view(ins.shape[0], -1)]

        # patchify
        ins = self.patchify(d0)

        b, c, d, h, w, device = *ins.shape, ins.device
        axis = [d, h, w]
        assert len(axis) == self.input_axis

        # concat proprio
        if self.low_dim_size > 0:
            p = self.proprio_preprocess(proprio)
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
            ins = torch.cat([ins, p], dim=1)

        # channel last
        ins = rearrange(ins, 'b d ... -> b ... d')

        # flatten and add positional encoding
        queries_orig_shape = ins.shape
        ins = rearrange(ins, 'b ... d -> b (...) d')
        ins = ins + self.pos_encoding

        # batchify latents
        x = repeat(self.latents, 'n d -> b n d', b=b)

        cross_attn, cross_ff = self.cross_attend_blocks

        for it in range(self.iterations):
            x = cross_attn(x, context=ins, mask=mask) + x
            x = cross_ff(x) + x
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # decoder cross attention
        latents = self.decoder_cross_attn(ins, context=x)

        # reshape to voxel grid
        latents = latents.view(b, *queries_orig_shape[1:-1], latents.shape[-1])
        latents = rearrange(latents, 'b ... d -> b d ...')

        feats.extend([self.ss1(latents.contiguous()), self.global_maxp(latents).view(b, -1)])

        # upsample
        u0 = self.up0(latents)

        if self.no_skip_connection:
            u = self.final(u0)
        elif self.no_perceiver:
            u = self.final(d0)
        else:
            u = self.final(torch.cat([d0, u0], dim=1))

        # translation decoder
        trans = self.trans_decoder(u)

        # rotation, gripper, collision MLPs
        rot_and_grip_out = None
        collision_out = None
        if self.num_rotation_classes > 0:
            feats.extend([self.ss_final(u.contiguous()), self.global_maxp(u).view(b, -1)])
            dense0 = self.dense0(torch.cat(feats, dim=1))
            dense1 = self.dense1(dense0)
            rot_and_grip_collision_out = self.rot_grip_collision_ff(dense1)
            rot_and_grip_out = rot_and_grip_collision_out[:, :-self.num_collision_classes]
            collision_out = rot_and_grip_collision_out[:, -self.num_collision_classes:]

        return trans, rot_and_grip_out, collision_out
