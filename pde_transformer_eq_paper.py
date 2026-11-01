"""Paper-oriented D4-equivariant PDE-Transformer components.

This module follows the core constructions from arXiv:2602.08047:
- Eq.(6): D4 lifting patch embedding
- Eq.(7): group-mixing equivariant linear maps for token projections
- Eq.(12)-(14): canonical pair-orbit relative positional encoding
- Eq.(16)-(17): canonicalize -> (down/up)-sample -> de-canonicalize

It is implemented as an in-place conversion of the official mixed-channel
PDE-Transformer graph used in this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput

import pde_transformer_eq_local as local
from pdetransformer.core.mixed_channels.pde_transformer import (
    Downsample,
    Mlp,
    PDEImpl,
    PDE_models,
    Upsample,
    WindowAttention2DTime,
)


D4_ACTIONS = local.D4_ACTIONS
D4_MUL = local.D4_MUL
D4_INV = local.D4_INV


def _d4_transform_feature(x: torch.Tensor, k: int, reflect: bool) -> torch.Tensor:
    return local._d4_transform_feature(x, k, reflect)


def _d4_inverse_transform_feature(x: torch.Tensor, k: int, reflect: bool) -> torch.Tensor:
    return local._d4_inverse_transform_feature(x, k, reflect)


def _d4_transform_kernel(weight: torch.Tensor, k: int, reflect: bool) -> torch.Tensor:
    return local._d4_transform_kernel(weight, k, reflect)


def _d4_inverse_transform_kernel(weight: torch.Tensor, k: int, reflect: bool) -> torch.Tensor:
    return local._d4_inverse_transform_kernel(weight, k, reflect)


class EQLinearGroupConv(nn.Module):
    """Group-mixing equivariant linear map from Eq.(7).

    If x has fibers x_g and weights W_h for h in G, this layer computes:
      y_B = sum_g x_g W_{B^{-1} g}
    with D4 group product over fiber indices.
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 8, bias: bool = True):
        super().__init__()
        if in_features % group_size != 0 or out_features % group_size != 0:
            raise ValueError(
                f"EQLinearGroupConv requires divisibility by group_size={group_size}, "
                f"got in={in_features}, out={out_features}."
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.in_base = self.in_features // self.group_size
        self.out_base = self.out_features // self.group_size

        # weights indexed by group element h: [G, out_base, in_base]
        self.weight = nn.Parameter(torch.empty(self.group_size, self.out_base, self.in_base))
        self.bias = nn.Parameter(torch.zeros(self.out_base)) if bias else None
        nn.init.xavier_uniform_(self.weight)

        # src_idx[B, g] = B^{-1} g
        src_idx = torch.empty((self.group_size, self.group_size), dtype=torch.long)
        for B in range(self.group_size):
            for g in range(self.group_size):
                src_idx[B, g] = D4_MUL[D4_INV[B], g]
        self.register_buffer("src_idx", src_idx, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.group_size, self.in_base)  # [Bf, G, Ib]
        w = self.weight[self.src_idx]  # [B(out fiber), g(in fiber), Ob, Ib]
        y = torch.einsum("bgi,hgoi->bho", x, w)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 8) -> "EQLinearGroupConv":
        eq = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            bias=linear.bias is not None,
        )
        g = group_size
        w_blocks = linear.weight.detach().view(g, eq.out_base, g, eq.in_base)  # [B, Ob, gin, Ib]

        # Recover W_h from blocks W[B, gin] where h = B^{-1} gin.
        rec = []
        for h in range(g):
            vals = []
            for B in range(g):
                gin = int(D4_MUL[B, h].item())  # h = B^{-1} gin => gin = B h
                vals.append(w_blocks[B, :, gin, :])
            rec.append(torch.stack(vals, dim=0).mean(dim=0))
        eq.weight.data.copy_(torch.stack(rec, dim=0))

        if linear.bias is not None:
            b = linear.bias.detach().view(g, eq.out_base)
            eq.bias.data.copy_(b.mean(dim=0))
        return eq


class EQMlpPaper(nn.Module):
    """MLP with Eq.(7)-style group-mixing linear layers."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        group_size: int = 8,
    ):
        super().__init__()
        self.fc1 = EQLinearGroupConv(in_features, hidden_features, group_size=group_size)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = EQLinearGroupConv(hidden_features, out_features, group_size=group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    @classmethod
    def from_mlp(cls, mlp: Mlp, group_size: int = 8) -> "EQMlpPaper":
        in_features = mlp.fc1.in_features
        hidden_features = mlp.fc1.out_features
        out_features = mlp.fc2.out_features
        drop = float(getattr(mlp.drop, "p", 0.0))
        eq = cls(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=type(mlp.act),
            drop=drop,
            group_size=group_size,
        )
        eq.fc1 = EQLinearGroupConv.from_linear(mlp.fc1, group_size=group_size)
        eq.fc2 = EQLinearGroupConv.from_linear(mlp.fc2, group_size=group_size)
        return eq


class EQRelativeBiasD4PairCanonical(nn.Module):
    """Pair-orbit canonical D4 relative bias from Eq.(12)-(14)."""

    def __init__(
        self,
        window_size: list[int],
        pretrained_window_size: list[int],
        num_heads: int,
        no_log: bool = False,
    ):
        super().__init__()
        self.window_size = [int(ws) for ws in window_size]
        self.num_heads = int(num_heads)

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_heads, bias=False),
        )

        h, w = self.window_size
        if h != w:
            raise ValueError(f"Expected square windows for D4 actions, got window_size={self.window_size}.")

        n_tokens = h * w

        # Build point transforms for each D4 action: old linear index -> (new_i, new_j)
        idx = torch.arange(n_tokens, dtype=torch.long).view(h, w)
        transformed = []
        for k, b in D4_ACTIONS:
            t = idx
            if k:
                t = torch.rot90(t, k, dims=(0, 1))
            if b:
                t = torch.flip(t, dims=(1,))
            transformed.append(t.contiguous())
        transformed = torch.stack(transformed, dim=0)  # [G,H,W], value=old_idx at new_pos

        # Invert mapping to get new_pos for each old_idx.
        flat = transformed.view(len(D4_ACTIONS), -1)
        inv_flat = torch.empty_like(flat)
        ar = torch.arange(n_tokens, dtype=torch.long)
        for g in range(len(D4_ACTIONS)):
            inv_flat[g, flat[g]] = ar
        new_i = (inv_flat // w).view(len(D4_ACTIONS), n_tokens)
        new_j = (inv_flat % w).view(len(D4_ACTIONS), n_tokens)

        # Pair-orbit canonicalization exactly as Eq.(12)-(14):
        # canonical pair first, then delta of canonical pair.
        canonical_pair_delta = torch.empty((h * w, h * w, 2), dtype=torch.float32)
        for ia in range(n_tokens):
            i1 = int(ia // w)
            j1 = int(ia % w)
            for ib in range(n_tokens):
                i2 = int(ib // w)
                j2 = int(ib % w)
                candidates = []
                for g in range(len(D4_ACTIONS)):
                    a_i = int(new_i[g, ia].item())
                    a_j = int(new_j[g, ia].item())
                    b_i = int(new_i[g, ib].item())
                    b_j = int(new_j[g, ib].item())
                    candidates.append((a_i, a_j, b_i, b_j))
                ci1, cj1, ci2, cj2 = min(candidates)
                canonical_pair_delta[ia, ib, 0] = float(ci2 - ci1)
                canonical_pair_delta[ia, ib, 1] = float(cj2 - cj1)

        canonical_table = canonical_pair_delta.unsqueeze(0)  # [1,N,N,2]

        if pretrained_window_size[0] > 0:
            canonical_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            canonical_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            canonical_table[:, :, :, 0] /= (self.window_size[0] - 1)
            canonical_table[:, :, :, 1] /= (self.window_size[1] - 1)

        if not no_log:
            canonical_table *= 8
            canonical_table = (
                torch.sign(canonical_table)
                * torch.log2(torch.abs(canonical_table) + 1.0)
                / torch.log2(torch.tensor(8.0))
            )

        self.register_buffer("canonical_pair_delta", canonical_table, persistent=False)

    def forward(self, input_tensor: torch.Tensor, local_window_size: int) -> torch.Tensor:
        # Eq.(12)-(14): pair orbit canonicalization then base RPE on canonical delta.
        relative_position_bias = self.cpb_mlp(self.canonical_pair_delta).squeeze(0)  # [N,N,H]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [H,N,N]
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        n_global_feature = input_tensor.shape[2] - local_window_size
        relative_position_bias = F.pad(relative_position_bias, (n_global_feature, 0, n_global_feature, 0)).contiguous()
        return input_tensor + relative_position_bias.unsqueeze(0)


class EQWindowAttention2DTimePaper(nn.Module):
    """Attention using Eq.(7) group-mixing QKV and Eq.(8) attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        resolution: int = 0,
        attn_type: str = "v2",
        group_size: int = 8,
        use_relative_bias: bool = False,
    ):
        super().__init__()
        if dim % group_size != 0:
            raise ValueError(
                f"EQWindowAttention2DTimePaper requires dim divisible by group_size={group_size}, got dim={dim}."
            )
        self.num_heads = int(num_heads)
        self.group_size = int(group_size)
        self.dim = int(dim)
        self.fiber_dim = self.dim // self.group_size
        self.num_heads_fiber = self._choose_compatible_heads(self.fiber_dim, self.num_heads)
        self.head_dim = self.fiber_dim // self.num_heads_fiber
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = EQLinearGroupConv(dim, 3 * dim, group_size=self.group_size, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.pos_emb_funct = None
        if use_relative_bias:
            self.pos_emb_funct = EQRelativeBiasD4PairCanonical(
                window_size=[resolution, resolution],
                pretrained_window_size=[resolution, resolution],
                num_heads=self.num_heads_fiber,
            )

        self.attn_type = attn_type
        if attn_type == "v2":
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((self.num_heads_fiber, 1, 1))),
                requires_grad=True,
            )

        self.resolution = int(resolution)

    @staticmethod
    def _choose_compatible_heads(channels: int, requested_heads: int) -> int:
        requested_heads = max(1, int(requested_heads))
        channels = int(channels)
        for h in range(requested_heads, 0, -1):
            if channels % h == 0:
                return h
        return 1

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, n_tokens, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected channels={self.dim}, got {channels}.")
        c_base = channels // self.group_size
        qkv = self.qkv(x).view(bsz, n_tokens, self.group_size, 3, c_base).permute(3, 0, 2, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, G, N, Cb]
        q = q.view(bsz, self.group_size, n_tokens, self.num_heads_fiber, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.view(bsz, self.group_size, n_tokens, self.num_heads_fiber, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.view(bsz, self.group_size, n_tokens, self.num_heads_fiber, self.head_dim).permute(0, 1, 3, 2, 4)

        if self.attn_type == "v2":
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
            attn = attn * logit_scale.view(1, 1, self.num_heads_fiber, 1, 1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.pos_emb_funct is not None:
            attn_view = attn.view(bsz * self.group_size, self.num_heads_fiber, n_tokens, n_tokens)
            attn_view = self.pos_emb_funct(attn_view, self.resolution**2)
            attn = attn_view.view(bsz, self.group_size, self.num_heads_fiber, n_tokens, n_tokens)

        if attn_mask is not None:
            nw = attn_mask.shape[0]
            attn = attn.view(
                bsz // nw,
                nw,
                self.group_size,
                self.num_heads_fiber,
                n_tokens,
                n_tokens,
            )
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            attn = attn.view(-1, self.group_size, self.num_heads_fiber, n_tokens, n_tokens)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.permute(0, 1, 3, 2, 4).reshape(bsz, self.group_size, n_tokens, c_base)
        return out.permute(0, 2, 1, 3).reshape(bsz, n_tokens, channels)

    @classmethod
    def from_attention(cls, attn: WindowAttention2DTime, group_size: int = 8) -> "EQWindowAttention2DTimePaper":
        def _resize_head_rows(src: torch.Tensor, out_heads: int) -> torch.Tensor:
            src_heads = int(src.shape[0])
            if src_heads == out_heads:
                return src
            if src_heads % out_heads == 0:
                return src.view(out_heads, src_heads // out_heads, *src.shape[1:]).mean(dim=1)
            idx = torch.linspace(0, src_heads - 1, steps=out_heads, device=src.device).round().long()
            return src.index_select(0, idx)

        use_relative_bias = bool(getattr(attn, "pos_emb_funct", None) is not None)
        eq = cls(
            dim=attn.qkv.in_features,
            num_heads=attn.num_heads,
            qkv_bias=attn.qkv.bias is not None,
            qk_scale=attn.scale,
            attn_drop=float(getattr(attn.attn_drop, "p", 0.0)),
            resolution=attn.resolution,
            attn_type=attn.attn_type,
            group_size=group_size,
            use_relative_bias=use_relative_bias,
        )
        eq.qkv = EQLinearGroupConv.from_linear(attn.qkv, group_size=group_size)

        if hasattr(attn, "logit_scale") and hasattr(eq, "logit_scale"):
            src_scale = _resize_head_rows(attn.logit_scale.data.detach(), eq.logit_scale.shape[0])
            eq.logit_scale.data.copy_(src_scale.view_as(eq.logit_scale))
        return eq


class EQDownsample2dPaper(nn.Module):
    """Eq.(16): canonicalize -> downsample -> de-canonicalize per group fiber."""

    def __init__(self, in_channels: int, out_channels: int, group_size: int = 8):
        super().__init__()
        self.group_size = int(group_size)
        if in_channels % self.group_size != 0 or out_channels % self.group_size != 0:
            raise ValueError("EQDownsample2dPaper requires channels divisible by group_size")

        self.in_base = int(in_channels // self.group_size)
        self.out_base = int(out_channels // self.group_size)
        if self.out_base % 4 != 0:
            raise ValueError("EQDownsample2dPaper requires out_base divisible by 4 (PixelUnshuffle)")

        self.down = nn.Sequential(
            nn.Conv2d(self.in_base, self.out_base // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x.shape
        xx = x.view(bsz, self.group_size, self.in_base, h, w)
        ys = []
        for g_idx, (k, b) in enumerate(D4_ACTIONS):
            xg = xx[:, g_idx]
            xg_can = _d4_inverse_transform_feature(xg, k, b)
            yg_can = self.down(xg_can)
            yg = _d4_transform_feature(yg_can, k, b)
            ys.append(yg)
        return torch.stack(ys, dim=1).reshape(bsz, self.group_size * self.out_base, h // 2, w // 2)

    @classmethod
    def from_downsample(cls, down: Downsample, group_size: int = 8) -> "EQDownsample2dPaper":
        conv = down.body[0]
        out_channels = int(conv.out_channels * 4)
        eq = cls(conv.in_channels, out_channels, group_size=group_size)

        g = group_size
        pre_out = conv.out_channels // g
        in_base = conv.in_channels // g
        w = conv.weight.detach().view(g, pre_out, g, in_base, *conv.kernel_size)
        idx = torch.arange(g, device=w.device)
        diag = w[idx, :, idx, ...]
        canonical = []
        for gi, (k, b) in enumerate(D4_ACTIONS):
            canonical.append(_d4_inverse_transform_kernel(diag[gi], k, b))
        eq.down[0].weight.data.copy_(torch.stack(canonical, dim=0).mean(dim=0))
        return eq


class EQUpsample2dPaper(nn.Module):
    """Eq.(17): canonicalize -> pixel-shuffle -> de-canonicalize per group fiber."""

    def __init__(self, in_channels: int, out_channels: int, group_size: int = 8):
        super().__init__()
        self.group_size = int(group_size)
        if in_channels % self.group_size != 0 or out_channels % self.group_size != 0:
            raise ValueError("EQUpsample2dPaper requires channels divisible by group_size")

        self.in_base = int(in_channels // self.group_size)
        self.out_base = int(out_channels // self.group_size)
        self.up = nn.Sequential(
            nn.Conv2d(self.in_base, self.out_base * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x.shape
        xx = x.view(bsz, self.group_size, self.in_base, h, w)
        ys = []
        for g_idx, (k, b) in enumerate(D4_ACTIONS):
            xg = xx[:, g_idx]
            xg_can = _d4_inverse_transform_feature(xg, k, b)
            yg_can = self.up(xg_can)
            yg = _d4_transform_feature(yg_can, k, b)
            ys.append(yg)
        return torch.stack(ys, dim=1).reshape(bsz, self.group_size * self.out_base, h * 2, w * 2)

    @classmethod
    def from_upsample(cls, up: Upsample, group_size: int = 8) -> "EQUpsample2dPaper":
        conv = up.body[0]
        out_channels = int(conv.out_channels // 4)
        eq = cls(conv.in_channels, out_channels, group_size=group_size)

        g = group_size
        pre_out = conv.out_channels // g
        in_base = conv.in_channels // g
        w = conv.weight.detach().view(g, pre_out, g, in_base, *conv.kernel_size)
        idx = torch.arange(g, device=w.device)
        diag = w[idx, :, idx, ...]
        canonical = []
        for gi, (k, b) in enumerate(D4_ACTIONS):
            canonical.append(_d4_inverse_transform_kernel(diag[gi], k, b))
        eq.up[0].weight.data.copy_(torch.stack(canonical, dim=0).mean(dim=0))
        return eq


def convert_pde_block_to_eq_paper(
    block: nn.Module,
    group_size: int = 8,
    relbias_symmetrize: bool = True,
) -> nn.Module:
    if hasattr(block, "norm1") and isinstance(block.norm1, nn.LayerNorm):
        dim = int(block.norm1.normalized_shape[0])
        block.norm1 = local.EQLayerNorm(dim, group_size=group_size)
    if hasattr(block, "norm2") and isinstance(block.norm2, nn.LayerNorm):
        dim = int(block.norm2.normalized_shape[0])
        block.norm2 = local.EQLayerNorm(dim, group_size=group_size)

    if hasattr(block, "hat_norm1") and isinstance(block.hat_norm1, nn.LayerNorm):
        dim = int(block.hat_norm1.normalized_shape[0])
        block.hat_norm1 = local.EQLayerNorm(dim, group_size=group_size)
    if hasattr(block, "hat_norm2") and isinstance(block.hat_norm2, nn.LayerNorm):
        dim = int(block.hat_norm2.normalized_shape[0])
        block.hat_norm2 = local.EQLayerNorm(dim, group_size=group_size)

    if hasattr(block, "mlp") and isinstance(block.mlp, Mlp):
        block.mlp = EQMlpPaper.from_mlp(block.mlp, group_size=group_size)
    if hasattr(block, "hat_mlp") and isinstance(block.hat_mlp, Mlp):
        block.hat_mlp = EQMlpPaper.from_mlp(block.hat_mlp, group_size=group_size)

    if hasattr(block, "attn") and isinstance(block.attn, WindowAttention2DTime):
        block.attn = EQWindowAttention2DTimePaper.from_attention(block.attn, group_size=group_size)
    if hasattr(block, "attn") and hasattr(block.attn, "pos_emb_funct"):
        pos_emb = block.attn.pos_emb_funct
        if hasattr(pos_emb, "d4_symmetrize"):
            pos_emb.d4_symmetrize = bool(relbias_symmetrize)

    # AdaLN modulation stays output-shared over group fibers.
    for name in ("adain_1", "adain_2"):
        if hasattr(block, name):
            ad = getattr(block, name)
            if hasattr(ad, "linear") and isinstance(ad.linear, nn.Linear):
                linear = ad.linear
                if linear.out_features % 6 == 0 and (linear.out_features // 6) % group_size == 0:
                    ad.linear = local.EQLinearOutSharedChunked.from_linear(
                        linear, num_chunks=6, group_size=group_size
                    )
                elif linear.out_features % group_size == 0:
                    ad.linear = local.EQLinearOutShared.from_linear(linear, group_size=group_size)

    return block


def _convert_resize_blocks_to_eq_paper(model: PDEImpl, group_size: int = 8) -> None:
    for i in range(model.num_encoder_layers):
        down = model.__getattr__(f"down{i}_{i+1}")
        if isinstance(down, Downsample):
            model.__setattr__(f"down{i}_{i+1}", EQDownsample2dPaper.from_downsample(down, group_size=group_size))

    for i in range(1, model.num_encoder_layers):
        up = model.__getattr__(f"up{i+1}_{i}")
        if isinstance(up, Upsample):
            model.__setattr__(f"up{i+1}_{i}", EQUpsample2dPaper.from_upsample(up, group_size=group_size))

    up = model.__getattr__("up1_0")
    if isinstance(up, Upsample):
        model.__setattr__("up1_0", EQUpsample2dPaper.from_upsample(up, group_size=group_size))


def convert_pde_impl_to_eq_paper(
    model: PDEImpl,
    group_size: int = 8,
    *,
    eq_timestep_embedders: bool = True,
    eq_patch_embed: bool = True,
    eq_resample: bool = True,
    eq_output_path: bool = True,
    eq_relbias_symmetrize: bool = True,
) -> PDEImpl:
    if eq_timestep_embedders:
        local._convert_timestep_embedders_to_eq(model, group_size=group_size)
    if eq_patch_embed:
        model.x_embedder = local._convert_patch_embed_to_eq(model.x_embedder, group_size=group_size)
    if eq_resample:
        _convert_resize_blocks_to_eq_paper(model, group_size=group_size)
    if eq_output_path:
        local._convert_global_output_path_to_eq(model, group_size=group_size)

    for i in range(model.num_encoder_layers):
        stage = model.__getattr__(f"encoder_level_{i}")
        for j, block in enumerate(stage.blocks):
            stage.blocks[j] = convert_pde_block_to_eq_paper(
                block,
                group_size=group_size,
                relbias_symmetrize=eq_relbias_symmetrize,
            )

    for j, block in enumerate(model.latent.blocks):
        model.latent.blocks[j] = convert_pde_block_to_eq_paper(
            block,
            group_size=group_size,
            relbias_symmetrize=eq_relbias_symmetrize,
        )

    for i in range(model.num_encoder_layers):
        stage = model.__getattr__(f"decoder_level_{i}")
        for j, block in enumerate(stage.blocks):
            stage.blocks[j] = convert_pde_block_to_eq_paper(
                block,
                group_size=group_size,
                relbias_symmetrize=eq_relbias_symmetrize,
            )
    return model


@dataclass
class PaperEQPDEOutput(BaseOutput):
    sample: torch.Tensor


class PaperEQPDETransformer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        type: Optional[str] = "PDE-S",
        periodic: bool = True,
        carrier_token_active: bool = False,
        window_size: int = 8,
        patch_size: Optional[int] = 4,
        group_size: int = 8,
        eq_timestep_embedders: bool = True,
        eq_patch_embed: bool = True,
        eq_resample: bool = True,
        eq_output_path: bool = True,
        eq_relbias_symmetrize: bool = True,
        **kwargs,
    ):
        super().__init__()
        if not eq_relbias_symmetrize:
            raise ValueError(
                "The paper variant uses canonical pair-orbit relative bias. "
                "--eq_no_relbias_symmetrize is supported only by tiny_eq."
            )
        args = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "patch_size": patch_size,
            "periodic": periodic,
            "carrier_token_active": carrier_token_active,
            "window_size": window_size,
        }
        args.update(kwargs)

        has_custom_impl_args = any(
            key in kwargs for key in ("hidden_size", "depth", "num_heads", "mlp_ratio", "max_hidden_size")
        )
        if type in PDE_models and not has_custom_impl_args:
            self.model: PDEImpl = PDE_models[type](**args)
        else:
            self.model = PDEImpl(**args)

        convert_pde_impl_to_eq_paper(
            self.model,
            group_size=group_size,
            eq_timestep_embedders=eq_timestep_embedders,
            eq_patch_embed=eq_patch_embed,
            eq_resample=eq_resample,
            eq_output_path=eq_output_path,
            eq_relbias_symmetrize=eq_relbias_symmetrize,
        )

        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.group_size = group_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        output = self.model.forward(hidden_states, timestep, class_labels)
        if not return_dict:
            return (output,)
        return PaperEQPDEOutput(sample=output)
