"""MVP D4-equivariant PDE-Transformer components.

This module keeps the official PDE-Transformer topology and swaps selected
submodules with group-aware variants. By default it preserves token-space
capacity (dense QKV/MLP) to avoid confounding regressions from low-rank
projections; strict shared-token conversions are opt-in.
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

from pdetransformer.core.mixed_channels.pde_transformer import (
    Downsample,
    Mlp,
    OverlapPatchEmbed,
    PDEImpl,
    PDE_models,
    SimplePatchEmbed,
    Upsample,
    WindowAttention2DTime,
)


D4_ACTIONS = tuple((k, bool(b)) for k in range(4) for b in range(2))


def _build_d4_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """Build multiplication and inverse tables for the fixed D4 action ordering."""
    base = torch.arange(49, dtype=torch.float32).view(1, 1, 7, 7)

    def _apply_idx(idx: int, x: torch.Tensor) -> torch.Tensor:
        k, b = D4_ACTIONS[idx]
        if k:
            x = torch.rot90(x, k, dims=(-2, -1))
        if b:
            x = torch.flip(x, dims=(-1,))
        return x.contiguous()

    def _compose(a: int, b: int) -> int:
        # Group product as function composition: a ∘ b
        y = _apply_idx(a, _apply_idx(b, base))
        for c in range(len(D4_ACTIONS)):
            if torch.equal(y, _apply_idx(c, base)):
                return c
        raise RuntimeError("D4 composition lookup failed.")

    mul = torch.empty((len(D4_ACTIONS), len(D4_ACTIONS)), dtype=torch.long)
    for a in range(len(D4_ACTIONS)):
        for b in range(len(D4_ACTIONS)):
            mul[a, b] = _compose(a, b)

    inv = torch.empty((len(D4_ACTIONS),), dtype=torch.long)
    for g in range(len(D4_ACTIONS)):
        for cand in range(len(D4_ACTIONS)):
            if mul[g, cand].item() == 0 and mul[cand, g].item() == 0:
                inv[g] = cand
                break
        else:
            raise RuntimeError(f"Could not find inverse for D4 element index {g}.")

    return mul, inv


D4_MUL, D4_INV = _build_d4_tables()


def _d4_transform_kernel(weight: torch.Tensor, k: int, reflect: bool) -> torch.Tensor:
    """Apply spatial D4 transform to convolution kernels [O, I, Kh, Kw]."""
    if k:
        weight = torch.rot90(weight, k, dims=(-2, -1))
    if reflect:
        weight = torch.flip(weight, dims=(-1,))
    return weight


def _d4_transform_feature(x: torch.Tensor, k: int, reflect: bool) -> torch.Tensor:
    """Apply spatial D4 transform to feature maps [B, C, H, W]."""
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    if reflect:
        x = torch.flip(x, dims=(-1,))
    return x.contiguous()


def _d4_inverse_action(k: int, reflect: bool) -> tuple[int, bool]:
    # Under the rotate-then-reflect convention used in this project:
    # - pure rotations invert as r^{-k}
    # - reflection elements are involutions
    if reflect:
        return int(k), True
    return int((-k) % 4), False


def _d4_inverse_transform_kernel(weight: torch.Tensor, k: int, reflect: bool) -> torch.Tensor:
    k_inv, r_inv = _d4_inverse_action(k, reflect)
    return _d4_transform_kernel(weight, k_inv, r_inv)


def _d4_inverse_transform_feature(x: torch.Tensor, k: int, reflect: bool) -> torch.Tensor:
    k_inv, r_inv = _d4_inverse_action(k, reflect)
    return _d4_transform_feature(x, k_inv, r_inv)


def _d4_symmetrize_kernel(weight: torch.Tensor) -> torch.Tensor:
    """Project kernels to the D4-invariant subspace by orbit averaging."""
    acc = 0.0
    for k, b in D4_ACTIONS:
        acc = acc + _d4_transform_kernel(weight, k, b)
    return acc / float(len(D4_ACTIONS))


class EQLayerNorm(nn.Module):
    """LayerNorm with affine parameters shared across group fibers."""

    def __init__(self, dim: int, group_size: int = 8, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.group_size = int(group_size)
        self.norm = nn.LayerNorm(self.dim, elementwise_affine=False, eps=eps)

        if self.dim % self.group_size == 0:
            self.base_dim = self.dim // self.group_size
            self.weight = nn.Parameter(torch.ones(self.base_dim))
            self.bias = nn.Parameter(torch.zeros(self.base_dim))
        else:
            # Fallback keeps model usable for non-divisible dimensions.
            self.base_dim = None
            self.weight_full = nn.Parameter(torch.ones(self.dim))
            self.bias_full = nn.Parameter(torch.zeros(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.base_dim is None:
            return x * self.weight_full + self.bias_full
        weight = self.weight.repeat(self.group_size)
        bias = self.bias.repeat(self.group_size)
        return x * weight + bias


class EQLinearShared(nn.Module):
    """Shared linear map applied independently to each group fiber."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 8, bias: bool = True):
        super().__init__()
        if in_features % group_size != 0 or out_features % group_size != 0:
            raise ValueError(
                f"EQLinearShared requires divisibility by group_size={group_size}, "
                f"got in={in_features}, out={out_features}."
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.in_base = self.in_features // self.group_size
        self.out_base = self.out_features // self.group_size

        self.weight = nn.Parameter(torch.empty(self.out_base, self.in_base))
        self.bias = nn.Parameter(torch.zeros(self.out_base)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.group_size, self.in_base)
        y = torch.einsum("bgi,oi->bgo", x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 8) -> "EQLinearShared":
        """Project a dense linear layer to a shared-fiber approximation."""
        eq = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            bias=linear.bias is not None,
        )
        g = group_size
        w = linear.weight.detach().view(g, eq.out_base, g, eq.in_base)
        idx = torch.arange(g, device=w.device)
        diag = w[idx, :, idx, :]  # [g, out_base, in_base]
        eq.weight.data.copy_(diag.mean(dim=0))

        if linear.bias is not None:
            b = linear.bias.detach().view(g, eq.out_base)
            eq.bias.data.copy_(b.mean(dim=0))
        return eq


class EQLinearOutShared(nn.Module):
    """Linear layer with output fibers tied; input is treated as non-fibered."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 8, bias: bool = True):
        super().__init__()
        if out_features % group_size != 0:
            raise ValueError(
                f"EQLinearOutShared requires out_features divisible by group_size={group_size}, "
                f"got out={out_features}."
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.out_base = self.out_features // self.group_size
        self.weight = nn.Parameter(torch.empty(self.out_base, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_base)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_base = F.linear(x, self.weight, self.bias)
        reps = [1] * y_base.ndim
        reps[-1] = self.group_size
        return y_base.repeat(*reps)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 8) -> "EQLinearOutShared":
        eq = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            bias=linear.bias is not None,
        )
        g = group_size
        w = linear.weight.detach().view(g, eq.out_base, linear.in_features)
        eq.weight.data.copy_(w.mean(dim=0))
        if linear.bias is not None:
            b = linear.bias.detach().view(g, eq.out_base)
            eq.bias.data.copy_(b.mean(dim=0))
        return eq


class EQLinearOutSharedChunked(nn.Module):
    """Output-shared linear with chunk-preserving layout.

    This is designed for layers whose output is immediately split with
    `chunk(num_chunks, dim=-1)`, e.g. AdaLN modulation heads.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_chunks: int,
        group_size: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        if out_features % num_chunks != 0:
            raise ValueError(
                f"EQLinearOutSharedChunked requires out_features divisible by num_chunks={num_chunks}, "
                f"got out={out_features}."
            )
        chunk_width = out_features // num_chunks
        if chunk_width % group_size != 0:
            raise ValueError(
                f"Each chunk width must be divisible by group_size={group_size}, "
                f"got chunk_width={chunk_width}."
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_chunks = int(num_chunks)
        self.group_size = int(group_size)
        self.chunk_base = chunk_width // self.group_size

        self.weight = nn.Parameter(torch.empty(self.num_chunks, self.chunk_base, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.num_chunks, self.chunk_base)) if bias else None
        nn.init.xavier_uniform_(self.weight.view(self.num_chunks * self.chunk_base, self.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat_w = self.weight.view(self.num_chunks * self.chunk_base, self.in_features)
        flat_b = self.bias.view(-1) if self.bias is not None else None
        y = F.linear(x, flat_w, flat_b)  # [..., num_chunks * chunk_base]
        y = y.view(*y.shape[:-1], self.num_chunks, self.chunk_base)
        y = y.unsqueeze(-2)  # [..., num_chunks, 1, chunk_base]
        repeat_dims = [1] * y.ndim
        repeat_dims[-2] = self.group_size
        y = y.repeat(*repeat_dims)
        return y.reshape(*y.shape[:-3], self.out_features)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        num_chunks: int,
        group_size: int = 8,
    ) -> "EQLinearOutSharedChunked":
        eq = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            num_chunks=num_chunks,
            group_size=group_size,
            bias=linear.bias is not None,
        )
        chunk_width = linear.out_features // num_chunks
        chunk_base = chunk_width // group_size
        w = linear.weight.detach().view(num_chunks, group_size, chunk_base, linear.in_features)
        eq.weight.data.copy_(w.mean(dim=1))
        if linear.bias is not None:
            b = linear.bias.detach().view(num_chunks, group_size, chunk_base)
            eq.bias.data.copy_(b.mean(dim=1))
        return eq


class EQLinearInInvariant(nn.Module):
    """Linear map invariant to permutations of group fibers in the input."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 8, bias: bool = True):
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError(
                f"EQLinearInInvariant requires in_features divisible by group_size={group_size}, "
                f"got in={in_features}."
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.in_base = self.in_features // self.group_size
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_base))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.group_size, self.in_base).mean(dim=1)
        y = F.linear(x, self.weight, self.bias)
        return y.reshape(*shape[:-1], self.out_features)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 8) -> "EQLinearInInvariant":
        eq = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            bias=linear.bias is not None,
        )
        g = group_size
        w = linear.weight.detach().view(eq.out_features, g, eq.in_base)
        eq.weight.data.copy_(w.mean(dim=1))
        if linear.bias is not None:
            eq.bias.data.copy_(linear.bias.detach())
        return eq


class EQConv2dShared(nn.Module):
    """Equivariant Conv2d over group fibers (block-diagonal regular action)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        group_size: int = 8,
    ):
        super().__init__()
        if in_channels % group_size != 0 or out_channels % group_size != 0:
            raise ValueError(
                f"EQConv2dShared requires divisibility by group_size={group_size}, "
                f"got in={in_channels}, out={out_channels}."
            )
        self.group_size = int(group_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.in_base = self.in_channels // self.group_size
        self.out_base = self.out_channels // self.group_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = kernel_size
        self.kernel_size = (kh, kw)

        self.weight = nn.Parameter(torch.empty(self.out_base, self.in_base, kh, kw))
        self.bias = nn.Parameter(torch.zeros(self.out_base)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For output fiber g, use T_g(weight). This is equivariant under
        # simultaneous spatial D4 transforms + regular permutation of fibers.
        w_rep = torch.stack(
            [_d4_transform_kernel(self.weight, k, b) for k, b in D4_ACTIONS],
            dim=0,
        ).view(self.group_size * self.out_base, self.in_base, *self.kernel_size)
        b_rep = None
        if self.bias is not None:
            b_rep = self.bias.repeat(self.group_size)
        y = F.conv2d(
            x,
            w_rep,
            b_rep,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.group_size,
        )
        return y

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, group_size: int = 8) -> "EQConv2dShared":
        eq = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=conv.bias is not None,
            group_size=group_size,
        )
        g = group_size
        w = conv.weight.detach().view(g, eq.out_base, g, eq.in_base, *conv.kernel_size)
        idx = torch.arange(g, device=w.device)
        diag = w[idx, :, idx, ...]  # [g, out_base, in_base, kh, kw]
        canonical = []
        for g_idx, (k, b) in enumerate(D4_ACTIONS):
            canonical.append(_d4_inverse_transform_kernel(diag[g_idx], k, b))
        eq.weight.data.copy_(torch.stack(canonical, dim=0).mean(dim=0))

        if conv.bias is not None:
            b = conv.bias.detach().view(g, eq.out_base)
            eq.bias.data.copy_(b.mean(dim=0))
        return eq


class EQConv2dOutShared(nn.Module):
    """Lift Conv2d from non-fibered input to D4-fibered output."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        group_size: int = 8,
    ):
        super().__init__()
        if out_channels % group_size != 0:
            raise ValueError(
                f"EQConv2dOutShared requires out_channels divisible by group_size={group_size}, got out={out_channels}."
            )
        self.group_size = int(group_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.out_base = self.out_channels // self.group_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = kernel_size
        self.kernel_size = (kh, kw)

        self.weight = nn.Parameter(torch.empty(self.out_base, self.in_channels, kh, kw))
        self.bias = nn.Parameter(torch.zeros(self.out_base)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_rep = torch.stack(
            [_d4_transform_kernel(self.weight, k, b) for k, b in D4_ACTIONS],
            dim=0,
        ).view(self.group_size * self.out_base, self.in_channels, *self.kernel_size)
        b_rep = self.bias.repeat(self.group_size) if self.bias is not None else None
        return F.conv2d(
            x,
            w_rep,
            b_rep,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, group_size: int = 8) -> "EQConv2dOutShared":
        eq = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=conv.bias is not None,
            group_size=group_size,
        )
        g = group_size
        w = conv.weight.detach().view(g, eq.out_base, conv.in_channels, *conv.kernel_size)
        canonical = []
        for g_idx, (k, b) in enumerate(D4_ACTIONS):
            canonical.append(_d4_inverse_transform_kernel(w[g_idx], k, b))
        eq.weight.data.copy_(torch.stack(canonical, dim=0).mean(dim=0))
        if conv.bias is not None:
            b = conv.bias.detach().view(g, eq.out_base)
            eq.bias.data.copy_(b.mean(dim=0))
        return eq


class EQReduceCatConv(nn.Module):
    """Adapter for skip concatenation layout before grouped-equivariant 1x1 reduction."""

    def __init__(self, conv: nn.Module, group_size: int = 8):
        super().__init__()
        self.conv = conv
        self.group_size = int(group_size)

    def _cat_to_grouped(self, x: torch.Tensor) -> torch.Tensor:
        # Official forward concatenates tensors as [left_groups..., right_groups...].
        # Grouped-equivariant conv expects per-group packed channels:
        # [g0(left+right), g1(left+right), ...].
        channels = int(x.shape[1])
        if channels % 2 != 0:
            return x
        left_c = channels // 2
        right_c = channels - left_c
        if left_c % self.group_size != 0 or right_c % self.group_size != 0:
            return x

        bsz, _, h, w = x.shape
        left, right = x.split((left_c, right_c), dim=1)
        left = left.view(bsz, self.group_size, left_c // self.group_size, h, w)
        right = right.view(bsz, self.group_size, right_c // self.group_size, h, w)
        packed = torch.cat((left, right), dim=2)
        return packed.reshape(bsz, -1, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self._cat_to_grouped(x))


def _patch_permutations(patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return direct/inverse channel permutations for patch-grid D4 actions."""
    p = int(patch_size)
    idx = torch.arange(p * p, dtype=torch.long).view(p, p)
    perm_inv = []
    for k, b in D4_ACTIONS:
        transformed = _d4_transform_kernel(idx.unsqueeze(0).unsqueeze(0), k, b).squeeze(0).squeeze(0)
        # transformed[u] = old index at new location u => pi_g^{-1}(u)
        perm_inv.append(transformed.reshape(-1))
    perm_inv = torch.stack(perm_inv, dim=0)  # [G, P]
    perm = torch.argsort(perm_inv, dim=1)  # [G, P], perm[g, s] = pi_g(s)
    return perm, perm_inv


class EQPatchOutConv2d(nn.Module):
    """D4-equivariant patch-output head via explicit group averaging in forward."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        group_size: int = 8,
    ):
        super().__init__()
        self.group_size = int(group_size)
        if self.group_size != len(D4_ACTIONS):
            raise ValueError(
                f"EQPatchOutConv2d currently expects group_size={len(D4_ACTIONS)} for D4 regular-action channels, "
                f"got group_size={self.group_size}."
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.patch_size = int(patch_size)
        self.patch_area = self.patch_size * self.patch_size
        if self.out_channels % self.patch_area != 0:
            raise ValueError(
                f"EQPatchOutConv2d requires out_channels divisible by patch_size^2={self.patch_area}, "
                f"got out={out_channels}."
            )
        self.out_phys = self.out_channels // self.patch_area

        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = kernel_size
        self.kernel_size = (int(kh), int(kw))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels, kh, kw))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        _, perm_inv = _patch_permutations(self.patch_size)
        self.register_buffer("patch_perm_inv", perm_inv, persistent=False)
        # Under regular-action channels, input fibers permute by h^{-1}·g.
        src_idx = torch.empty((self.group_size, self.group_size), dtype=torch.long)
        for h in range(self.group_size):
            for g in range(self.group_size):
                src_idx[h, g] = D4_MUL[D4_INV[h], g]
        self.register_buffer("regular_src_idx", src_idx, persistent=False)
        self.register_buffer("d4_inv_idx", D4_INV.clone(), persistent=False)

    def _apply_regular_action_in(self, x: torch.Tensor, action_idx: int) -> torch.Tensor:
        bsz, channels, h, w = x.shape
        if channels % self.group_size != 0:
            raise ValueError(
                f"Expected regular-action channels divisible by {self.group_size}, got {channels}."
            )
        c_base = channels // self.group_size
        xx = x.view(bsz, self.group_size, c_base, h, w)
        xx = xx[:, self.regular_src_idx[action_idx]]
        k, reflect = D4_ACTIONS[action_idx]
        xx = _d4_transform_feature(xx, k, reflect)
        return xx.reshape(bsz, channels, h, w)

    def _apply_patch_action(self, y: torch.Tensor, action_idx: int) -> torch.Tensor:
        bsz, channels, h, w = y.shape
        yy = y.view(bsz, self.out_phys, self.patch_area, h, w)
        yy = yy[:, :, self.patch_perm_inv[action_idx], :, :]
        k, reflect = D4_ACTIONS[action_idx]
        yy = _d4_transform_feature(yy, k, reflect)
        return yy.reshape(bsz, channels, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}.")
        # Reynolds-style construction:
        # y(x) = 1/|G| sum_h pi(h^{-1}) [conv(rho(h) x)]
        acc = 0.0
        for h_idx in range(self.group_size):
            x_h = self._apply_regular_action_in(x, h_idx)
            y_h = F.conv2d(
                x_h,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            y_back = self._apply_patch_action(y_h, int(self.d4_inv_idx[h_idx].item()))
            acc = acc + y_back
        return acc / float(self.group_size)

    @classmethod
    def from_conv(
        cls, conv: nn.Conv2d, *, patch_size: int, group_size: int = 8
    ) -> "EQPatchOutConv2d":
        eq = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            patch_size=patch_size,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=conv.bias is not None,
            group_size=group_size,
        )

        # Initialize from the original conv; exact equivariance is enforced in forward().
        eq.weight.data.copy_(conv.weight.detach())
        if conv.bias is not None:
            eq.bias.data.copy_(conv.bias.detach())
        return eq


class EQDownsample2d(nn.Module):
    """D4-equivariant downsampling via average-pooling + grouped EQ convolution."""

    def __init__(self, in_channels: int, out_channels: int, group_size: int = 8):
        super().__init__()
        self.group_size = int(group_size)
        self.fallback: Optional[nn.Module] = None
        if in_channels % self.group_size == 0 and out_channels % self.group_size == 0:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.conv = EQConv2dShared(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                group_size=self.group_size,
            )
            self.body = None
        else:
            # Non-grouped fallback keeps behavior defined for unexpected widths.
            self.fallback = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            self.pool = None
            self.conv = None
            self.body = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fallback is not None:
            return self.fallback(x)
        x = self.pool(x)
        return self.conv(x)


class EQUpsample2d(nn.Module):
    """D4-equivariant upsampling via nearest-neighbor resize + grouped EQ conv."""

    def __init__(self, in_channels: int, out_channels: int, group_size: int = 8):
        super().__init__()
        self.group_size = int(group_size)
        self.fallback: Optional[nn.Module] = None
        if in_channels % self.group_size == 0 and out_channels % self.group_size == 0:
            self.conv = EQConv2dShared(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                group_size=self.group_size,
            )
        else:
            self.fallback = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fallback is not None:
            return self.fallback(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class EQMlp(nn.Module):
    """Fiber-shared MLP in token space."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        group_size: int = 8,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = EQLinearShared(in_features, hidden_features, group_size=group_size)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = EQLinearShared(hidden_features, out_features, group_size=group_size)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    @classmethod
    def from_mlp(cls, mlp: Mlp, group_size: int = 8) -> nn.Module:
        in_features = mlp.fc1.in_features
        hidden_features = mlp.fc1.out_features
        out_features = mlp.fc2.out_features
        if in_features % group_size != 0 or hidden_features % group_size != 0 or out_features % group_size != 0:
            # Keep original module if dimensions do not fit grouped parameterization.
            return mlp
        drop = float(getattr(mlp.drop, "p", 0.0))
        eq = cls(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=type(mlp.act),
            drop=drop,
            group_size=group_size,
        )
        eq.fc1 = EQLinearShared.from_linear(mlp.fc1, group_size=group_size)
        eq.fc2 = EQLinearShared.from_linear(mlp.fc2, group_size=group_size)
        return eq


class EQRelativeBiasD4(nn.Module):
    """Relative bias module (MVP: same structure as official module)."""

    def __init__(
        self,
        window_size: list[int],
        pretrained_window_size: list[int],
        num_heads: int,
        group_size: int = 8,
        no_log: bool = False,
    ):
        super().__init__()
        self.window_size = [int(ws) for ws in window_size]
        self.num_heads = int(num_heads)
        self.d4_symmetrize = True
        self.group_size = int(group_size)

        if 512 % self.group_size == 0:
            first = EQLinearOutShared(2, 512, group_size=self.group_size, bias=True)
            second = EQLinearInInvariant(512, self.num_heads, group_size=self.group_size, bias=False)
        else:
            first = nn.Linear(2, 512, bias=True)
            second = nn.Linear(512, self.num_heads, bias=False)
        self.cpb_mlp = nn.Sequential(first, nn.ReLU(inplace=True), second)

        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij")
        ).permute(1, 2, 0).contiguous().unsqueeze(0)

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        if not no_log:
            relative_coords_table *= 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / torch.log2(
                torch.tensor(8.0)
            )

        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).int()
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    @staticmethod
    def _transform_coords(coords: torch.Tensor, element: str) -> torch.Tensor:
        # coords[..., 0]=dh, coords[..., 1]=dw
        dh, dw = coords[..., 0], coords[..., 1]
        if element == "e":
            pass
        elif element == "r":
            dh, dw = -dw, dh
        elif element == "r2":
            dh, dw = -dh, -dw
        elif element == "r3":
            dh, dw = dw, -dh
        elif element == "s":
            dh, dw = dh, -dw
        elif element == "sr":
            dh, dw = -dw, -dh
        elif element == "sr2":
            dh, dw = -dh, dw
        elif element == "sr3":
            dh, dw = dw, dh
        else:
            raise ValueError(f"Unknown D4 element: {element}")
        return torch.stack((dh, dw), dim=-1)

    def forward(self, input_tensor: torch.Tensor, local_window_size: int) -> torch.Tensor:
        if self.d4_symmetrize:
            elements = ("e", "r", "r2", "r3", "s", "sr", "sr2", "sr3")
            table_vals = []
            for g in elements:
                table_g = self._transform_coords(self.relative_coords_table, g)
                table_vals.append(self.cpb_mlp(table_g))
            table = torch.stack(table_vals, dim=0).mean(dim=0).view(-1, self.num_heads)
        else:
            table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        n_global_feature = input_tensor.shape[2] - local_window_size
        relative_position_bias = F.pad(
            relative_position_bias,
            (n_global_feature, 0, n_global_feature, 0),
        ).contiguous()

        return input_tensor + relative_position_bias.unsqueeze(0)


class EQWindowAttention2DTime(nn.Module):
    """Window attention with optional shared-fiber QKV projection."""

    @staticmethod
    def _choose_compatible_heads(channels: int, requested_heads: int) -> int:
        requested_heads = max(1, int(requested_heads))
        channels = int(channels)
        for h in range(requested_heads, 0, -1):
            if channels % h == 0:
                return h
        return 1

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        resolution: int = 0,
        attn_type: str = "v2",
        group_size: int = 8,
        use_relative_bias: bool = False,
        shared_qkv: bool = False,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.shared_qkv = bool(shared_qkv)
        self.group_size = int(group_size)
        self.use_relative_bias = bool(use_relative_bias)
        self.fiber_attention = bool(self.shared_qkv and dim % self.group_size == 0)
        if self.fiber_attention:
            self.fiber_dim = int(dim // self.group_size)
            self.num_heads_fiber = self._choose_compatible_heads(self.fiber_dim, self.num_heads)
            self.head_dim = self.fiber_dim // self.num_heads_fiber
        else:
            if dim % self.num_heads != 0:
                raise ValueError(
                    f"WindowAttention requires dim divisible by num_heads in dense mode, got dim={dim}, heads={num_heads}."
                )
            self.fiber_dim = int(dim)
            self.num_heads_fiber = self.num_heads
            self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        if self.fiber_attention:
            self.qkv = EQLinearShared(dim, 3 * dim, group_size=self.group_size, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # Relative bias can be enabled later with a fully equivariant construction.
        # For now we default to no token-position bias to preserve permutation
        # equivariance within local windows.
        self.pos_emb_funct = None
        if self.use_relative_bias:
            self.pos_emb_funct = EQRelativeBiasD4(
                window_size=[resolution, resolution],
                pretrained_window_size=[resolution, resolution],
                num_heads=self.num_heads_fiber,
                group_size=self.group_size,
            )
        self.attn_type = attn_type
        if attn_type == "v2":
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((self.num_heads_fiber, 1, 1))),
                requires_grad=True,
            )

        self.resolution = int(resolution)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, n_tokens, channels = x.shape
        if self.fiber_attention:
            c_base = channels // self.group_size
            qkv = self.qkv(x).view(bsz, n_tokens, self.group_size, 3, c_base).permute(3, 0, 2, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # [B, G, N, Cb]
            q = q.view(bsz, self.group_size, n_tokens, self.num_heads_fiber, self.head_dim).permute(0, 1, 3, 2, 4)
            k = k.view(bsz, self.group_size, n_tokens, self.num_heads_fiber, self.head_dim).permute(0, 1, 3, 2, 4)
            v = v.view(bsz, self.group_size, n_tokens, self.num_heads_fiber, self.head_dim).permute(0, 1, 3, 2, 4)
        else:
            qkv = (
                self.qkv(x)
                .reshape(bsz, n_tokens, 3, self.num_heads, channels // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.attn_type == "v2":
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
            if self.fiber_attention:
                attn = attn * logit_scale.view(1, 1, self.num_heads_fiber, 1, 1)
            else:
                attn = attn * logit_scale
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.pos_emb_funct is not None:
            if self.fiber_attention:
                attn_view = attn.view(bsz * self.group_size, self.num_heads_fiber, n_tokens, n_tokens)
                attn_view = self.pos_emb_funct(attn_view, self.resolution**2)
                attn = attn_view.view(bsz, self.group_size, self.num_heads_fiber, n_tokens, n_tokens)
            else:
                attn = self.pos_emb_funct(attn, self.resolution**2)

        if attn_mask is not None:
            mask_shape = attn_mask.shape[0]
            if self.fiber_attention:
                attn = attn.view(
                    bsz // mask_shape,
                    mask_shape,
                    self.group_size,
                    self.num_heads_fiber,
                    n_tokens,
                    n_tokens,
                )
                attn = attn + attn_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                attn = attn.view(-1, self.group_size, self.num_heads_fiber, n_tokens, n_tokens)
            else:
                attn = attn.view(
                    bsz // mask_shape,
                    mask_shape,
                    self.num_heads,
                    n_tokens,
                    n_tokens,
                )
                attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
                attn = attn.view(-1, self.num_heads, n_tokens, n_tokens)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        if self.fiber_attention:
            c_base = channels // self.group_size
            out = out.permute(0, 1, 3, 2, 4).reshape(bsz, self.group_size, n_tokens, c_base)
            return out.permute(0, 2, 1, 3).reshape(bsz, n_tokens, channels)
        return out.transpose(1, 2).reshape(bsz, n_tokens, channels)

    @classmethod
    def from_attention(
        cls,
        attn: WindowAttention2DTime,
        group_size: int = 8,
        *,
        shared_qkv: bool = False,
    ) -> "EQWindowAttention2DTime":
        def _resize_head_rows(src: torch.Tensor, out_heads: int) -> torch.Tensor:
            src_heads = int(src.shape[0])
            if src_heads == out_heads:
                return src
            if src_heads % out_heads == 0:
                return src.view(out_heads, src_heads // out_heads, *src.shape[1:]).mean(dim=1)
            idx = torch.linspace(0, src_heads - 1, steps=out_heads, device=src.device).round().long()
            return src.index_select(0, idx)

        qkv_bias = attn.qkv.bias is not None
        qk_scale = attn.scale
        attn_drop = float(getattr(attn.attn_drop, "p", 0.0))
        use_relative_bias = bool(getattr(attn, "pos_emb_funct", None) is not None)
        eq = cls(
            dim=attn.qkv.in_features,
            num_heads=attn.num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=0.0,
            resolution=attn.resolution,
            attn_type=attn.attn_type,
            group_size=group_size,
            use_relative_bias=use_relative_bias,
            shared_qkv=shared_qkv,
        )
        if eq.fiber_attention and isinstance(eq.qkv, EQLinearShared):
            eq.qkv = EQLinearShared.from_linear(attn.qkv, group_size=group_size)
        else:
            eq.qkv.load_state_dict(attn.qkv.state_dict())
        if use_relative_bias and eq.pos_emb_funct is not None and attn.pos_emb_funct is not None:
            src0 = attn.pos_emb_funct.cpb_mlp[0]
            src2 = attn.pos_emb_funct.cpb_mlp[2]
            dst0 = eq.pos_emb_funct.cpb_mlp[0]
            dst2 = eq.pos_emb_funct.cpb_mlp[2]
            if isinstance(dst0, EQLinearOutShared) and isinstance(src0, nn.Linear):
                eq.pos_emb_funct.cpb_mlp[0] = EQLinearOutShared.from_linear(src0, group_size=group_size)
            if isinstance(dst2, EQLinearInInvariant) and isinstance(src2, nn.Linear):
                src_w = src2.weight.detach()
                if src_w.shape[1] % group_size == 0:
                    src_w = src_w.view(src_w.shape[0], group_size, src_w.shape[1] // group_size).mean(dim=1)
                src_w = _resize_head_rows(src_w, dst2.out_features)
                dst2.weight.data.copy_(src_w.to(dst2.weight.dtype))
            elif isinstance(dst2, nn.Linear) and isinstance(src2, nn.Linear):
                src_w = _resize_head_rows(src2.weight.detach(), dst2.out_features)
                dst2.weight.data.copy_(src_w.to(dst2.weight.dtype))
                if dst2.bias is not None and src2.bias is not None:
                    src_b = _resize_head_rows(src2.bias.detach().unsqueeze(1), dst2.out_features).squeeze(1)
                    dst2.bias.data.copy_(src_b.to(dst2.bias.dtype))
        if hasattr(attn, "logit_scale") and hasattr(eq, "logit_scale"):
            src_scale = attn.logit_scale.data.detach().view(-1)
            if src_scale.numel() == eq.logit_scale.numel():
                eq.logit_scale.data.copy_(src_scale.view_as(eq.logit_scale))
            else:
                eq.logit_scale.data.copy_(src_scale.mean().expand_as(eq.logit_scale))
        return eq


def convert_pde_block_to_eq(
    block: nn.Module,
    group_size: int = 8,
    relbias_symmetrize: bool = True,
    eq_token_linear_shared: bool = False,
    eq_attention_shared_qkv: bool = False,
) -> nn.Module:
    """In-place conversion for one PDEBlock instance."""
    if hasattr(block, "norm1") and isinstance(block.norm1, nn.LayerNorm):
        dim = int(block.norm1.normalized_shape[0])
        block.norm1 = EQLayerNorm(dim, group_size=group_size)
    if hasattr(block, "norm2") and isinstance(block.norm2, nn.LayerNorm):
        dim = int(block.norm2.normalized_shape[0])
        block.norm2 = EQLayerNorm(dim, group_size=group_size)

    if hasattr(block, "hat_norm1") and isinstance(block.hat_norm1, nn.LayerNorm):
        dim = int(block.hat_norm1.normalized_shape[0])
        block.hat_norm1 = EQLayerNorm(dim, group_size=group_size)
    if hasattr(block, "hat_norm2") and isinstance(block.hat_norm2, nn.LayerNorm):
        dim = int(block.hat_norm2.normalized_shape[0])
        block.hat_norm2 = EQLayerNorm(dim, group_size=group_size)

    if eq_token_linear_shared and hasattr(block, "mlp") and isinstance(block.mlp, Mlp):
        block.mlp = EQMlp.from_mlp(block.mlp, group_size=group_size)
    if eq_token_linear_shared and hasattr(block, "hat_mlp") and isinstance(block.hat_mlp, Mlp):
        block.hat_mlp = EQMlp.from_mlp(block.hat_mlp, group_size=group_size)

    if hasattr(block, "attn") and isinstance(block.attn, WindowAttention2DTime):
        block.attn = EQWindowAttention2DTime.from_attention(
            block.attn,
            group_size=group_size,
            shared_qkv=eq_attention_shared_qkv,
        )
    if hasattr(block, "attn") and hasattr(block.attn, "pos_emb_funct"):
        pos_emb = block.attn.pos_emb_funct
        if hasattr(pos_emb, "d4_symmetrize"):
            pos_emb.d4_symmetrize = bool(relbias_symmetrize)

    # AdaLN modulation takes global conditioning vectors (not group fibers) and
    # should emit tied per-fiber modulation parameters.
    if hasattr(block, "adain_2") and hasattr(block.adain_2, "linear") and isinstance(block.adain_2.linear, nn.Linear):
        linear = block.adain_2.linear
        if linear.out_features % 6 == 0 and (linear.out_features // 6) % group_size == 0:
            block.adain_2.linear = EQLinearOutSharedChunked.from_linear(
                linear,
                num_chunks=6,
                group_size=group_size,
            )
        elif linear.out_features % group_size == 0:
            block.adain_2.linear = EQLinearOutShared.from_linear(linear, group_size=group_size)
    if hasattr(block, "adain_1") and hasattr(block.adain_1, "linear") and isinstance(block.adain_1.linear, nn.Linear):
        linear = block.adain_1.linear
        if linear.out_features % 6 == 0 and (linear.out_features // 6) % group_size == 0:
            block.adain_1.linear = EQLinearOutSharedChunked.from_linear(
                linear,
                num_chunks=6,
                group_size=group_size,
            )
        elif linear.out_features % group_size == 0:
            block.adain_1.linear = EQLinearOutShared.from_linear(linear, group_size=group_size)

    return block


def _convert_conv2d_if_possible(conv: nn.Conv2d, group_size: int = 8, allow_out_shared: bool = False) -> nn.Module:
    if conv.in_channels % group_size == 0 and conv.out_channels % group_size == 0:
        return EQConv2dShared.from_conv(conv, group_size=group_size)
    if allow_out_shared and conv.out_channels % group_size == 0:
        return EQConv2dOutShared.from_conv(conv, group_size=group_size)
    return conv


def _convert_patch_embed_to_eq(x_embedder: nn.Module, group_size: int = 8) -> nn.Module:
    if isinstance(x_embedder, (SimplePatchEmbed, OverlapPatchEmbed)):
        x_embedder.proj = _convert_conv2d_if_possible(
            x_embedder.proj,
            group_size=group_size,
            allow_out_shared=True,
        )
    return x_embedder


def _convert_resize_blocks_to_eq(model: PDEImpl, group_size: int = 8) -> None:
    # Encoder downsampling
    for i in range(model.num_encoder_layers):
        down = model.__getattr__(f"down{i}_{i+1}")
        if isinstance(down, Downsample):
            conv = down.body[0]
            # Original block: conv -> PixelUnshuffle(2), so channels are multiplied by 4.
            out_channels = int(conv.out_channels * 4)
            model.__setattr__(
                f"down{i}_{i+1}",
                EQDownsample2d(
                    in_channels=int(conv.in_channels),
                    out_channels=out_channels,
                    group_size=group_size,
                ),
            )

    # Decoder upsampling
    for i in range(1, model.num_encoder_layers):
        up = model.__getattr__(f"up{i+1}_{i}")
        if isinstance(up, Upsample):
            conv = up.body[0]
            # Original block: conv -> PixelShuffle(2), so channels are divided by 4.
            out_channels = int(conv.out_channels // 4)
            model.__setattr__(
                f"up{i+1}_{i}",
                EQUpsample2d(
                    in_channels=int(conv.in_channels),
                    out_channels=out_channels,
                    group_size=group_size,
                ),
            )

    up = model.__getattr__("up1_0")
    if isinstance(up, Upsample):
        conv = up.body[0]
        out_channels = int(conv.out_channels // 4)
        model.__setattr__(
            "up1_0",
            EQUpsample2d(
                in_channels=int(conv.in_channels),
                out_channels=out_channels,
                group_size=group_size,
            ),
        )


def _convert_timestep_embedders_to_eq(model: PDEImpl, group_size: int = 8) -> None:
    for i in range(model.num_encoder_layers + 1):
        name = f"t_embedder_{i}"
        if not hasattr(model, name):
            continue
        t_embedder = model.__getattr__(name)
        if not hasattr(t_embedder, "mlp") or not isinstance(t_embedder.mlp, nn.Sequential):
            continue

        if len(t_embedder.mlp) >= 1 and isinstance(t_embedder.mlp[0], nn.Linear):
            l0 = t_embedder.mlp[0]
            if l0.out_features % group_size == 0:
                t_embedder.mlp[0] = EQLinearOutShared.from_linear(l0, group_size=group_size)

        if len(t_embedder.mlp) >= 3 and isinstance(t_embedder.mlp[2], nn.Linear):
            l2 = t_embedder.mlp[2]
            if l2.in_features % group_size == 0 and l2.out_features % group_size == 0:
                t_embedder.mlp[2] = EQLinearShared.from_linear(l2, group_size=group_size)
            elif l2.out_features % group_size == 0:
                t_embedder.mlp[2] = EQLinearOutShared.from_linear(l2, group_size=group_size)


def _convert_global_output_path_to_eq(model: PDEImpl, group_size: int = 8) -> None:
    # Skip-path channel reducers in decoder.
    for i in range(model.num_encoder_layers):
        name = f"reduce_chan_level{i}"
        if hasattr(model, name):
            conv = model.__getattr__(name)
            if isinstance(conv, nn.Conv2d):
                converted = _convert_conv2d_if_possible(conv, group_size=group_size, allow_out_shared=False)
                if isinstance(converted, EQConv2dShared):
                    converted = EQReduceCatConv(converted, group_size=group_size)
                model.__setattr__(
                    name,
                    converted,
                )

    # Final refinement conv before head.
    if hasattr(model, "output") and isinstance(model.output, nn.Conv2d):
        model.output = _convert_conv2d_if_possible(model.output, group_size=group_size, allow_out_shared=False)

    # Final AdaLN modulation + output projection.
    if hasattr(model, "final_layer"):
        final_layer = model.final_layer
        if hasattr(final_layer, "out_proj") and isinstance(final_layer.out_proj, nn.Conv2d):
            patch_size = int(getattr(model, "patch_size", 1))
            # Preserve patch-channel semantics: final head maps from D4 fibers to
            # patch-position channels using an explicit equivariant tying scheme.
            if (
                final_layer.out_proj.in_channels > 0
                and final_layer.out_proj.out_channels % (patch_size * patch_size) == 0
            ):
                final_layer.out_proj = EQPatchOutConv2d.from_conv(
                    final_layer.out_proj, patch_size=patch_size, group_size=group_size
                )
            else:
                final_layer.out_proj = _convert_conv2d_if_possible(
                    final_layer.out_proj, group_size=group_size, allow_out_shared=False
                )
        if (
            hasattr(final_layer, "adaLN_modulation")
            and isinstance(final_layer.adaLN_modulation, nn.Sequential)
            and len(final_layer.adaLN_modulation) > 0
            and isinstance(final_layer.adaLN_modulation[-1], nn.Linear)
        ):
            linear = final_layer.adaLN_modulation[-1]
            if linear.out_features % 2 == 0 and (linear.out_features // 2) % group_size == 0:
                final_layer.adaLN_modulation[-1] = EQLinearOutSharedChunked.from_linear(
                    linear,
                    num_chunks=2,
                    group_size=group_size,
                )
            elif linear.out_features % group_size == 0:
                final_layer.adaLN_modulation[-1] = EQLinearOutShared.from_linear(linear, group_size=group_size)


def convert_pde_impl_to_eq(
    model: PDEImpl,
    group_size: int = 8,
    *,
    eq_timestep_embedders: bool = True,
    eq_patch_embed: bool = True,
    eq_resample: bool = True,
    eq_output_path: bool = True,
    eq_relbias_symmetrize: bool = True,
    eq_token_linear_shared: bool = False,
    eq_attention_shared_qkv: bool = False,
) -> PDEImpl:
    """In-place conversion of official PDEImpl to EQ MVP modules."""
    if eq_timestep_embedders:
        _convert_timestep_embedders_to_eq(model, group_size=group_size)
    if eq_patch_embed:
        model.x_embedder = _convert_patch_embed_to_eq(model.x_embedder, group_size=group_size)
    if eq_resample:
        _convert_resize_blocks_to_eq(model, group_size=group_size)
    if eq_output_path:
        _convert_global_output_path_to_eq(model, group_size=group_size)

    for i in range(model.num_encoder_layers):
        stage = model.__getattr__(f"encoder_level_{i}")
        for j, block in enumerate(stage.blocks):
            stage.blocks[j] = convert_pde_block_to_eq(
                block,
                group_size=group_size,
                relbias_symmetrize=eq_relbias_symmetrize,
                eq_token_linear_shared=eq_token_linear_shared,
                eq_attention_shared_qkv=eq_attention_shared_qkv,
            )

    for j, block in enumerate(model.latent.blocks):
        model.latent.blocks[j] = convert_pde_block_to_eq(
            block,
            group_size=group_size,
            relbias_symmetrize=eq_relbias_symmetrize,
            eq_token_linear_shared=eq_token_linear_shared,
            eq_attention_shared_qkv=eq_attention_shared_qkv,
        )

    for i in range(model.num_encoder_layers):
        stage = model.__getattr__(f"decoder_level_{i}")
        for j, block in enumerate(stage.blocks):
            stage.blocks[j] = convert_pde_block_to_eq(
                block,
                group_size=group_size,
                relbias_symmetrize=eq_relbias_symmetrize,
                eq_token_linear_shared=eq_token_linear_shared,
                eq_attention_shared_qkv=eq_attention_shared_qkv,
            )
    return model


class EQPDEImpl(PDEImpl):
    """Official PDEImpl + in-place MVP EQ conversion."""

    def __init__(
        self,
        *args,
        group_size: int = 8,
        eq_timestep_embedders: bool = True,
        eq_patch_embed: bool = True,
        eq_resample: bool = True,
        eq_output_path: bool = True,
        eq_relbias_symmetrize: bool = True,
        eq_token_linear_shared: bool = False,
        eq_attention_shared_qkv: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.group_size = int(group_size)
        convert_pde_impl_to_eq(
            self,
            group_size=self.group_size,
            eq_timestep_embedders=eq_timestep_embedders,
            eq_patch_embed=eq_patch_embed,
            eq_resample=eq_resample,
            eq_output_path=eq_output_path,
            eq_relbias_symmetrize=eq_relbias_symmetrize,
            eq_token_linear_shared=eq_token_linear_shared,
            eq_attention_shared_qkv=eq_attention_shared_qkv,
        )


@dataclass
class EQPDEOutput(BaseOutput):
    sample: torch.Tensor


class EQPDETransformer(ModelMixin, ConfigMixin):
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
        eq_token_linear_shared: bool = False,
        eq_attention_shared_qkv: bool = False,
        **kwargs,
    ):
        super().__init__()
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

        # Match official presets unless explicit architectural overrides are provided.
        if type in PDE_models and not has_custom_impl_args:
            self.model: PDEImpl = PDE_models[type](**args)
        else:
            self.model = PDEImpl(**args)
        convert_pde_impl_to_eq(
            self.model,
            group_size=group_size,
            eq_timestep_embedders=eq_timestep_embedders,
            eq_patch_embed=eq_patch_embed,
            eq_resample=eq_resample,
            eq_output_path=eq_output_path,
            eq_relbias_symmetrize=eq_relbias_symmetrize,
            eq_token_linear_shared=eq_token_linear_shared,
            eq_attention_shared_qkv=eq_attention_shared_qkv,
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
        return EQPDEOutput(sample=output)
