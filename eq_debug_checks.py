#!/usr/bin/env python3
"""Lightweight diagnostics for D4-equivariance in local EQ components.

Usage:
  python3 eq_debug_checks.py
"""

from __future__ import annotations

import numpy as np
import torch

import pde_transformer_eq_local as eq
from navierstokes_official import OfficialTinyEQPDEModel, _apply_d4_single

ACTIONS = list(eq.D4_ACTIONS)
G = len(ACTIONS)


def apply_spatial(x: torch.Tensor, h: int) -> torch.Tensor:
    k, b = ACTIONS[h]
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    if b:
        x = torch.flip(x, dims=(-1,))
    return x.contiguous()


def build_group_tables() -> tuple[list[list[int]], list[int]]:
    base = torch.arange(49, dtype=torch.float32).view(1, 1, 7, 7)

    def compose(a: int, b: int) -> int:
        y = apply_spatial(apply_spatial(base, b), a)
        for c in range(G):
            if torch.equal(y, apply_spatial(base, c)):
                return c
        raise RuntimeError("Group composition lookup failed")

    mul = [[compose(a, b) for b in range(G)] for a in range(G)]
    inv = [next(j for j in range(G) if mul[i][j] == 0 and mul[j][i] == 0) for i in range(G)]
    return mul, inv


MUL, INV = build_group_tables()


def rho_regular(x: torch.Tensor, h: int) -> torch.Tensor:
    # x: [B, G*C, H, W]
    bsz, gc, hgt, wid = x.shape
    c = gc // G
    xx = x.view(bsz, G, c, hgt, wid)
    src_idx = [MUL[INV[h]][g] for g in range(G)]
    y = xx[:, src_idx]
    y = apply_spatial(y, h)
    return y.reshape(bsz, gc, hgt, wid)


def rho_regular_tokens(x: torch.Tensor, h: int, window_size: int) -> torch.Tensor:
    # x: [B, N, G*C], N = window_size^2
    bsz, n_tokens, gc = x.shape
    c = gc // G
    xx = x.view(bsz, n_tokens, G, c)
    src_idx = [MUL[INV[h]][g] for g in range(G)]
    xx = xx[:, :, src_idx, :]

    idx = torch.arange(window_size * window_size, dtype=torch.long).view(window_size, window_size)
    idx = apply_spatial(idx.unsqueeze(0).unsqueeze(0).float(), h).squeeze(0).squeeze(0).long().reshape(-1)
    xx = xx[:, idx, :, :]
    return xx.reshape(bsz, n_tokens, gc)


def unpatchify_scalar(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    # x: [B, patch_size^2, H, W] -> [B, 1, H*patch_size, W*patch_size]
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(*x.shape[:3], patch_size, patch_size, 1)
    height = x.shape[1]
    width = x.shape[2]
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(-1, 1, height * patch_size, width * patch_size)


def check_attention_module() -> tuple[float, float]:
    ws = 8
    attn = eq.EQWindowAttention2DTime(
        dim=48,
        num_heads=3,
        qkv_bias=True,
        resolution=ws,
        group_size=8,
        use_relative_bias=False,
    )
    x = torch.randn(2, ws * ws, 48)
    with torch.no_grad():
        y = attn(x)
    errs = []
    for g in range(G):
        with torch.no_grad():
            lhs = attn(rho_regular_tokens(x, g, ws))
            rhs = rho_regular_tokens(y, g, ws)
        errs.append((lhs - rhs).abs().mean().item())
    return float(np.mean(errs)), float(np.max(errs))


def check_patch_head() -> tuple[float, float]:
    patch_size = 4
    head = eq.EQPatchOutConv2d(
        in_channels=16,
        out_channels=patch_size * patch_size,
        patch_size=patch_size,
        kernel_size=3,
        padding=1,
        group_size=8,
    )
    x = torch.randn(2, 16, 32, 32)
    with torch.no_grad():
        y = unpatchify_scalar(head(x), patch_size=patch_size)
    errs = []
    for g in range(G):
        with torch.no_grad():
            lhs = unpatchify_scalar(head(rho_regular(x, g)), patch_size=patch_size)
            rhs = apply_spatial(y, g)
        errs.append((lhs - rhs).abs().mean().item())
    return float(np.mean(errs)), float(np.max(errs))


def check_regular_module(module: torch.nn.Module, in_channels: int, h: int, w: int) -> tuple[float, float]:
    x = torch.randn(2, in_channels, h, w)
    with torch.no_grad():
        y = module(x)
    errs = []
    for g in range(G):
        with torch.no_grad():
            lhs = module(rho_regular(x, g))
            rhs = rho_regular(y, g)
        errs.append((lhs - rhs).abs().mean().item())
    return float(np.mean(errs)), float(np.max(errs))


def check_lift_pipeline() -> tuple[float, float]:
    lift = eq.EQConv2dOutShared(1, 16, 3, padding=1, group_size=8)
    shared = eq.EQConv2dShared(16, 16, 3, padding=1, group_size=8)
    down = eq.EQDownsample2d(16, 32, group_size=8)
    up = eq.EQUpsample2d(32, 16, group_size=8)

    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        y = up(down(shared(lift(x))))

    errs = []
    for g in range(G):
        with torch.no_grad():
            lhs = up(down(shared(lift(apply_spatial(x, g)))))
            rhs = rho_regular(y, g)
        errs.append((lhs - rhs).abs().mean().item())

    return float(np.mean(errs)), float(np.max(errs))


def check_full_model(patch_size: int = 4) -> tuple[float, float]:
    model = OfficialTinyEQPDEModel(
        sample_size=128,
        in_channels=1,
        out_channels=1,
        patch_size=patch_size,
        window_size=8,
    )
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, mean=0.0, std=0.02)

    x = torch.randn(1, 1, 128, 128)
    dt = torch.tensor([0.01])

    errs = []
    with torch.no_grad():
        fx = model(x, dt)
        for k in range(4):
            for b in (False, True):
                lhs = model(_apply_d4_single(x, k, b), dt)
                rhs = _apply_d4_single(fx, k, b)
                errs.append((lhs - rhs).abs().mean().item())

    return float(np.mean(errs)), float(np.max(errs))


def main() -> None:
    print("=== Regular-Feature Module Checks ===")
    m, mx = check_regular_module(eq.EQConv2dShared(16, 24, 3, padding=1, group_size=8), 16, 32, 32)
    print(f"EQConv2dShared: mean={m:.3e} max={mx:.3e}")

    m, mx = check_regular_module(eq.EQDownsample2d(32, 64, group_size=8), 32, 32, 32)
    print(f"EQDownsample2d: mean={m:.3e} max={mx:.3e}")

    m, mx = check_regular_module(eq.EQUpsample2d(64, 32, group_size=8), 64, 16, 16)
    print(f"EQUpsample2d: mean={m:.3e} max={mx:.3e}")

    print("\n=== Lift->Shared->Down->Up Pipeline ===")
    m, mx = check_lift_pipeline()
    print(f"Pipeline: mean={m:.3e} max={mx:.3e}")

    print("\n=== Attention/Patch Checks ===")
    m, mx = check_attention_module()
    print(f"EQWindowAttention2DTime: mean={m:.3e} max={mx:.3e}")
    m, mx = check_patch_head()
    print(f"EQPatchOutConv2d + unpatchify: mean={m:.3e} max={mx:.3e}")

    print("\n=== Full-Model (Random-Weight) Checks ===")
    m4, x4 = check_full_model(patch_size=4)
    print(f"OfficialTinyEQPDEModel patch_size=4: mean={m4:.3e} max={x4:.3e}")

    m1, x1 = check_full_model(patch_size=1)
    print(f"OfficialTinyEQPDEModel patch_size=1: mean={m1:.3e} max={x1:.3e}")


if __name__ == "__main__":
    main()
