#!/usr/bin/env python3
"""Deep layer-by-layer audit for local D4-equivariant PDE-Transformer components.

Runs deterministic numerical checks for:
- Custom EQ primitives (linear, norm, conv, patch head, resize, attention)
- Conversion integrity (module graph audit)
- End-to-end random-weight equivariance and train-step stability
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

import pde_transformer_eq_local as eq
from navierstokes_official import OfficialTinyEQPDEModel

torch.manual_seed(0)


@dataclass
class CheckResult:
    name: str
    value: float
    threshold: float

    @property
    def ok(self) -> bool:
        return self.value <= self.threshold


G = len(eq.D4_ACTIONS)
MUL = eq.D4_MUL
INV = eq.D4_INV


def apply_spatial(x: torch.Tensor, g: int) -> torch.Tensor:
    k, b = eq.D4_ACTIONS[g]
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    if b:
        x = torch.flip(x, dims=(-1,))
    return x.contiguous()


def rho_regular_feat(x: torch.Tensor, g: int) -> torch.Tensor:
    # x: [B, G*C, H, W]
    bsz, gc, h, w = x.shape
    c = gc // G
    xx = x.view(bsz, G, c, h, w)
    src_idx = [int(MUL[int(INV[g]), gg]) for gg in range(G)]
    xx = xx[:, src_idx]
    xx = apply_spatial(xx, g)
    return xx.reshape(bsz, gc, h, w)


def rho_regular_tokens(x: torch.Tensor, g: int) -> torch.Tensor:
    # x: [B, N, G*C]
    bsz, n, gc = x.shape
    c = gc // G
    xx = x.view(bsz, n, G, c)
    src_idx = [int(MUL[int(INV[g]), gg]) for gg in range(G)]
    xx = xx[:, :, src_idx, :]
    return xx.reshape(bsz, n, gc)


def rho_tokens_window(x: torch.Tensor, g: int, ws: int) -> torch.Tensor:
    # x: [B, N, G*C], N=ws*ws
    bsz, n, gc = x.shape
    assert n == ws * ws
    c = gc // G
    xx = x.view(bsz, ws, ws, G, c)
    src_idx = [int(MUL[int(INV[g]), gg]) for gg in range(G)]
    xx = xx[:, :, :, src_idx, :]
    k, b = eq.D4_ACTIONS[g]
    if k:
        xx = torch.rot90(xx, k, dims=(1, 2))
    if b:
        xx = torch.flip(xx, dims=(2,))
    return xx.reshape(bsz, n, gc)


def token_perm_indices(ws: int, g: int) -> torch.Tensor:
    idx = torch.arange(ws * ws, dtype=torch.long).view(1, 1, ws, ws)
    idx_t = apply_spatial(idx, g).reshape(-1)
    return idx_t


def unpatchify_scalar(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    # x: [B, patch_size^2, H, W] -> [B, 1, H*patch_size, W*patch_size]
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(*x.shape[:3], patch_size, patch_size, 1)
    height = x.shape[1]
    width = x.shape[2]
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(-1, 1, height * patch_size, width * patch_size)


def rho_chunked_tokens(x: torch.Tensor, g: int, num_chunks: int, group_size: int) -> torch.Tensor:
    # x: [B, N, num_chunks * group_size * chunk_base]
    bsz, n, d = x.shape
    chunk_width = d // num_chunks
    chunk_base = chunk_width // group_size
    xx = x.view(bsz, n, num_chunks, group_size, chunk_base)
    src_idx = [int(MUL[int(INV[g]), gg]) for gg in range(group_size)]
    xx = xx[:, :, :, src_idx, :]
    return xx.reshape(bsz, n, d)


def mean_abs(x: torch.Tensor) -> float:
    return float(x.abs().mean().item())


def check_equiv_layer(
    layer: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    rho_in: Callable[[torch.Tensor, int], torch.Tensor],
    rho_out: Callable[[torch.Tensor, int], torch.Tensor],
    name: str,
    threshold: float = 1e-5,
) -> CheckResult:
    with torch.no_grad():
        y = layer(x)
    errs = []
    for g in range(G):
        with torch.no_grad():
            lhs = layer(rho_in(x, g))
            rhs = rho_out(y, g)
        errs.append(mean_abs(lhs - rhs))
    return CheckResult(name=name, value=sum(errs) / len(errs), threshold=threshold)


def run_deep_audit() -> list[CheckResult]:
    results: list[CheckResult] = []

    # 1) EQLayerNorm commutes with regular channel permutations.
    ln = eq.EQLayerNorm(dim=48, group_size=8).eval()
    x_ln = torch.randn(3, 64, 48)
    results.append(
        check_equiv_layer(
            ln,
            x_ln,
            rho_in=rho_regular_tokens,
            rho_out=rho_regular_tokens,
            name="EQLayerNorm",
        )
    )

    # 2) Shared linear maps
    l_shared = eq.EQLinearShared(48, 96, group_size=8).eval()
    x_ls = torch.randn(2, 40, 48)
    results.append(
        check_equiv_layer(
            l_shared,
            x_ls,
            rho_in=rho_regular_tokens,
            rho_out=rho_regular_tokens,
            name="EQLinearShared",
        )
    )

    # 3) Output-shared maps: output should be invariant to regular channel action.
    l_out = eq.EQLinearOutShared(12, 48, group_size=8).eval()
    x_lo = torch.randn(2, 40, 12)
    with torch.no_grad():
        y_lo = l_out(x_lo)
    errs_lo = [mean_abs(rho_regular_tokens(y_lo, g) - y_lo) for g in range(G)]
    results.append(
        CheckResult(
            name="EQLinearOutShared(output-invariant)",
            value=sum(errs_lo) / len(errs_lo),
            threshold=1e-6,
        )
    )

    # 4) Chunked output-shared maps
    l_chunk = eq.EQLinearOutSharedChunked(12, 6 * 48, num_chunks=6, group_size=8).eval()
    with torch.no_grad():
        y_chunk = l_chunk(x_lo)
    errs_chunk = [mean_abs(rho_chunked_tokens(y_chunk, g, num_chunks=6, group_size=8) - y_chunk) for g in range(G)]
    results.append(
        CheckResult(
            name="EQLinearOutSharedChunked(output-invariant)",
            value=sum(errs_chunk) / len(errs_chunk),
            threshold=1e-6,
        )
    )

    # 5) Input-invariant map
    l_inv = eq.EQLinearInInvariant(48, 7, group_size=8).eval()
    x_li = torch.randn(2, 50, 48)
    with torch.no_grad():
        y_li = l_inv(x_li)
    errs_li = []
    for g in range(G):
        with torch.no_grad():
            lhs = l_inv(rho_regular_tokens(x_li, g))
        errs_li.append(mean_abs(lhs - y_li))
    results.append(CheckResult(name="EQLinearInInvariant", value=sum(errs_li) / len(errs_li), threshold=1e-6))

    # 6) Conv primitives
    conv_shared = eq.EQConv2dShared(16, 24, 3, padding=1, group_size=8).eval()
    x_cs = torch.randn(2, 16, 32, 32)
    results.append(
        check_equiv_layer(
            conv_shared,
            x_cs,
            rho_in=rho_regular_feat,
            rho_out=rho_regular_feat,
            name="EQConv2dShared",
        )
    )

    conv_lift = eq.EQConv2dOutShared(1, 16, 3, padding=1, group_size=8).eval()
    x_cl = torch.randn(2, 1, 32, 32)
    results.append(
        check_equiv_layer(
            conv_lift,
            x_cl,
            rho_in=apply_spatial,
            rho_out=rho_regular_feat,
            name="EQConv2dOutShared",
        )
    )

    # 7) Skip-concat reducer adapter
    reduce_conv = eq.EQReduceCatConv(eq.EQConv2dShared(32, 16, 1, group_size=8), group_size=8).eval()
    x_left = torch.randn(2, 16, 32, 32)
    x_right = torch.randn(2, 16, 32, 32)
    x_cat = torch.cat([x_left, x_right], dim=1)

    def rho_cat(z: torch.Tensor, g: int) -> torch.Tensor:
        zl, zr = z.split([16, 16], dim=1)
        return torch.cat([rho_regular_feat(zl, g), rho_regular_feat(zr, g)], dim=1)

    results.append(
        check_equiv_layer(
            reduce_conv,
            x_cat,
            rho_in=rho_cat,
            rho_out=rho_regular_feat,
            name="EQReduceCatConv",
        )
    )

    # 8) Patch-out head
    patch_head = eq.EQPatchOutConv2d(
        in_channels=16,
        out_channels=16,
        patch_size=4,
        kernel_size=3,
        padding=1,
        group_size=8,
    ).eval()
    x_ph = torch.randn(2, 16, 16, 16)
    results.append(
        check_equiv_layer(
            lambda z: unpatchify_scalar(patch_head(z), patch_size=4),
            x_ph,
            rho_in=rho_regular_feat,
            rho_out=apply_spatial,
            name="EQPatchOutConv2d",
            threshold=1e-5,
        )
    )

    # 9) Resize operators
    down = eq.EQDownsample2d(16, 32, group_size=8).eval()
    up = eq.EQUpsample2d(32, 16, group_size=8).eval()
    x_down = torch.randn(2, 16, 32, 32)
    results.append(
        check_equiv_layer(
            down,
            x_down,
            rho_in=rho_regular_feat,
            rho_out=rho_regular_feat,
            name="EQDownsample2d",
        )
    )
    x_up = torch.randn(2, 32, 16, 16)
    results.append(
        check_equiv_layer(
            up,
            x_up,
            rho_in=rho_regular_feat,
            rho_out=rho_regular_feat,
            name="EQUpsample2d",
        )
    )

    # 10) Token MLP
    mlp = eq.EQMlp(48, 96, out_features=48, group_size=8).eval()
    x_mlp = torch.randn(2, 64, 48)
    results.append(
        check_equiv_layer(
            mlp,
            x_mlp,
            rho_in=rho_regular_tokens,
            rho_out=rho_regular_tokens,
            name="EQMlp",
        )
    )

    # 11) Relative bias tensor equivariance under token permutation.
    rb = eq.EQRelativeBiasD4(window_size=[8, 8], pretrained_window_size=[8, 8], num_heads=4, group_size=8).eval()
    with torch.no_grad():
        b0 = rb(torch.zeros(1, 4, 64, 64), local_window_size=64)[0]
    errs_rb = []
    for g in range(G):
        p = token_perm_indices(8, g)
        bp = b0[:, p][:, :, p]
        errs_rb.append(mean_abs(bp - b0))
    results.append(CheckResult(name="EQRelativeBiasD4(token-conjugation)", value=sum(errs_rb) / len(errs_rb), threshold=5e-3))

    # 12) Attention paths (without and with relative bias).
    attn_plain = eq.EQWindowAttention2DTime(
        dim=48,
        num_heads=4,
        qkv_bias=True,
        resolution=8,
        group_size=8,
        use_relative_bias=False,
        shared_qkv=True,
    ).eval()
    x_attn = torch.randn(2, 64, 48)
    results.append(
        check_equiv_layer(
            attn_plain,
            x_attn,
            rho_in=lambda t, g: rho_tokens_window(t, g, 8),
            rho_out=lambda t, g: rho_tokens_window(t, g, 8),
            name="EQWindowAttention2DTime(no-rbias)",
        )
    )

    attn_rb = eq.EQWindowAttention2DTime(
        dim=48,
        num_heads=4,
        qkv_bias=True,
        resolution=8,
        group_size=8,
        use_relative_bias=True,
        shared_qkv=True,
    ).eval()
    results.append(
        check_equiv_layer(
            attn_rb,
            x_attn,
            rho_in=lambda t, g: rho_tokens_window(t, g, 8),
            rho_out=lambda t, g: rho_tokens_window(t, g, 8),
            name="EQWindowAttention2DTime(with-rbias)",
            threshold=8e-3,
        )
    )

    # 13) Model structure + no fallback checks.
    model = OfficialTinyEQPDEModel(
        sample_size=128,
        in_channels=1,
        out_channels=1,
        patch_size=4,
        window_size=8,
        eq_token_linear_shared=True,
        eq_attention_shared_qkv=True,
    ).eval()
    core = model.model.model
    type_checks = {
        "x_embedder.proj": isinstance(core.x_embedder.proj, eq.EQConv2dOutShared),
        "final_layer.out_proj": isinstance(core.final_layer.out_proj, eq.EQPatchOutConv2d),
    }
    for i in range(core.num_encoder_layers):
        type_checks[f"down{i}_{i+1}"] = isinstance(getattr(core, f"down{i}_{i+1}"), eq.EQDownsample2d)
    for i in range(1, core.num_encoder_layers):
        type_checks[f"up{i+1}_{i}"] = isinstance(getattr(core, f"up{i+1}_{i}"), eq.EQUpsample2d)
    type_checks["up1_0"] = isinstance(getattr(core, "up1_0"), eq.EQUpsample2d)
    for i in range(core.num_encoder_layers):
        type_checks[f"reduce_chan_level{i}"] = isinstance(getattr(core, f"reduce_chan_level{i}"), eq.EQReduceCatConv)

    bad_types = [name for name, ok in type_checks.items() if not ok]
    results.append(CheckResult(name="EQModelStructure(types)", value=float(len(bad_types)), threshold=0.0))

    fallback_count = 0
    for module in core.modules():
        if isinstance(module, (eq.EQDownsample2d, eq.EQUpsample2d)):
            if getattr(module, "fallback", None) is not None:
                fallback_count += 1
    results.append(CheckResult(name="EQResizeFallbackCount", value=float(fallback_count), threshold=0.0))

    # 14) End-to-end random-weight equivariance.
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
    x_full = torch.randn(1, 1, 128, 128)
    dt = torch.tensor([0.01], dtype=x_full.dtype)
    with torch.no_grad():
        y0 = model(x_full, dt)
    errs_full = []
    for g in range(G):
        with torch.no_grad():
            lhs = model(apply_spatial(x_full, g), dt)
            rhs = apply_spatial(y0, g)
        errs_full.append(mean_abs(lhs - rhs))
    results.append(CheckResult(name="EQFullModel(random-weights)", value=sum(errs_full) / len(errs_full), threshold=1e-5))

    # 15) One train step stability check (finite grads/loss).
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_train = torch.randn(2, 1, 128, 128)
    y_train = torch.randn(2, 1, 128, 128)
    dt_train = torch.full((2,), 0.01)
    pred = model(x_train, dt_train)
    loss = F.mse_loss(pred, y_train)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    grad_finite = True
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            grad_finite = False
            break
    loss_finite = bool(torch.isfinite(loss))
    results.append(CheckResult(name="EQTrainStep(finite-loss)", value=0.0 if loss_finite else 1.0, threshold=0.0))
    results.append(CheckResult(name="EQTrainStep(finite-grads)", value=0.0 if grad_finite else 1.0, threshold=0.0))

    return results


def main() -> None:
    results = run_deep_audit()
    width = max(len(r.name) for r in results) + 2
    failures = []
    print("=== EQ Deep Audit ===")
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"{status:4} {r.name:<{width}} value={r.value:.6e}  threshold={r.threshold:.6e}")
        if not r.ok:
            failures.append(r)

    if failures:
        names = ", ".join(f.name for f in failures)
        raise SystemExit(f"\nAudit failed: {len(failures)} check(s) failed -> {names}")

    print("\nAll deep audit checks passed.")


if __name__ == "__main__":
    main()
