"""Reconstruct project models and evaluation settings from saved run metadata."""

from pathlib import Path

import torch

from navierstokes_official import (
    OfficialTinyEQPDEModel,
    OfficialTinyPDEModel,
    OfficialTinyPaperEQPDEModel,
    _normalize_model_depth,
    disable_adaln_conditioning,
)


def build_model_from_args(args: dict) -> torch.nn.Module:
    variant = args.get("model_variant", "tiny")
    kwargs = {
        "sample_size": int(args.get("sample_size", 128)),
        "in_channels": int(args.get("history", args.get("in_channels", 1))),
        "out_channels": int(args.get("out_channels", 1)),
        "patch_size": int(args.get("patch_size", 4)),
        "window_size": int(args.get("window_size", 8)),
        "hidden_size": int(args.get("model_hidden_size", 48)),
        "num_heads": int(args.get("model_num_heads", 4)),
        "depth": _normalize_model_depth(args.get("model_depth")),
        "mlp_ratio": float(args.get("model_mlp_ratio", 3.0)),
    }
    if variant in {"tiny_eq", "tiny_eq_paper"}:
        kwargs["group_size"] = int(args.get("eq_group_size", 8))
        for name in ("timestep_embedders", "patch_embed", "resample", "output_path", "relbias_symmetrize"):
            kwargs[f"eq_{name}"] = not args.get(f"eq_no_{name}", False)
        if variant == "tiny_eq":
            kwargs["eq_token_linear_shared"] = bool(args.get("eq_token_linear_shared", False))
            kwargs["eq_attention_shared_qkv"] = bool(args.get("eq_attention_shared_qkv", False))
            model = OfficialTinyEQPDEModel(**kwargs)
        else:
            model = OfficialTinyPaperEQPDEModel(**kwargs)
    elif variant == "tiny":
        model = OfficialTinyPDEModel(**kwargs)
    else:
        raise ValueError(f"Unknown saved model_variant: {variant!r}")
    if args.get("disable_adaln", False):
        disable_adaln_conditioning(model)
    return model


def instantiate_from_checkpoint(ckpt_path: Path):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_model_from_args(payload.get("args", {}))
    model.load_state_dict(payload["model_state_dict"])
    return model.eval(), payload


def evaluation_data_args(payload: dict, split_seed=None, field_idx=None) -> dict:
    saved = payload.get("args", {})
    return {
        "history": int(saved.get("history", saved.get("in_channels", 1))),
        "horizon": 1,
        "split_seed": int(saved.get("split_seed", 0) if split_seed is None else split_seed),
        "field_idx": int(saved.get("field_idx", 0) if field_idx is None else field_idx),
    }


def common_evaluation_data_args(payloads: list[dict], split_seed=None, field_idx=None) -> dict:
    """A comparison must use the same held-out trajectories for every model."""
    configs = [evaluation_data_args(p, split_seed, field_idx) for p in payloads]
    if not configs:
        raise ValueError("At least one checkpoint is required.")
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError(
            "Checkpoint data settings differ. Compare runs with the same history, "
            "split_seed, and field_idx; use explicit split/field overrides only intentionally."
        )
    return configs[0]
