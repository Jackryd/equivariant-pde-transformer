import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from neuralop.models import FNO

from checkpoint_utils import evaluation_data_args, instantiate_from_checkpoint
from data_handling import PDEBenchShallowWaterDataset
from navierstokes_official import (
    d4_group_average_predict,
    resolve_output_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full-test rollout RMSE curves for multiple models.")
    parser.add_argument("--base48-checkpoint", type=str, required=True)
    parser.add_argument("--aug48-checkpoint", type=str, required=True)
    parser.add_argument("--reg48-checkpoint", type=str, required=True)
    parser.add_argument("--eq112-checkpoint", type=str, required=True)
    parser.add_argument("--fno-checkpoint", type=str, required=True)
    parser.add_argument("--base48-long-checkpoint", type=str, default=None)
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--field-idx", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--artifact-prefix",
        "--artifact_prefix",
        dest="artifact_prefix",
        type=str,
        default="full_test_rollout_eval",
    )
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


load_transformer_checkpoint = instantiate_from_checkpoint


def load_fno_checkpoint(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(ckpt_path, map_location="cpu")
    kwargs = payload["model_kwargs"]
    model = FNO(
        n_modes=tuple(kwargs["n_modes"]),
        in_channels=int(kwargs["in_channels"]),
        out_channels=int(kwargs["out_channels"]),
        hidden_channels=int(kwargs["hidden_channels"]),
        n_layers=int(kwargs["n_layers"]),
        positional_embedding="grid",
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def rollout_transformer(
    model: torch.nn.Module,
    x0: torch.Tensor,
    dt: torch.Tensor,
    n_steps: int,
    use_group_average: bool = False,
) -> torch.Tensor:
    states = [x0]
    for _ in range(int(n_steps)):
        if use_group_average:
            states.append(d4_group_average_predict(model, states[-1], dt))
        else:
            states.append(model(states[-1], dt))
    return torch.stack(states, dim=2)


def rollout_fno(model: torch.nn.Module, x0: torch.Tensor, _dt: torch.Tensor, n_steps: int) -> torch.Tensor:
    states = [x0]
    for _ in range(int(n_steps)):
        states.append(model(states[-1]))
    return torch.stack(states, dim=2)


def compute_curve(
    model: torch.nn.Module,
    dataset: PDEBenchShallowWaterDataset,
    device: torch.device,
    rollout_steps: int,
    batch_size: int,
    model_kind: str,
    use_group_average: bool = False,
) -> tuple[torch.Tensor, dict]:
    model.to(device)
    model.eval()
    if rollout_steps >= dataset.num_frames:
        raise ValueError("Requested rollout exceeds the available trajectory length.")
    n = len(dataset.trajectory_indices)
    sumsq = torch.zeros(rollout_steps + 1, dtype=torch.float64)
    count = 0
    total_seconds = 0.0
    rollout_fn = rollout_fno if model_kind == "fno" else rollout_transformer

    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            gt = torch.stack(
                [dataset.get_trajectory(i)[: rollout_steps + 1] for i in range(start, stop)],
                dim=0,
            ).unsqueeze(1).to(device)
            x0 = gt[:, :, 0].contiguous()
            dt = torch.full((gt.shape[0],), float(dataset.dt), dtype=x0.dtype, device=device)
            _sync_device(device)
            t0 = time.perf_counter()
            if model_kind == "fno":
                pred = rollout_fn(model, x0, dt, rollout_steps)
            else:
                pred = rollout_fn(model, x0, dt, rollout_steps, use_group_average=use_group_average)
            _sync_device(device)
            total_seconds += time.perf_counter() - t0
            se = (pred - gt).pow(2).sum(dim=(1, 3, 4)).double()
            sumsq += se.sum(dim=0).cpu()
            count += gt.shape[0] * gt.shape[3] * gt.shape[4]
            tag = f"{model_kind}_ga" if use_group_average else model_kind
            print(f"{tag}: processed trajectories {start:03d}-{stop - 1:03d}")

    frames_predicted = n * rollout_steps
    timing = {
        "inference_total_seconds": float(total_seconds),
        "inference_ms_per_pred_frame": float(1000.0 * total_seconds / max(frames_predicted, 1)),
        "inference_ms_per_trajectory": float(1000.0 * total_seconds / max(n, 1)),
    }
    return torch.sqrt(sumsq / count), timing


def save_curve_values(csv_path: Path, curves: dict[str, torch.Tensor], rollout_steps: int) -> None:
    fieldnames = ["timestep", *curves.keys()]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for t in range(rollout_steps + 1):
            row = {"timestep": t}
            for name, curve in curves.items():
                row[name] = f"{curve[t].item():.10f}"
            writer.writerow(row)


def save_summary(
    json_path: Path,
    curves: dict[str, torch.Tensor],
    timings: dict[str, dict],
    rollout_steps: int,
) -> None:
    key_steps = [1, 10, 30, 50, 80, 100]
    summary = {}
    for name, curve in curves.items():
        metrics = {
            "mean_rmse_t1_to_tN": float(curve[1:].mean().item()),
            "final_step_rmse": float(curve[rollout_steps].item()),
        }
        for step in key_steps:
            if step <= rollout_steps:
                metrics[f"t{step}_rmse"] = float(curve[step].item())
        metrics.update(timings.get(name, {}))
        summary[name] = metrics
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_plot(
    plot_path: Path,
    curves: dict[str, torch.Tensor],
    rollout_steps: int,
    *,
    log_scale: bool = False,
    title: str = "Full-test rollout RMSE vs timestep",
) -> None:
    plt.figure(figsize=(8.5, 5.5))
    for name, curve in curves.items():
        plt.plot(range(rollout_steps + 1), curve.numpy(), label=name, linewidth=2)
    plt.xlabel("Rollout timestep")
    plt.ylabel("RMSE")
    plt.title(title)
    if log_scale:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = resolve_output_dir()
    device = pick_device()

    data_file = args.data_file or os.environ.get("PDEBENCH_FILE") or "./data/2D/shallow-water/2D_rdb_NA_NA.h5"
    reference_payload = torch.load(args.base48_checkpoint, map_location="cpu", weights_only=True)
    data_args = evaluation_data_args(reference_payload, args.split_seed, args.field_idx)
    del reference_payload
    if data_args["history"] != 1:
        raise ValueError("The full benchmark expects history=1; use the checkpoint evaluator for longer histories.")
    dataset = PDEBenchShallowWaterDataset(
        file_path=data_file,
        split=args.split,
        **data_args,
        normalize=False,
        augment=False,
    )

    specs = [
        ("base48", Path(args.base48_checkpoint), "transformer"),
        ("base48_ga", Path(args.base48_checkpoint), "transformer"),
        ("aug48", Path(args.aug48_checkpoint), "transformer"),
        ("reg48", Path(args.reg48_checkpoint), "transformer"),
        ("eq112", Path(args.eq112_checkpoint), "transformer"),
        ("fno", Path(args.fno_checkpoint), "fno"),
    ]
    if args.base48_long_checkpoint:
        specs.extend(
            [
                ("base48_long", Path(args.base48_long_checkpoint), "transformer"),
                ("base48_long_ga", Path(args.base48_long_checkpoint), "transformer"),
            ]
        )

    curves: dict[str, torch.Tensor] = {}
    timings: dict[str, dict] = {}
    for name, ckpt_path, kind in specs:
        print(f"Loading {name} from {ckpt_path}")
        if kind == "fno":
            model, payload = load_fno_checkpoint(ckpt_path)
        else:
            model, payload = load_transformer_checkpoint(ckpt_path)
        if evaluation_data_args(payload, args.split_seed, args.field_idx) != data_args:
            raise ValueError(f"{name}: checkpoint split, field, or history differs from the reference run.")
        if kind == "fno" and int(payload["model_kwargs"]["out_channels"]) != 1:
            raise ValueError("The full benchmark expects a one-step FNO checkpoint.")
        print(f"{name}: epoch={payload.get('epoch')} device={device}")
        curve, timing = compute_curve(
            model=model,
            dataset=dataset,
            device=device,
            rollout_steps=args.rollout_steps,
            batch_size=args.batch_size,
            model_kind=kind,
            use_group_average=(name in {"base48_ga", "base48_long_ga"}),
        )
        curves[name] = curve
        timings[name] = timing
        torch.save(curves[name], out_dir / f"{name}_rmse_curve.pt")

    csv_path = out_dir / f"{args.artifact_prefix}.csv"
    json_path = out_dir / f"{args.artifact_prefix}.json"
    plot_path = out_dir / f"{args.artifact_prefix}.png"
    log_plot_path = out_dir / f"{args.artifact_prefix}_log.png"
    transformer_plot_path = out_dir / f"{args.artifact_prefix}_transformers.png"

    save_curve_values(csv_path, curves, args.rollout_steps)
    save_summary(json_path, curves, timings, args.rollout_steps)
    split_label = args.split.replace("_", " ").title()
    save_plot(
        plot_path,
        curves,
        args.rollout_steps,
        title=f"{split_label} rollout RMSE vs timestep",
    )
    save_plot(
        log_plot_path,
        curves,
        args.rollout_steps,
        log_scale=True,
        title=f"{split_label} rollout RMSE vs timestep (log scale)",
    )
    save_plot(
        transformer_plot_path,
        {k: v for k, v in curves.items() if k != "fno"},
        args.rollout_steps,
        title=f"{split_label} rollout RMSE vs timestep (transformer models)",
    )

    print(f"Saved curve table to {csv_path}")
    print(f"Saved summary to {json_path}")
    print(f"Saved plot to {plot_path}")
    print(f"Saved log plot to {log_plot_path}")
    print(f"Saved transformer-only plot to {transformer_plot_path}")


if __name__ == "__main__":
    main()
