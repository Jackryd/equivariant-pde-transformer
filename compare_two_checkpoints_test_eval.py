import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from checkpoint_utils import common_evaluation_data_args, instantiate_from_checkpoint
from data_handling import PDEBenchShallowWaterDataset
from navierstokes_official import (
    evaluate_one_step_mse,
    resolve_output_dir,
    resolve_shallow_water_file,
    rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two checkpoints on one-step and rollout RMSE.")
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--eq-checkpoint", type=str, required=True)
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--field-idx", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--artifact-prefix", type=str, default="two_checkpoint_test_eval")
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_rollout_summary(
    model: torch.nn.Module,
    dataset: PDEBenchShallowWaterDataset,
    device: torch.device,
    rollout_steps: int,
    batch_size: int,
) -> dict:
    model.to(device)
    model.eval()
    n = len(dataset.trajectory_indices)
    sumsq = torch.zeros(rollout_steps + 1, dtype=torch.float64)
    count = 0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            gt = torch.stack(
                [dataset.get_trajectory(i)[: dataset.history + rollout_steps] for i in range(start, stop)],
                dim=0,
            ).to(device)
            if gt.shape[1] != dataset.history + rollout_steps:
                raise ValueError("Requested rollout exceeds the available trajectory length.")
            x0 = gt[:, : dataset.history].contiguous()
            gt = gt[:, dataset.history - 1 :].unsqueeze(1)
            dt = torch.full((gt.shape[0],), float(dataset.dt), dtype=x0.dtype, device=device)
            pred = rollout(model, x0, dt, n_steps=rollout_steps)
            se = (pred - gt).pow(2).sum(dim=(1, 3, 4)).double()
            sumsq += se.sum(dim=0).cpu()
            count += gt.shape[0] * gt.shape[3] * gt.shape[4]

    curve = torch.sqrt(sumsq / count)
    return {
        "mean_rmse_t1_to_tN": float(curve[1:].mean().item()),
        "final_step_rmse": float(curve[rollout_steps].item()),
        "t1_rmse": float(curve[1].item()),
        "t10_rmse": float(curve[min(10, rollout_steps)].item()),
        "t30_rmse": float(curve[min(30, rollout_steps)].item()),
        "t50_rmse": float(curve[min(50, rollout_steps)].item()),
        "t80_rmse": float(curve[min(80, rollout_steps)].item()),
    }


def main() -> None:
    args = parse_args()
    out_dir = resolve_output_dir()
    device = pick_device()

    baseline_model, baseline_payload = instantiate_from_checkpoint(Path(args.baseline_checkpoint))
    eq_model, eq_payload = instantiate_from_checkpoint(Path(args.eq_checkpoint))

    data_args = common_evaluation_data_args(
        [baseline_payload, eq_payload], args.split_seed, args.field_idx
    )
    data_file = args.data_file or resolve_shallow_water_file()
    dataset = PDEBenchShallowWaterDataset(
        file_path=data_file,
        split=args.split,
        **data_args,
        normalize=False,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )


    baseline_model.to(device)
    eq_model.to(device)

    baseline_one_step_mse = evaluate_one_step_mse(baseline_model, loader, device)
    eq_one_step_mse = evaluate_one_step_mse(eq_model, loader, device)

    summary = {
        "baseline": {
            "checkpoint": args.baseline_checkpoint,
            "epoch": int(baseline_payload.get("epoch", -1)),
            "one_step_mse": float(baseline_one_step_mse),
            "one_step_rmse": float(baseline_one_step_mse ** 0.5),
        },
        "eq": {
            "checkpoint": args.eq_checkpoint,
            "epoch": int(eq_payload.get("epoch", -1)),
            "one_step_mse": float(eq_one_step_mse),
            "one_step_rmse": float(eq_one_step_mse ** 0.5),
        },
    }
    summary["baseline"].update(
        compute_rollout_summary(baseline_model, dataset, device, args.rollout_steps, args.batch_size)
    )
    summary["eq"].update(
        compute_rollout_summary(eq_model, dataset, device, args.rollout_steps, args.batch_size)
    )

    out_path = out_dir / f"{args.artifact_prefix}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
