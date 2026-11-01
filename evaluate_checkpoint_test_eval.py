import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from compare_two_checkpoints_test_eval import compute_rollout_summary
from checkpoint_utils import evaluation_data_args, instantiate_from_checkpoint
from data_handling import PDEBenchShallowWaterDataset
from navierstokes_official import evaluate_d4_error, evaluate_one_step_mse, resolve_output_dir, resolve_shallow_water_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint on one-step and rollout RMSE.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--field-idx", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--artifact-prefix", type=str, default="checkpoint_test_eval")
    parser.add_argument("--equiv-eval-batches", type=int, default=8)
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    args = parse_args()
    out_dir = resolve_output_dir()
    device = pick_device()

    checkpoint = Path(args.checkpoint)
    model, payload = instantiate_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()

    data_file = args.data_file or resolve_shallow_water_file()
    dataset = PDEBenchShallowWaterDataset(
        file_path=data_file,
        split=args.split,
        **evaluation_data_args(payload, args.split_seed, args.field_idx),
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

    one_step_mse = evaluate_one_step_mse(model, loader, device)
    d4_error = evaluate_d4_error(model, loader, device, max_batches=args.equiv_eval_batches)

    summary = {
        "checkpoint": str(checkpoint),
        "epoch": int(payload.get("epoch", -1)),
        "one_step_mse": float(one_step_mse),
        "one_step_rmse": float(one_step_mse ** 0.5),
        "d4_error": float(d4_error),
    }
    summary.update(compute_rollout_summary(model, dataset, device, args.rollout_steps, args.batch_size))

    out_path = out_dir / f"{args.artifact_prefix}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
