import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from checkpoint_utils import evaluation_data_args, instantiate_from_checkpoint
from data_handling import PDEBenchShallowWaterDataset
from navierstokes_official import (
    evaluate_d4_error,
    evaluate_one_step_mse,
    resolve_output_dir,
    resolve_shallow_water_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint on one-step RMSE.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--field-idx", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--artifact-prefix", type=str, default="checkpoint_one_step_eval")
    parser.add_argument("--group-average-eval", action="store_true")
    parser.add_argument("--equiv-eval-batches", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    args = parse_args()
    out_dir = resolve_output_dir()
    device = torch.device("cpu")

    checkpoint = Path(args.checkpoint)
    model, payload = instantiate_from_checkpoint(checkpoint)
    model.to(device)

    dataset = PDEBenchShallowWaterDataset(
        file_path=args.data_file or resolve_shallow_water_file(),
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

    mse = evaluate_one_step_mse(model, loader, device, use_group_average=args.group_average_eval)
    rmse = float(mse ** 0.5)
    d4_error = evaluate_d4_error(
        model,
        loader,
        device,
        use_group_average=args.group_average_eval,
        max_batches=args.equiv_eval_batches,
    )

    summary = {
        "checkpoint": str(checkpoint),
        "epoch": int(payload.get("epoch", -1)),
        "split": args.split,
        "group_average_eval": bool(args.group_average_eval),
        "one_step_mse": float(mse),
        "one_step_rmse": rmse,
        "d4_error": float(d4_error),
    }

    suffix = "_gaeval" if args.group_average_eval else ""
    json_path = out_dir / f"{args.artifact_prefix}{suffix}.json"
    mse_path = out_dir / f"{args.artifact_prefix}_test_one_step_mse{suffix}.txt"
    rmse_path = out_dir / f"{args.artifact_prefix}_test_one_step_rmse{suffix}.txt"
    d4_path = out_dir / f"{args.artifact_prefix}_test_d4_error{suffix}.txt"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    mse_path.write_text(f"{mse:.10f}\n", encoding="utf-8")
    rmse_path.write_text(f"{rmse:.10f}\n", encoding="utf-8")
    d4_path.write_text(f"{d4_error:.10f}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()
