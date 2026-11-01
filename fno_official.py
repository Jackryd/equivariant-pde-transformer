import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from neuralop.models import FNO

from data_handling import build_pdebench_shallow_water_loader


def maybe_redirect_logs_to_cloud() -> None:
    log_path = os.environ.get("LOG_PATH")
    if not log_path:
        return
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_file, "a", encoding="utf-8")
    sys.stdout = log
    sys.stderr = log


def resolve_shallow_water_file() -> str:
    override = os.environ.get("PDEBENCH_FILE")
    if override:
        return override

    candidates = [
        Path("./data/2D/shallow-water/2D_rdb_NA_NA.h5"),
        Path("./data/pdebench/2D/shallow-water/2D_rdb_NA_NA.h5"),
        Path("./data/2D_rdb_NA_NA.h5"),
    ]

    bucket_uri = os.environ.get("BUCKET", "")
    if bucket_uri.startswith("gs://"):
        bucket_name = bucket_uri.split("gs://", 1)[1]
        candidates.extend(
            [
                Path("/gcs") / bucket_name / "2D/shallow-water/2D_rdb_NA_NA.h5",
                Path("/gcs") / bucket_name / "pdebench/2D/shallow-water/2D_rdb_NA_NA.h5",
                Path("/gcs") / bucket_name / "data/2D/shallow-water/2D_rdb_NA_NA.h5",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def resolve_output_dir() -> Path:
    override = os.environ.get("OUTPUT_DIR")
    if override:
        out_dir = Path(override)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    log_path = os.environ.get("LOG_PATH")
    if log_path:
        out_dir = Path(log_path).parent / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    out_dir = Path("./output")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the official neuraloperator FNO on shallow water.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--hidden_channels", type=int, default=93)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--scheduler_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--field_idx", type=int, default=0)
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--artifact_prefix", type=str, default="fno_official")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> FNO:
    model = FNO(
        n_modes=(args.modes, args.modes),
        in_channels=args.history,
        out_channels=args.horizon,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        positional_embedding="grid",
    )
    return model.to(device)


def prepare_batch(batch, device: torch.device):
    x, y, _dt = batch
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    x = x.contiguous()
    y = y.contiguous()
    return x, y


@torch.no_grad()
def evaluate_one_step_rmse(model: FNO, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    mse_total = 0.0
    n = 0
    for batch in loader:
        x, y = prepare_batch(batch, device)
        pred = model(x)
        mse = F.mse_loss(pred, y, reduction="sum")
        mse_total += float(mse.item())
        n += int(np.prod(y.shape))
    mean_mse = mse_total / max(n, 1)
    return mean_mse, math.sqrt(mean_mse)


def save_checkpoint(path: Path, model: FNO, optimizer, scheduler, epoch: int, args: argparse.Namespace) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "model_kwargs": {
            "n_modes": (args.modes, args.modes),
            "in_channels": args.history,
            "out_channels": args.horizon,
            "hidden_channels": args.hidden_channels,
            "n_layers": args.n_layers,
        },
        "args": vars(args),
    }
    torch.save(payload, path)


def run_smoke_test(args: argparse.Namespace, model: FNO, device: torch.device, train_loader, out_dir: Path) -> None:
    batch = next(iter(train_loader))
    x, y = prepare_batch(batch, device)
    pred = model(x)
    loss = F.mse_loss(pred, y)
    loss.backward()
    summary = {
        "device": str(device),
        "input_shape": list(x.shape),
        "target_shape": list(y.shape),
        "pred_shape": list(pred.shape),
        "loss": float(loss.item()),
        "param_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    out_path = out_dir / f"{args.artifact_prefix}_smoke_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    maybe_redirect_logs_to_cloud()
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = resolve_output_dir()
    file_path = resolve_shallow_water_file()

    print(f"Dataset file: {file_path}")
    print(f"Output dir: {out_dir}")
    print(f"Device: {device}")

    train_loader = build_pdebench_shallow_water_loader(
        file_path=file_path,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        history=args.history,
        horizon=args.horizon,
        field_idx=args.field_idx,
        split_seed=args.split_seed,
    )
    test_loader = build_pdebench_shallow_water_loader(
        file_path=file_path,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        history=args.history,
        horizon=args.horizon,
        field_idx=args.field_idx,
        split_seed=args.split_seed,
        shuffle=False,
    )

    model = build_model(args, device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model trainable params: {param_count}")

    if args.smoke_test:
        run_smoke_test(args, model, device, train_loader, out_dir)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.learning_rate * args.scheduler_min_lr_ratio,
    )

    latest_ckpt = out_dir / "latest_checkpoint.pt"
    final_ckpt = out_dir / f"{args.artifact_prefix}_checkpoint.pt"
    rmse_log = out_dir / f"{args.artifact_prefix}_test_one_step_rmse_by_epoch.txt"
    mse_log = out_dir / f"{args.artifact_prefix}_test_one_step_mse.txt"

    rmse_entries = []
    train_steps_total = math.ceil(len(train_loader) * max(min(args.limit_train_batches, 1.0), 0.0))
    print(f"Train batches per epoch (effective): {train_steps_total}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        max_batches = math.ceil(len(train_loader) * max(min(args.limit_train_batches, 1.0), 0.0))
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            x, y = prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            if args.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1
            if (batch_idx + 1) % 1000 == 0 or batch_idx == 0:
                print(
                    f"epoch={epoch} batch={batch_idx+1}/{max_batches} "
                    f"loss={loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.6e}"
                )

        mean_mse, rmse = evaluate_one_step_rmse(model, test_loader, device)
        rmse_entries.append((epoch, rmse))
        with rmse_log.open("w", encoding="utf-8") as fh:
            for ep, value in rmse_entries:
                fh.write(f"{ep}\t{value:.10f}\n")
        mse_log.write_text(f"{mean_mse:.10f}\n", encoding="utf-8")

        save_checkpoint(latest_ckpt, model, optimizer, scheduler, epoch, args)
        print(
            f"epoch={epoch} train_loss={running_loss / max(n_batches, 1):.6f} "
            f"test_rmse={rmse:.6f}"
        )
        scheduler.step()

    save_checkpoint(final_ckpt, model, optimizer, scheduler, args.epochs, args)
    save_checkpoint(latest_ckpt, model, optimizer, scheduler, args.epochs, args)
    print(f"Saved final checkpoint to {final_ckpt}")


if __name__ == "__main__":
    main()
