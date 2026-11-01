import argparse
import os
import sys
import time
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from PIL import Image

from data_handling import PDEBenchShallowWaterDataset, build_pdebench_shallow_water_loader


D4_ELEMENTS = [(k, bool(b)) for k in range(4) for b in range(2)]
DEFAULT_TINY_DEPTH = (1, 1, 2, 1, 1)


def _apply_d4_single(x: torch.Tensor, k: int, b: bool) -> torch.Tensor:
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    if b:
        x = torch.flip(x, dims=(-1,))
    return x.contiguous()


def _invert_d4_single(x: torch.Tensor, k: int, b: bool) -> torch.Tensor:
    # For the chosen transform order (rotate, then reflect), reflections are involutions.
    if b:
        return _apply_d4_single(x, k, b)
    return _apply_d4_single(x, (-k) % 4, False)


def d4_group_average_predict(model: nn.Module, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    preds = []
    for k, b in D4_ELEMENTS:
        x_g = _apply_d4_single(x, k, b)
        y_g = model(x_g, dt)
        y_back = _invert_d4_single(y_g, k, b)
        preds.append(y_back)
    return torch.stack(preds, dim=0).mean(dim=0)


def _advance_history_window(curr: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if curr.shape[1] <= 1:
        return pred
    return torch.cat([curr[:, 1:], pred], dim=1).contiguous()


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

    candidates: list[Path] = [
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


def _timestep_from_dt(dt, x):
    if isinstance(dt, torch.Tensor):
        dt = dt.to(device=x.device, dtype=x.dtype)
        if dt.ndim == 1:
            return dt
        return dt.view(dt.shape[0], -1)[:, 0]
    return torch.full((x.shape[0],), float(dt), dtype=x.dtype, device=x.device)


def _normalize_model_depth(depth) -> list[int]:
    if depth is None:
        return list(DEFAULT_TINY_DEPTH)
    if isinstance(depth, str):
        parsed = _parse_csv_ints(depth)
        return parsed or list(DEFAULT_TINY_DEPTH)
    return [int(v) for v in depth]


def _ensure_pde_model_registered(
    PDEImpl,
    PDE_models,
    *,
    hidden_size: int,
    num_heads: int,
    depth,
    mlp_ratio: float,
) -> str:
    depth_list = _normalize_model_depth(depth)
    mlp_ratio_value = float(mlp_ratio)
    is_default_tiny = (
        int(hidden_size) == 48
        and int(num_heads) == 4
        and tuple(depth_list) == DEFAULT_TINY_DEPTH
        and float(mlp_ratio_value) == 3.0
    )
    if is_default_tiny:
        model_name = "PDE-TINY"
    else:
        depth_tag = "-".join(str(v) for v in depth_list)
        mlp_tag = str(mlp_ratio_value).replace(".", "p")
        model_name = f"PDE-CUSTOM-h{int(hidden_size)}-nh{int(num_heads)}-d{depth_tag}-m{mlp_tag}"

    if model_name not in PDE_models:
        def _factory(**kwargs):
            return PDEImpl(
                down_factor=2,
                hidden_size=int(hidden_size),
                num_heads=int(num_heads),
                depth=depth_list,
                mlp_ratio=mlp_ratio_value,
                **kwargs,
            )

        PDE_models[model_name] = _factory

    return model_name


class MuonAdamW(Optimizer):
    """Use native torch.optim.Muon for 2D params and AdamW for remaining params."""

    def __init__(
        self,
        muon_opt: optim.Optimizer,
        adamw_opt: optim.Optimizer | None,
    ):
        params = []
        for group in muon_opt.param_groups:
            params.extend(group["params"])
        if adamw_opt is not None:
            for group in adamw_opt.param_groups:
                params.extend(group["params"])

        super().__init__(params, defaults={})
        self.muon_opt = muon_opt
        self.adamw_opt = adamw_opt
        merged_groups = list(muon_opt.param_groups)
        if adamw_opt is not None:
            merged_groups.extend(adamw_opt.param_groups)
        self.param_groups = merged_groups

    def zero_grad(self, set_to_none: bool = True):
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        if self.adamw_opt is not None:
            self.adamw_opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon_opt.step()
        if self.adamw_opt is not None:
            self.adamw_opt.step()
        return loss

    def state_dict(self):
        return {
            "muon": self.muon_opt.state_dict(),
            "adamw": None if self.adamw_opt is None else self.adamw_opt.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.muon_opt.load_state_dict(state_dict["muon"])
        if self.adamw_opt is not None and state_dict.get("adamw") is not None:
            self.adamw_opt.load_state_dict(state_dict["adamw"])


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    muon_beta: float,
    muon_ns_steps: int,
):
    name = str(optimizer_name).lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "muon":
        if not hasattr(optim, "Muon"):
            raise RuntimeError(
                "torch.optim.Muon is not available in this torch build. "
                f"Installed torch={torch.__version__}."
            )
        params = [p for p in model.parameters() if p.requires_grad]
        muon_params = [p for p in params if p.ndim == 2]
        aux_params = [p for p in params if p.ndim != 2]
        if not muon_params:
            raise RuntimeError("No 2D parameters found for torch.optim.Muon.")

        muon_opt = optim.Muon(
            muon_params,
            lr=lr,
            momentum=muon_beta,
            weight_decay=weight_decay,
            ns_steps=muon_ns_steps,
        )
        adamw_opt = (
            optim.AdamW(aux_params, lr=lr, weight_decay=weight_decay)
            if aux_params
            else None
        )
        return MuonAdamW(muon_opt=muon_opt, adamw_opt=adamw_opt)
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")


def build_scheduler(
    optimizer: Optimizer,
    scheduler_name: str,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
    total_steps: int = 0,
    warmup_steps: int = 0,
):
    name = str(scheduler_name).lower()
    total_epochs = max(1, int(total_epochs))
    warmup_epochs = max(0, int(warmup_epochs))
    min_lr_ratio = float(min_lr_ratio)
    total_steps = max(0, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))

    if name == "none":
        return None, "epoch"

    eta_min = min(group["lr"] for group in optimizer.param_groups) * min_lr_ratio

    if name == "cosine":
        if total_steps > 0:
            return (
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                    eta_min=eta_min,
                ),
                "step",
            )
        return (
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=eta_min,
            ),
            "epoch",
        )

    if name == "warmup_cosine":
        if warmup_epochs <= 0 and warmup_steps <= 0:
            raise ValueError("warmup_cosine requires warmup_epochs > 0")
        if total_steps > 0:
            if warmup_steps <= 0:
                raise ValueError("warmup_cosine step schedule requires warmup_steps > 0")
            if warmup_steps >= total_steps:
                raise ValueError("warmup_cosine requires warmup_steps < total_steps")
            warmup = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=eta_min,
            )
            return (
                optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_steps],
                ),
                "step",
            )
        if warmup_epochs >= total_epochs:
            raise ValueError("warmup_cosine requires warmup_epochs < total_epochs")
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=eta_min,
        )
        return (
            optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            ),
            "epoch",
        )

    raise ValueError(f"Unsupported scheduler '{scheduler_name}'.")


class OfficialTinyPDEModel(nn.Module):
    def __init__(
        self,
        sample_size: int = 128,
        in_channels: int = 1,
        out_channels: int = 1,
        patch_size: int = 4,
        window_size: int = 8,
        hidden_size: int = 48,
        num_heads: int = 4,
        depth=None,
        mlp_ratio: float = 3.0,
    ):
        super().__init__()
        try:
            from pdetransformer.core.mixed_channels.pde_transformer import (
                PDEImpl,
                PDETransformer,
                PDE_models,
            )
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Could not import pde-transformer package internals.\n"
                "Install with: python3 -m pip install -U pdetransformer peft\n"
                f"Original import error: {e!r}"
            ) from e

        model_type = _ensure_pde_model_registered(
            PDEImpl,
            PDE_models,
            hidden_size=hidden_size,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
        )

        self.model = PDETransformer(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            type=model_type,
            periodic=True,
            carrier_token_active=False,
            window_size=window_size,
            patch_size=patch_size,
        )

    def forward(self, x, dt):
        timestep = _timestep_from_dt(dt, x)
        out = self.model(hidden_states=x, timestep=timestep, return_dict=True)
        return out.sample


class OfficialTinyEQPDEModel(nn.Module):
    def __init__(
        self,
        sample_size: int = 128,
        in_channels: int = 1,
        out_channels: int = 1,
        patch_size: int = 4,
        window_size: int = 8,
        group_size: int = 8,
        eq_timestep_embedders: bool = True,
        eq_patch_embed: bool = True,
        eq_resample: bool = True,
        eq_output_path: bool = True,
        eq_relbias_symmetrize: bool = True,
        eq_token_linear_shared: bool = False,
        eq_attention_shared_qkv: bool = False,
        hidden_size: int = 48,
        num_heads: int = 4,
        depth=None,
        mlp_ratio: float = 3.0,
    ):
        super().__init__()
        if int(group_size) != 8:
            raise ValueError(
                "OfficialTinyEQPDEModel currently supports only D4 regular action with group_size=8. "
                f"Got group_size={group_size}."
            )
        try:
            from pdetransformer.core.mixed_channels.pde_transformer import PDEImpl, PDE_models
            from pde_transformer_eq_local import EQPDETransformer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Could not import EQ PDE-Transformer dependencies.\n"
                "Install with: python3 -m pip install -U pdetransformer peft\n"
                f"Original import error: {e!r}"
            ) from e

        model_type = _ensure_pde_model_registered(
            PDEImpl,
            PDE_models,
            hidden_size=hidden_size,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
        )

        self.model = EQPDETransformer(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            type=model_type,
            periodic=True,
            carrier_token_active=False,
            window_size=window_size,
            patch_size=patch_size,
            group_size=group_size,
            eq_timestep_embedders=eq_timestep_embedders,
            eq_patch_embed=eq_patch_embed,
            eq_resample=eq_resample,
            eq_output_path=eq_output_path,
            eq_relbias_symmetrize=eq_relbias_symmetrize,
            eq_token_linear_shared=eq_token_linear_shared,
            eq_attention_shared_qkv=eq_attention_shared_qkv,
        )

    def forward(self, x, dt):
        timestep = _timestep_from_dt(dt, x)
        out = self.model(hidden_states=x, timestep=timestep, return_dict=True)
        return out.sample


class OfficialTinyPaperEQPDEModel(nn.Module):
    def __init__(
        self,
        sample_size: int = 128,
        in_channels: int = 1,
        out_channels: int = 1,
        patch_size: int = 4,
        window_size: int = 8,
        group_size: int = 8,
        eq_timestep_embedders: bool = True,
        eq_patch_embed: bool = True,
        eq_resample: bool = True,
        eq_output_path: bool = True,
        eq_relbias_symmetrize: bool = True,
        hidden_size: int = 48,
        num_heads: int = 4,
        depth=None,
        mlp_ratio: float = 3.0,
    ):
        super().__init__()
        if int(group_size) != 8:
            raise ValueError(
                "OfficialTinyPaperEQPDEModel currently supports only D4 regular action with group_size=8. "
                f"Got group_size={group_size}."
            )
        try:
            from pdetransformer.core.mixed_channels.pde_transformer import PDEImpl, PDE_models
            from pde_transformer_eq_paper import PaperEQPDETransformer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Could not import paper-faithful EQ PDE-Transformer dependencies.\n"
                "Install with: python3 -m pip install -U pdetransformer peft\n"
                f"Original import error: {e!r}"
            ) from e

        model_type = _ensure_pde_model_registered(
            PDEImpl,
            PDE_models,
            hidden_size=hidden_size,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
        )

        self.model = PaperEQPDETransformer(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            type=model_type,
            periodic=True,
            carrier_token_active=False,
            window_size=window_size,
            patch_size=patch_size,
            group_size=group_size,
            eq_timestep_embedders=eq_timestep_embedders,
            eq_patch_embed=eq_patch_embed,
            eq_resample=eq_resample,
            eq_output_path=eq_output_path,
            eq_relbias_symmetrize=eq_relbias_symmetrize,
        )

    def forward(self, x, dt):
        timestep = _timestep_from_dt(dt, x)
        out = self.model(hidden_states=x, timestep=timestep, return_dict=True)
        return out.sample


class PLPDEOfficialModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        regularization: float = 0.0,
        use_group_average_train: bool = False,
        use_group_average_eval: bool = False,
        train_rollout_steps: int = 1,
        train_rollout_mode: str = "bptt",
        pushforward_first_step_weight: float = 0.0,
        pushforward_last_step_weight: float = 1.0,
        rollout_loss_decay: float = 1.0,
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        muon_beta: float = 0.95,
        muon_ns_steps: int = 5,
        scheduler_name: str = "none",
        scheduler_total_epochs: int = 0,
        scheduler_warmup_epochs: int = 0,
        scheduler_min_lr_ratio: float = 0.1,
        scheduler_total_steps: int = 0,
        scheduler_warmup_steps: int = 0,
    ):
        super().__init__()
        self.model = model
        self.lr = float(lr)
        self.criterion = nn.MSELoss()
        self.train_losses: list[float] = []
        self.regularization = float(regularization)
        self.use_group_average_train = bool(use_group_average_train)
        self.use_group_average_eval = bool(use_group_average_eval)
        self.train_rollout_steps = max(1, int(train_rollout_steps))
        self.train_rollout_mode = str(train_rollout_mode).lower()
        if self.train_rollout_mode not in {"bptt", "pushforward"}:
            raise ValueError(
                f"Unsupported train_rollout_mode={train_rollout_mode!r}. "
                "Expected one of {'bptt', 'pushforward'}."
            )
        self.rollout_loss_decay = float(rollout_loss_decay)
        self.pushforward_first_step_weight = float(pushforward_first_step_weight)
        self.pushforward_last_step_weight = float(pushforward_last_step_weight)
        self.optimizer_name = str(optimizer_name).lower()
        self.weight_decay = float(weight_decay)
        self.muon_beta = float(muon_beta)
        self.muon_ns_steps = int(muon_ns_steps)
        self.scheduler_name = str(scheduler_name).lower()
        self.scheduler_total_epochs = int(scheduler_total_epochs)
        self.scheduler_warmup_epochs = int(scheduler_warmup_epochs)
        self.scheduler_min_lr_ratio = float(scheduler_min_lr_ratio)
        self.scheduler_total_steps = int(scheduler_total_steps)
        self.scheduler_warmup_steps = int(scheduler_warmup_steps)

    def forward(self, x, dt):
        return self.model(x, dt)

    def _predict(self, x, dt, use_group_average: bool):
        if use_group_average:
            return d4_group_average_predict(self.model, x, dt)
        return self(x, dt)

    def training_step(self, batch, batch_idx):
        x_hist, y_future, dt = batch
        x_t = x_hist.contiguous()
        rollout_steps = min(self.train_rollout_steps, y_future.shape[1])
        curr = x_t
        pred_first = None
        if self.train_rollout_mode == "pushforward" and rollout_steps > 1:
            # Hybrid pushforward: anchor the next-step update while also supervising
            # a later detached-unrolled step to encourage rollout stability.
            loss = 0.0
            denom = 0.0

            pred = self._predict(curr, dt, self.use_group_average_train)
            pred_first = pred
            if self.pushforward_first_step_weight > 0.0:
                target_first = y_future[:, :1].contiguous()
                loss = loss + self.pushforward_first_step_weight * self.criterion(pred, target_first)
                denom += self.pushforward_first_step_weight
            curr = _advance_history_window(curr, pred.detach())

            for _ in range(1, rollout_steps - 1):
                with torch.no_grad():
                    pred_mid = self._predict(curr, dt, self.use_group_average_train)
                curr = _advance_history_window(curr, pred_mid)

            pred_last = self._predict(curr, dt, self.use_group_average_train)
            if self.pushforward_last_step_weight > 0.0:
                target_last = y_future[:, rollout_steps - 1 : rollout_steps].contiguous()
                loss = loss + self.pushforward_last_step_weight * self.criterion(pred_last, target_last)
                denom += self.pushforward_last_step_weight

            if denom <= 0.0:
                raise ValueError("Pushforward loss weights must sum to a positive value.")
            base_loss = loss / denom
        else:
            loss = 0.0
            denom = 0.0
            for step in range(rollout_steps):
                pred = self._predict(curr, dt, self.use_group_average_train)
                if pred_first is None:
                    pred_first = pred
                target = y_future[:, step:step + 1].contiguous()
                step_w = self.rollout_loss_decay ** step
                loss = loss + step_w * self.criterion(pred, target)
                denom += step_w
                curr = _advance_history_window(curr, pred)
            base_loss = loss / max(denom, 1e-8)
        loss = base_loss
        self.log("train_base_loss", base_loss, on_step=True, on_epoch=True, batch_size=x_t.shape[0])
        if self.regularization:
            k = int(torch.randint(0, 4, (1,)).item())
            b = bool(torch.randint(0, 2, (1,)).item())
            transformed_x = _apply_d4_single(x_t, k, b)
            transformed_pred = _apply_d4_single(pred_first, k, b)
            pred2 = self._predict(transformed_x, dt, self.use_group_average_train)
            eq_loss = F.mse_loss(pred2, transformed_pred)
            self.log("train_eq_loss", eq_loss, on_step=True, on_epoch=True, batch_size=x_t.shape[0])
            loss = loss + self.regularization * eq_loss
        self.train_losses.append(float(loss.detach().cpu()))
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x_t.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x_hist, y_future, dt = batch
        x_t = x_hist.contiguous()
        y_t_dt = y_future[:, :1].contiguous()
        pred = self._predict(x_t, dt, self.use_group_average_eval)
        loss = self.criterion(pred, y_t_dt)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x_t.shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = build_optimizer(
            self.model,
            optimizer_name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
            muon_beta=self.muon_beta,
            muon_ns_steps=self.muon_ns_steps,
        )
        scheduler, interval = build_scheduler(
            optimizer,
            scheduler_name=self.scheduler_name,
            total_epochs=self.scheduler_total_epochs,
            warmup_epochs=self.scheduler_warmup_epochs,
            min_lr_ratio=self.scheduler_min_lr_ratio,
            total_steps=self.scheduler_total_steps,
            warmup_steps=self.scheduler_warmup_steps,
        )
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": 1,
            },
        }


class EpochProgressCheckpoint(Callback):
    """Save resumable checkpoints each epoch for quick sanity/early-stop workflows."""

    def __init__(
        self,
        latest_path: Path,
        mirror_path: Path,
        args_dict: dict,
        data_file: str,
        epoch_offset: int = 0,
    ):
        super().__init__()
        self.latest_path = Path(latest_path)
        self.mirror_path = Path(mirror_path)
        self.args_dict = dict(args_dict)
        self.data_file = str(data_file)
        self.epoch_offset = int(epoch_offset)

    def _payload(self, trainer: Trainer, pl_module: L.LightningModule, epoch: int) -> dict:
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        optim_state = (
            trainer.optimizers[0].state_dict()
            if getattr(trainer, "optimizers", None)
            else None
        )
        scheduler_state = None
        if getattr(trainer, "lr_scheduler_configs", None):
            scheduler_state = trainer.lr_scheduler_configs[0].scheduler.state_dict()
        train_losses = list(getattr(pl_module, "train_losses", []))
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim_state,
            "scheduler_state_dict": scheduler_state,
            "train_losses": train_losses,
            "args": self.args_dict,
            "data_file": self.data_file,
            "epoch": int(epoch),
            "global_step": int(getattr(trainer, "global_step", 0)),
        }

    def on_train_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        epoch_num = self.epoch_offset + int(trainer.current_epoch) + 1
        payload = self._payload(trainer, pl_module, epoch_num)
        torch.save(payload, self.latest_path)
        torch.save(payload, self.mirror_path)
        print(
            "Saved epoch checkpoint:",
            f"epoch={epoch_num}",
            f"latest={self.latest_path}",
        )


class EpochTestRMSELogger(Callback):
    """Compute and append one-step test RMSE once per epoch."""

    def __init__(
        self,
        test_loader,
        out_path: Path,
        use_group_average_eval: bool = False,
        epoch_offset: int = 0,
    ):
        super().__init__()
        self.test_loader = test_loader
        self.out_path = Path(out_path)
        self.use_group_average_eval = bool(use_group_average_eval)
        self.epoch_offset = int(epoch_offset)
        self._header_written = self.out_path.exists() and self.out_path.stat().st_size > 0

    def _append(self, epoch: int, rmse: float) -> None:
        if not self._header_written:
            with self.out_path.open("w", encoding="utf-8") as f:
                f.write("epoch\ttest_one_step_rmse\n")
            self._header_written = True
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(f"{int(epoch)}\t{float(rmse):.10f}\n")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        device = next(model.parameters()).device
        mse = evaluate_one_step_mse(
            model,
            self.test_loader,
            device,
            use_group_average=self.use_group_average_eval,
        )
        rmse = float(mse**0.5)
        epoch_num = self.epoch_offset + int(trainer.current_epoch) + 1
        self._append(epoch_num, rmse)
        print(f"Epoch {epoch_num}: test one-step RMSE={rmse:.8f} (saved to {self.out_path})")


class EpochTimingLogger(Callback):
    """Log per-epoch wallclock time and peak GPU memory."""

    def __init__(self, out_path: Path, epoch_offset: int = 0):
        super().__init__()
        self.out_path = Path(out_path)
        self.epoch_offset = int(epoch_offset)
        self._header_written = self.out_path.exists() and self.out_path.stat().st_size > 0
        self._epoch_start_time = None

    def _append(self, epoch: int, seconds: float, peak_mem_mb: float) -> None:
        if not self._header_written:
            with self.out_path.open("w", encoding="utf-8") as f:
                f.write("epoch\tseconds\tpeak_gpu_mem_mb\n")
            self._header_written = True
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(f"{int(epoch)}\t{float(seconds):.6f}\t{float(peak_mem_mb):.2f}\n")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        self._epoch_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        if self._epoch_start_time is None:
            return
        seconds = time.perf_counter() - self._epoch_start_time
        peak_mem_mb = 0.0
        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        epoch_num = self.epoch_offset + int(trainer.current_epoch) + 1
        self._append(epoch_num, seconds, peak_mem_mb)
        print(
            f"Epoch {epoch_num}: wallclock={seconds:.2f}s peak_gpu_mem_mb={peak_mem_mb:.2f} "
            f"(saved to {self.out_path})"
        )


class RestoreTrainingStateCallback(Callback):
    """Restore optimizer and scheduler state into Lightning-managed objects."""

    def __init__(self, optimizer_state_dict=None, scheduler_state_dict=None):
        super().__init__()
        self.optimizer_state_dict = optimizer_state_dict
        self.scheduler_state_dict = scheduler_state_dict
        self._restored = False

    def on_fit_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        if self._restored:
            return
        if self.optimizer_state_dict is not None and getattr(trainer, "optimizers", None):
            trainer.optimizers[0].load_state_dict(self.optimizer_state_dict)
            print("Restored optimizer state into Lightning optimizer.")
        if (
            self.scheduler_state_dict is not None
            and getattr(trainer, "lr_scheduler_configs", None)
        ):
            trainer.lr_scheduler_configs[0].scheduler.load_state_dict(self.scheduler_state_dict)
            print("Restored scheduler state into Lightning scheduler.")
        self._restored = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def disable_adaln_conditioning(model: nn.Module) -> int:
    """Zero and freeze AdaLN modulation layers for a Swin-like ablation."""
    def _zero_and_freeze_module_params(layer: nn.Module) -> int:
        frozen_local = 0
        with torch.no_grad():
            for p in layer.parameters(recurse=False):
                p.zero_()
                p.requires_grad_(False)
                frozen_local += int(p.numel())
        return frozen_local

    frozen = 0
    with torch.no_grad():
        for module in model.modules():
            for attr_name in ("adain_1", "adain_2"):
                adain = getattr(module, attr_name, None)
                if adain is None or not hasattr(adain, "linear"):
                    continue
                frozen += _zero_and_freeze_module_params(adain.linear)

            if hasattr(module, "adaLN_modulation") and isinstance(module.adaLN_modulation, nn.Sequential):
                if len(module.adaLN_modulation) > 0:
                    frozen += _zero_and_freeze_module_params(module.adaLN_modulation[-1])
    return frozen


@torch.no_grad()
def evaluate_one_step_mse(
    model: nn.Module,
    loader,
    device: torch.device,
    use_group_average: bool = False,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for x_hist, y_future, dt in loader:
        x_t = x_hist.to(device)
        y_t_dt = y_future[:, :1].to(device)
        dt = dt.to(device)
        if use_group_average:
            pred = d4_group_average_predict(model, x_t, dt)
        else:
            pred = model(x_t, dt)
        mse_per_sample = torch.mean((pred - y_t_dt) ** 2, dim=(1, 2, 3))
        total += float(mse_per_sample.sum().item())
        count += int(mse_per_sample.numel())
    return total / max(1, count)


@torch.no_grad()
def evaluate_d4_error(
    model: nn.Module,
    loader,
    device: torch.device,
    use_group_average: bool = False,
    max_batches: int = 8,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch_idx, (x_hist, _y_future, dt) in enumerate(loader):
        if batch_idx >= int(max_batches):
            break
        x_t = x_hist.to(device)
        dt = dt.to(device)
        if use_group_average:
            pred = d4_group_average_predict(model, x_t, dt)
        else:
            pred = model(x_t, dt)
        for k, b in D4_ELEMENTS:
            x_g = _apply_d4_single(x_t, k, b)
            if use_group_average:
                pred_g = d4_group_average_predict(model, x_g, dt)
            else:
                pred_g = model(x_g, dt)
            target = _apply_d4_single(pred, k, b)
            err = torch.mean(torch.abs(pred_g - target), dim=(1, 2, 3))
            total += float(err.sum().item())
            count += int(err.numel())
    return total / max(1, count)


@torch.no_grad()
def rollout(
    model: nn.Module,
    x0: torch.Tensor,
    dt: float | torch.Tensor,
    n_steps: int,
    use_group_average: bool = False,
) -> torch.Tensor:
    model.eval()
    curr = x0.contiguous()
    states = [curr[:, -1:].contiguous()]
    for _ in range(int(n_steps)):
        if use_group_average:
            x_next = d4_group_average_predict(model, curr, dt)
        else:
            x_next = model(curr, dt)
        states.append(x_next)
        curr = _advance_history_window(curr, x_next)
    return torch.stack(states, dim=2)  # [B, 1, T, H, W]


def save_rollout_gif(
    trajectory: torch.Tensor,
    out_path: Path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    frames = trajectory[0, 0]  # [T,H,W]
    if vmin is None:
        vmin = float(frames.min())
    if vmax is None:
        vmax = float(frames.max())
    denom = max(float(vmax - vmin), 1e-8)
    frames_u8 = ((frames - float(vmin)) / denom * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    images = [Image.fromarray(frame, mode="L") for frame in frames_u8]
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=120, loop=0)


def _parse_csv_ints(value: str) -> list[int]:
    items = [part.strip() for part in str(value).split(",")]
    out: list[int] = []
    for item in items:
        if not item:
            continue
        out.append(int(item))
    return out


def estimate_train_batches(train_loader, limit_train_batches: float) -> int:
    total = len(train_loader)
    value = limit_train_batches
    if isinstance(value, float) and value <= 1.0:
        return max(1, int(total * value))
    return max(1, int(value))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "muon"])
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine", "warmup_cosine"],
    )
    parser.add_argument("--scheduler_warmup_epochs", type=int, default=0)
    parser.add_argument("--scheduler_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--muon_beta", type=float, default=0.95)
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--max_start_step",
        type=int,
        default=-1,
        help="Restrict window starts to [0..max_start_step]. Use 9 for first 10 timesteps. -1 disables.",
    )
    parser.add_argument("--field_idx", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--rollout_steps", type=int, default=50)
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Optional path to a checkpoint (.pt) to continue training from.",
    )
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--equiv_reg_lambda", type=float, default=0.0)
    parser.add_argument("--group_average_train", action="store_true")
    parser.add_argument("--group_average_eval", action="store_true")
    parser.add_argument(
        "--group_average",
        action="store_true",
        help="Enable D4 group averaging for both training and evaluation.",
    )
    parser.add_argument("--artifact_prefix", type=str, default="official_tiny")
    parser.add_argument(
        "--model_variant",
        type=str,
        default="tiny",
        choices=["tiny", "tiny_eq", "tiny_eq_paper"],
    )
    parser.add_argument("--eq_group_size", type=int, default=8)
    parser.add_argument("--eq_no_timestep_embedders", action="store_true")
    parser.add_argument("--eq_no_patch_embed", action="store_true")
    parser.add_argument("--eq_no_resample", action="store_true")
    parser.add_argument("--eq_no_output_path", action="store_true")
    parser.add_argument("--eq_no_relbias_symmetrize", action="store_true")
    parser.add_argument(
        "--eq_token_linear_shared",
        action="store_true",
        help="Enable shared-fiber token MLP conversion (capacity-reducing; off by default).",
    )
    parser.add_argument(
        "--eq_attention_shared_qkv",
        action="store_true",
        help="Enable shared-fiber QKV projection in EQ attention (capacity-reducing; off by default).",
    )
    parser.add_argument(
        "--disable_adaln",
        action="store_true",
        help="Zero/freeze AdaLN modulation layers to approximate a Swin-style no-conditioning ablation.",
    )
    parser.add_argument("--save_loss_pdf", action="store_true")
    parser.add_argument(
        "--train_rollout_steps",
        type=int,
        default=1,
        help="Autoregressive steps used in training loss (requires horizon >= this value).",
    )
    parser.add_argument(
        "--train_rollout_mode",
        type=str,
        default="bptt",
        choices=["bptt", "pushforward"],
        help="Use full backprop through unrolled steps or PDEBench-style pushforward training.",
    )
    parser.add_argument(
        "--pushforward_first_step_weight",
        type=float,
        default=0.0,
        help="Weight on the t+1 loss term when using pushforward training.",
    )
    parser.add_argument(
        "--pushforward_last_step_weight",
        type=float,
        default=1.0,
        help="Weight on the final unrolled-step loss term when using pushforward training.",
    )
    parser.add_argument(
        "--rollout_loss_decay",
        type=float,
        default=1.0,
        help="Geometric decay for multi-step training loss weights.",
    )
    parser.add_argument(
        "--equiv_eval_batches",
        type=int,
        default=8,
        help="Number of test batches for D4-equivariance error evaluation.",
    )
    parser.add_argument(
        "--rollout_split_indices",
        type=str,
        default="0,1,2",
        help="Comma-separated test split trajectory indices for GIF export.",
    )
    parser.add_argument("--model_hidden_size", type=int, default=48)
    parser.add_argument("--model_num_heads", type=int, default=4)
    parser.add_argument("--model_depth", type=str, default="1,1,2,1,1")
    parser.add_argument("--model_mlp_ratio", type=float, default=3.0)
    parser.add_argument(
        "--no_track_test_rmse",
        action="store_true",
        help="Disable per-epoch test one-step RMSE logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    maybe_redirect_logs_to_cloud()
    args = parse_args()
    L.seed_everything(int(args.seed), workers=True)
    if args.group_average:
        args.group_average_train = True
        args.group_average_eval = True

    data_file = resolve_shallow_water_file()
    if not Path(data_file).exists():
        raise FileNotFoundError(
            "Could not find shallow-water dataset file.\n"
            f"Tried: {data_file}\n"
            "Upload file to bucket and/or set PDEBENCH_FILE explicitly."
        )

    out_dir = resolve_output_dir()
    checkpoint_path = out_dir / f"{args.artifact_prefix}_{args.model_variant}_pde_model_optim_checkpoint.pt"
    latest_checkpoint_path = out_dir / "latest_checkpoint.pt"
    model_depth = _normalize_model_depth(args.model_depth)

    data_horizon = max(int(args.horizon), int(args.train_rollout_steps))
    max_start_step = None if int(args.max_start_step) < 0 else int(args.max_start_step)

    train_loader = build_pdebench_shallow_water_loader(
        file_path=data_file,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        history=args.history,
        horizon=data_horizon,
        max_start_step=max_start_step,
        field_idx=args.field_idx,
        split_seed=args.split_seed,
        augment=args.augment,
    )
    val_loader = build_pdebench_shallow_water_loader(
        file_path=data_file,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        history=args.history,
        horizon=data_horizon,
        max_start_step=max_start_step,
        field_idx=args.field_idx,
        split_seed=args.split_seed,
        shuffle=False,
    )
    test_loader = build_pdebench_shallow_water_loader(
        file_path=data_file,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        history=args.history,
        horizon=data_horizon,
        max_start_step=max_start_step,
        field_idx=args.field_idx,
        split_seed=args.split_seed,
        shuffle=False,
    )
    effective_train_batches = estimate_train_batches(train_loader, args.limit_train_batches)
    scheduler_total_steps = int(args.epochs) * effective_train_batches
    scheduler_warmup_steps = int(args.scheduler_warmup_epochs) * effective_train_batches

    if args.model_variant == "tiny_eq":
        model = OfficialTinyEQPDEModel(
            sample_size=128,
            in_channels=args.history,
            out_channels=1,
            patch_size=4,
            window_size=8,
            group_size=args.eq_group_size,
            eq_timestep_embedders=not args.eq_no_timestep_embedders,
            eq_patch_embed=not args.eq_no_patch_embed,
            eq_resample=not args.eq_no_resample,
            eq_output_path=not args.eq_no_output_path,
            eq_relbias_symmetrize=not args.eq_no_relbias_symmetrize,
            eq_token_linear_shared=args.eq_token_linear_shared,
            eq_attention_shared_qkv=args.eq_attention_shared_qkv,
            hidden_size=args.model_hidden_size,
            num_heads=args.model_num_heads,
            depth=model_depth,
            mlp_ratio=args.model_mlp_ratio,
        )
    elif args.model_variant == "tiny_eq_paper":
        model = OfficialTinyPaperEQPDEModel(
            sample_size=128,
            in_channels=args.history,
            out_channels=1,
            patch_size=4,
            window_size=8,
            group_size=args.eq_group_size,
            eq_timestep_embedders=not args.eq_no_timestep_embedders,
            eq_patch_embed=not args.eq_no_patch_embed,
            eq_resample=not args.eq_no_resample,
            eq_output_path=not args.eq_no_output_path,
            eq_relbias_symmetrize=not args.eq_no_relbias_symmetrize,
            hidden_size=args.model_hidden_size,
            num_heads=args.model_num_heads,
            depth=model_depth,
            mlp_ratio=args.model_mlp_ratio,
        )
    else:
        model = OfficialTinyPDEModel(
            sample_size=128,
            in_channels=args.history,
            out_channels=1,
            patch_size=4,
            window_size=8,
            hidden_size=args.model_hidden_size,
            num_heads=args.model_num_heads,
            depth=model_depth,
            mlp_ratio=args.model_mlp_ratio,
        )
    if args.disable_adaln:
        frozen = disable_adaln_conditioning(model)
        print(f"Disabled AdaLN conditioning; froze {frozen:,} AdaLN parameters.")
    pl_model = PLPDEOfficialModule(
        model,
        lr=args.learning_rate,
        regularization=args.equiv_reg_lambda,
        use_group_average_train=args.group_average_train,
        use_group_average_eval=args.group_average_eval,
        train_rollout_steps=args.train_rollout_steps,
        train_rollout_mode=args.train_rollout_mode,
        pushforward_first_step_weight=args.pushforward_first_step_weight,
        pushforward_last_step_weight=args.pushforward_last_step_weight,
        rollout_loss_decay=args.rollout_loss_decay,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        muon_beta=args.muon_beta,
        muon_ns_steps=args.muon_ns_steps,
        scheduler_name=args.scheduler,
        scheduler_total_epochs=args.epochs,
        scheduler_warmup_epochs=args.scheduler_warmup_epochs,
        scheduler_min_lr_ratio=args.scheduler_min_lr_ratio,
        scheduler_total_steps=scheduler_total_steps,
        scheduler_warmup_steps=scheduler_warmup_steps,
    )

    print(f"Dataset file: {data_file}")
    print(f"Output dir: {out_dir}")
    print(f"Seed: {args.seed}")
    print(f"Model variant: {args.model_variant}")
    print(
        "Backbone config:",
        f"hidden_size={args.model_hidden_size}",
        f"num_heads={args.model_num_heads}",
        f"depth={model_depth}",
        f"mlp_ratio={args.model_mlp_ratio}",
    )
    print(f"Model trainable params: {count_trainable_params(model):,}")
    print(
        f"Optimizer: {args.optimizer} (lr={args.learning_rate}, weight_decay={args.weight_decay}, "
        f"muon_beta={args.muon_beta}, muon_ns_steps={args.muon_ns_steps})"
    )
    print(f"Gradient clip val: {args.gradient_clip_val}")
    print(
        f"Scheduler: {args.scheduler} "
        f"(warmup_epochs={args.scheduler_warmup_epochs}, min_lr_ratio={args.scheduler_min_lr_ratio})"
    )
    print(
        f"Estimated train batches/epoch: {effective_train_batches}, "
        f"scheduler_total_steps={scheduler_total_steps}, "
        f"scheduler_warmup_steps={scheduler_warmup_steps}"
    )
    print(f"Equiv reg lambda: {args.equiv_reg_lambda}")
    print(f"Group averaging (train/eval): {args.group_average_train}/{args.group_average_eval}")
    print(
        f"Train rollout: steps={args.train_rollout_steps} mode={args.train_rollout_mode} "
        f"(data horizon={data_horizon}), decay={args.rollout_loss_decay}"
    )
    if args.train_rollout_mode == "pushforward":
        print(
            f"Pushforward weights: first={args.pushforward_first_step_weight}, "
            f"last={args.pushforward_last_step_weight}"
        )
    print(f"Max start step: {max_start_step}")
    if args.model_variant in {"tiny_eq", "tiny_eq_paper"}:
        eq_msg = (
            "EQ toggles: "
            f"timestep_embedders={not args.eq_no_timestep_embedders}, "
            f"patch_embed={not args.eq_no_patch_embed}, "
            f"resample={not args.eq_no_resample}, "
            f"output_path={not args.eq_no_output_path}, "
            f"relbias_symmetrize={not args.eq_no_relbias_symmetrize}"
        )
        if args.model_variant == "tiny_eq":
            eq_msg += (
                f", token_linear_shared={args.eq_token_linear_shared}, "
                f"attention_shared_qkv={args.eq_attention_shared_qkv}"
            )
        print(eq_msg)

    resume_path = None
    if args.resume_checkpoint is not None:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Requested resume checkpoint not found: {resume_path}")

    initial_epoch = 0
    optim_state = None
    scheduler_state = None
    if resume_path is not None and not args.force_retrain:
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optim_state = checkpoint.get("optimizer_state_dict")
        scheduler_state = checkpoint.get("scheduler_state_dict")
        pl_model.train_losses = checkpoint.get("train_losses", [])
        initial_epoch = int(checkpoint.get("epoch", 0))
        print(f"Loaded model + optimizer checkpoint from {resume_path}")

    epoch_checkpoint_callback = EpochProgressCheckpoint(
        latest_path=latest_checkpoint_path,
        mirror_path=checkpoint_path,
        args_dict=vars(args),
        data_file=data_file,
        epoch_offset=initial_epoch,
    )
    timing_log_path = out_dir / f"{args.artifact_prefix}_epoch_timing.txt"
    callbacks: list[Callback] = [
        epoch_checkpoint_callback,
        EpochTimingLogger(out_path=timing_log_path, epoch_offset=initial_epoch),
    ]
    if optim_state is not None or scheduler_state is not None:
        callbacks.append(
            RestoreTrainingStateCallback(
                optimizer_state_dict=optim_state,
                scheduler_state_dict=scheduler_state,
            )
        )
    if not args.no_track_test_rmse:
        rmse_log_path = out_dir / f"{args.artifact_prefix}_test_one_step_rmse_by_epoch.txt"
        callbacks.append(
            EpochTestRMSELogger(
                test_loader=test_loader,
                out_path=rmse_log_path,
                use_group_average_eval=args.group_average_eval,
                epoch_offset=initial_epoch,
            )
        )
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        limit_train_batches=args.limit_train_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
    )
    trainer.fit(pl_model, train_loader, val_loader)

    trained_optim_state = trainer.optimizers[0].state_dict() if getattr(trainer, "optimizers", None) else None
    trained_scheduler_state = (
        trainer.lr_scheduler_configs[0].scheduler.state_dict()
        if getattr(trainer, "lr_scheduler_configs", None)
        else None
    )
    final_payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trained_optim_state,
        "scheduler_state_dict": trained_scheduler_state,
        "train_losses": pl_model.train_losses,
        "args": vars(args),
        "data_file": data_file,
        "epoch": initial_epoch + int(args.epochs),
    }
    torch.save(final_payload, checkpoint_path)
    torch.save(final_payload, latest_checkpoint_path)
    print(f"Saved model + optimizer checkpoint to {checkpoint_path}")
    print(f"Updated latest checkpoint at {latest_checkpoint_path}")

    if pl_model.train_losses:
        loss_txt_path = out_dir / f"{args.artifact_prefix}_pde_training_loss.txt"
        with loss_txt_path.open("w", encoding="utf-8") as f:
            f.write("step\tloss\n")
            for step, loss_val in enumerate(pl_model.train_losses):
                f.write(f"{step}\t{loss_val:.10e}\n")
        print(f"Saved training loss TXT to {loss_txt_path}")

        if args.save_loss_pdf:
            loss_pdf_path = out_dir / f"{args.artifact_prefix}_pde_training_loss.pdf"
            plt.figure(figsize=(6, 4))
            plt.plot(pl_model.train_losses)
            plt.title("Official Tiny PDE-Transformer Training Loss")
            plt.xlabel("Training Step")
            plt.ylabel("MSE")
            plt.tight_layout()
            plt.savefig(loss_pdf_path)
            plt.close()
            print(f"Saved training loss PDF to {loss_pdf_path}")

    device = next(model.parameters()).device
    test_mse = evaluate_one_step_mse(
        model,
        test_loader,
        device,
        use_group_average=args.group_average_eval,
    )
    print(f"Test one-step MSE: {test_mse:.8f}")
    eval_suffix = "_gaeval" if args.group_average_eval else ""
    (out_dir / f"{args.artifact_prefix}_test_one_step_mse{eval_suffix}.txt").write_text(
        f"{test_mse:.10f}\n",
        encoding="utf-8",
    )
    d4_error = evaluate_d4_error(
        model,
        test_loader,
        device,
        use_group_average=args.group_average_eval,
        max_batches=args.equiv_eval_batches,
    )
    print(f"Test D4 equivariance error (mean |f(gx)-g f(x)|): {d4_error:.8f}")
    (out_dir / f"{args.artifact_prefix}_test_d4_error{eval_suffix}.txt").write_text(
        f"{d4_error:.10f}\n",
        encoding="utf-8",
    )

    test_dataset = PDEBenchShallowWaterDataset(
        file_path=data_file,
        split="test",
        history=args.history,
        horizon=args.horizon,
        max_start_step=max_start_step,
        field_idx=args.field_idx,
        split_seed=args.split_seed,
        normalize=False,
    )
    split_indices = _parse_csv_ints(args.rollout_split_indices)
    if not split_indices:
        split_indices = [0]

    for split_index in split_indices:
        if not (0 <= split_index < len(test_dataset.trajectory_indices)):
            print(
                f"Skipping rollout index {split_index}: outside test split range "
                f"[0, {len(test_dataset.trajectory_indices) - 1}]"
            )
            continue

        traj_key = test_dataset.trajectory_keys[test_dataset.trajectory_indices[split_index]]
        gt_traj = test_dataset.get_trajectory(split_index)  # [T,H,W]
        if gt_traj.shape[0] < args.history:
            print(
                f"Skipping rollout for {traj_key}: trajectory length {gt_traj.shape[0]} is shorter than history={args.history}"
            )
            continue
        x0 = gt_traj[: args.history].unsqueeze(0).to(device)
        dt = torch.tensor([test_dataset.dt], dtype=x0.dtype, device=device)
        pred_traj = rollout(
            model,
            x0,
            dt,
            n_steps=args.rollout_steps,
            use_group_average=args.group_average_eval,
        )
        print(f"Trajectory shape ({traj_key}): {tuple(pred_traj.shape)}")

        gt_start = int(args.history) - 1
        n_gt = min(gt_traj.shape[0] - gt_start, args.rollout_steps + 1)
        gt_traj_clip = gt_traj[gt_start:gt_start + n_gt].unsqueeze(0).unsqueeze(0).to(device)
        vmin = float(gt_traj_clip.min().item())
        vmax = float(gt_traj_clip.max().item())

        pred_gif_path = out_dir / f"{args.artifact_prefix}_{traj_key}_pred_rollout{eval_suffix}.gif"
        save_rollout_gif(pred_traj, pred_gif_path, vmin=vmin, vmax=vmax)
        print(f"Saved PDE rollout GIF to {pred_gif_path}")

        gt_gif_path = out_dir / f"{args.artifact_prefix}_{traj_key}_real_rollout.gif"
        save_rollout_gif(gt_traj_clip, gt_gif_path, vmin=vmin, vmax=vmax)
        print(f"Saved GT rollout GIF to {gt_gif_path}")
