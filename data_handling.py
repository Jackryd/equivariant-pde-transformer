from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

def _apply_d4(x: torch.Tensor, y: torch.Tensor, k, b) -> tuple[torch.Tensor, torch.Tensor]:
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
        y = torch.rot90(y, k, dims=(-2, -1))
    if b:
        x = torch.flip(x, dims=(-1,))
        y = torch.flip(y, dims=(-1,))
    return x.contiguous(), y.contiguous()

def _apply_random_d4(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same random D4 transform to x and y (last two dims are H,W)."""
    k = int(torch.randint(0, 4, (1,)).item())
    b = bool(torch.randint(0, 2, (1,)).item()) 
    return _apply_d4(x, y, k, b)

class PDEBenchShallowWaterDataset(Dataset):
    def __init__(
        self,
        file_path: str = "./data/2D/shallow-water/2D_rdb_NA_NA.h5",
        split: str = "train",
        history: int = 1,
        horizon: int = 1,
        max_start_step: int | None = None,
        field_idx: int = 0,
        split_seed: int = 0,
        normalize: bool = False,
        augment: bool = False,
    ):
        if h5py is None:
            raise ImportError("h5py is required for PDEBench HDF5 loading. Install with: python3 -m pip install h5py")

        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Use 'train', 'val', or 'test'.")
        if history < 1:
            raise ValueError("history must be at least 1.")
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")
        if max_start_step is not None and int(max_start_step) < 0:
            raise ValueError("max_start_step must be non-negative or None.")

        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Could not find PDEBench shallow-water file: {self.file_path}")

        self.split = split
        self.history = int(history)
        self.horizon = int(horizon)
        self.max_start_step = None if max_start_step is None else int(max_start_step)
        self.field_idx = int(field_idx)
        self.normalize = bool(normalize)
        self.augment = bool(augment)
        self._h5 = None

        with h5py.File(self.file_path, "r") as f:
            self.trajectory_keys = sorted(f.keys())
            if not self.trajectory_keys:
                raise ValueError(f"No trajectories found in {self.file_path}")

            first_key = self.trajectory_keys[0]
            first_data = f[first_key]["data"]
            if first_data.ndim != 4:
                raise ValueError(
                    f"Expected trajectory tensors with shape [T,H,W,C], got {tuple(first_data.shape)}"
                )

            self.num_frames, self.height, self.width, self.num_fields = map(int, first_data.shape)
            if not (0 <= self.field_idx < self.num_fields):
                raise ValueError(f"field_idx must be in [0, {self.num_fields - 1}], got {self.field_idx}")

            time_grid = torch.from_numpy(f[first_key]["grid"]["t"][:]).float()
            self.dt = float(time_grid[1] - time_grid[0]) if time_grid.numel() >= 2 else 1.0

        total_windows = self.num_frames - self.history - self.horizon + 1
        if self.max_start_step is None:
            self.windows_per_trajectory = total_windows
        else:
            # start index is 0-based; max_start_step=9 keeps starts 0..9 (10 timesteps)
            self.windows_per_trajectory = min(total_windows, self.max_start_step + 1)
        if self.windows_per_trajectory <= 0:
            raise ValueError(
                f"history={self.history} and horizon={self.horizon} are too large for "
                f"{self.num_frames} frames."
            )

        num_trajectories = len(self.trajectory_keys)
        permutation = torch.randperm(num_trajectories, generator=torch.Generator().manual_seed(split_seed)).tolist()
        train_end = int(0.8 * num_trajectories)
        val_end = train_end + int(0.1 * num_trajectories)
        split_indices = {
            "train": permutation[:train_end],
            "val": permutation[train_end:val_end],
            "test": permutation[val_end:],
        }
        self.trajectory_indices = split_indices[self.split]
        if not self.trajectory_indices:
            raise ValueError(f"No trajectories assigned to split '{self.split}'.")

        self.dt_tensor = torch.tensor(self.dt, dtype=torch.float32).view(1, 1, 1)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.file_path, "r")
        return self._h5

    def __len__(self):
        return len(self.trajectory_indices) * self.windows_per_trajectory

    def __getitem__(self, idx):
        split_traj_idx = idx // self.windows_per_trajectory
        start = idx % self.windows_per_trajectory
        traj_key = self.trajectory_keys[self.trajectory_indices[split_traj_idx]]

        data = self._get_h5()[traj_key]["data"]
        x = torch.from_numpy(data[start:start + self.history, :, :, self.field_idx]).float()
        y = torch.from_numpy(
            data[start + self.history:start + self.history + self.horizon, :, :, self.field_idx]
        ).float()

        if self.augment and self.split == "train":
            x, y = _apply_random_d4(x, y)

        if self.normalize:
            mean = x.mean()
            std = x.std().clamp_min(1e-6)
            x = (x - mean) / std
            y = (y - mean) / std

        return x.contiguous(), y.contiguous(), self.dt_tensor

    def get_trajectory(self, split_index: int) -> torch.Tensor:
        if not (0 <= split_index < len(self.trajectory_indices)):
            raise IndexError(f"split_index must be in [0, {len(self.trajectory_indices) - 1}]")
        traj_key = self.trajectory_keys[self.trajectory_indices[split_index]]
        data = self._get_h5()[traj_key]["data"][:, :, :, self.field_idx]
        return torch.from_numpy(data[:]).float().contiguous()


def build_pdebench_shallow_water_loader(
    file_path: str = "./data/2D/shallow-water/2D_rdb_NA_NA.h5",
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool | None = None,
    num_workers: int = 0,
    history: int = 1,
    horizon: int = 1,
    max_start_step: int | None = None,
    field_idx: int = 0,
    split_seed: int = 0,
    normalize: bool = False,
    augment: bool = False,
):
    dataset = PDEBenchShallowWaterDataset(
        file_path=file_path,
        split=split,
        history=history,
        horizon=horizon,
        max_start_step=max_start_step,
        field_idx=field_idx,
        split_seed=split_seed,
        normalize=normalize,
        augment=augment,
    )
    if shuffle is None:
        shuffle = split.lower() == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
