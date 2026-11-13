from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class MatReader:


    def __init__(
        self,
        file_path: str | Path,
        to_torch: bool = True,
        to_cuda: bool = False,
        to_float: bool = True,
    ) -> None:
        self.file_path = str(file_path)
        self.to_torch = bool(to_torch)
        self.to_cuda = bool(to_cuda)
        self.to_float = bool(to_float)
        self.old_mat = True
        self.data: Any = None
        self._load_file()

    def _load_file(self) -> None:
        path = Path(self.file_path)
        if not path.is_file():
            raise FileNotFoundError(f"MAT file does not exist: {path}")

        scipy_error: Exception | None = None
        try:
            import scipy.io

            self.data = scipy.io.loadmat(path)
            self.old_mat = True
            return
        except Exception as exc:
            scipy_error = exc

        try:
            import h5py

            self.data = h5py.File(path, "r")
            self.old_mat = False
        except Exception as h5_error:
            raise RuntimeError(
                f"Cannot open MAT file {path} with scipy or h5py. "
                f"scipy error: {scipy_error}; h5py error: {h5_error}"
            ) from h5_error

    def set_cuda(self, to_cuda: bool) -> None:
        self.to_cuda = bool(to_cuda)

    def set_torch(self, to_torch: bool) -> None:
        self.to_torch = bool(to_torch)

    def set_float(self, to_float: bool) -> None:
        self.to_float = bool(to_float)

    def read_field(self, field: str):
        if field not in self.data:
            visible = [str(k) for k in self.data.keys() if not str(k).startswith("__")]
            preview = ", ".join(visible[:20])
            raise KeyError(f"Field {field!r} not found in {self.file_path}. Available: {preview}")

        if self.old_mat:
            value = self.data[field]
        else:
            value = self.data[field][()]
            if isinstance(value, np.ndarray) and value.ndim > 1:
                value = np.transpose(value, axes=tuple(range(value.ndim - 1, -1, -1)))

        value = np.asarray(value)
        if value.dtype.kind not in "biufc":
            raise TypeError(
                f"Field {field!r} is not a plain numeric array (dtype={value.dtype})."
            )
        if self.to_float:
            value = value.astype(np.float32, copy=False)
        if self.to_torch:
            value = torch.from_numpy(np.ascontiguousarray(value))
            if self.to_cuda:
                value = value.cuda(non_blocking=True)
        return value

    def close(self) -> None:
        if not self.old_mat and self.data is not None:
            try:
                self.data.close()
            finally:
                self.data = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class GaussianNormalizer:


    def __init__(self, x: torch.Tensor, flag: str = "Point", eps: float = 1.0e-6):
        x = torch.as_tensor(x)
        self.flag = str(flag)
        self.eps = float(eps)
        if self.flag.lower() == "point":
            self.mean = torch.mean(x, dim=0)
            self.std = torch.std(x, dim=0, unbiased=False)
        elif self.flag.lower() == "unit":
            self.mean = torch.mean(x)
            self.std = torch.std(x, unbiased=False)
        else:
            raise ValueError("flag must be 'Point' or 'Unit'.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor, sample_idx=None) -> torch.Tensor:
        del sample_idx
        return x * (self.std + self.eps) + self.mean

    def to(self, device=None, dtype=None):
        self.mean = self.mean.to(device=device, dtype=dtype)
        self.std = self.std.to(device=device, dtype=dtype)
        return self

    def cuda(self, device=None):
        return self.to(device=torch.device("cuda" if device is None else f"cuda:{device}"))

    def cpu(self):
        return self.to(device=torch.device("cpu"))


class GaussianNormalizer_out(GaussianNormalizer):
    pass


UnitGaussianNormalizer = GaussianNormalizer


def count_params(model: torch.nn.Module) -> int:


    total = 0
    for parameter in model.parameters():
        multiplier = 2 if parameter.is_complex() else 1
        total += multiplier * parameter.numel()
    return int(total)


class LpLoss:


    def __init__(self, p: int = 2, size_average: bool = True, eps: float = 1.0e-12):
        self.p = int(p)
        self.size_average = bool(size_average)
        self.eps = float(eps)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        diff = torch.linalg.vector_norm((x - y).reshape(batch, -1), ord=self.p, dim=1)
        denom = torch.linalg.vector_norm(y.reshape(batch, -1), ord=self.p, dim=1)
        values = diff / denom.clamp_min(self.eps)
        return values.mean() if self.size_average else values.sum()
