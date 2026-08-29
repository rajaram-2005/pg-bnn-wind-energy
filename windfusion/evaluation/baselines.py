"""Reference baselines so WindFusion numbers are never self-referential.

All baselines share the same numpy interface (``fit``/``predict``) and are
scored by the same metric code as the neural models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BaselineArrays:
    """Flattened arrays extracted from a window dataset."""

    X: np.ndarray  # (N, T, C)
    P: np.ndarray  # (N, Pf)
    y: np.ndarray  # (N, 3)

    @property
    def flat(self) -> np.ndarray:
        return self.X.reshape(self.X.shape[0], -1)

    def combined(self) -> np.ndarray:
        return np.concatenate([self.flat, self.P], axis=1)


def dataset_arrays(dataset, limit: int | None = None, batch_size: int = 256) -> BaselineArrays:
    """Materialise a window dataset into numpy arrays (order is dataset order)."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    xs, ps, ys = [], [], []
    count = 0
    for batch in loader:
        xs.append(batch["sequence"].numpy())
        ps.append(batch["physics"].numpy())
        ys.append(batch["target"].numpy())
        count += batch["sequence"].shape[0]
        if limit is not None and count >= limit:
            break
    return BaselineArrays(
        X=np.concatenate(xs)[: limit or count],
        P=np.concatenate(ps)[: limit or count],
        y=np.concatenate(ys)[: limit or count],
    )


class MeanBaseline:
    """Predicts the training mean with a constant residual standard deviation."""

    name = "mean"

    def fit(self, data: BaselineArrays) -> "MeanBaseline":
        self.mean = data.y.mean(0)
        self.std = data.y.std(0) + 1e-6
        return self

    def predict(self, data: BaselineArrays) -> tuple[np.ndarray, np.ndarray]:
        mean = np.tile(self.mean, (data.y.shape[0], 1))
        std = np.tile(self.std, (data.y.shape[0], 1))
        return mean, std


class RidgeBaseline:
    """Ridge regression on flattened windows (scikit-learn when available)."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, data: BaselineArrays) -> "RidgeBaseline":
        X = data.combined()
        y = data.y
        X = np.nan_to_num(X)
        try:
            from sklearn.linear_model import Ridge

            model = Ridge(alpha=self.alpha)
            model.fit(X, y)
            self.coef_ = model.coef_
            self.intercept_ = model.intercept_
        except Exception:  # pragma: no cover - fallback keeps the baseline runnable
            ones = np.ones((X.shape[0], 1))
            Xd = np.concatenate([X, ones], axis=1)
            penalty = self.alpha * np.eye(Xd.shape[1])
            solution = np.linalg.solve(Xd.T @ Xd + penalty, Xd.T @ y)
            self.coef_ = solution[:-1].T
            self.intercept_ = solution[-1]
        residual = y - (X @ self.coef_.T + self.intercept_)
        self.residual_std_ = residual.std(0) + 1e-6
        return self

    def predict(self, data: BaselineArrays) -> tuple[np.ndarray, np.ndarray]:
        X = np.nan_to_num(data.combined())
        mean = X @ self.coef_.T + self.intercept_
        std = np.tile(self.residual_std_, (X.shape[0], 1))
        return mean, std


class TorchBaseline:
    """Shared training skeleton for the small torch baselines."""

    name = "torch"

    def __init__(self, epochs: int = 20, lr: float = 1e-3, batch_size: int = 128, seed: int = 7):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

    def _features(self, data: "BaselineArrays"):
        """Model input tensor: sequence models keep time, pooled models do not."""
        X = torch.from_numpy(np.nan_to_num(data.X, nan=0.0).astype(np.float32))
        P = torch.from_numpy(np.nan_to_num(data.P, nan=0.0).astype(np.float32))
        return torch.cat([X, P.unsqueeze(1).expand(-1, X.shape[1], -1)], dim=-1)

    def _build(self, input_dim: int, output_dim: int):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, data: BaselineArrays) -> "TorchBaseline":
        torch.manual_seed(self.seed)
        inputs = self._features(data)
        y = torch.from_numpy(data.y.astype(np.float32))
        self.output_dim = y.shape[-1]
        self.model = self._build(inputs.shape[-1], y.shape[-1])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        n = inputs.shape[0]
        self.model.train()
        for _ in range(self.epochs):
            order = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                pred = self.model(inputs[idx])
                loss = torch.nn.functional.mse_loss(pred, y[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            residual = y - self.model(inputs)
            self.residual_std_ = residual.std(0).numpy() + 1e-6
        return self

    def predict(self, data: BaselineArrays) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            mean = self.model(self._features(data)).numpy()
        return mean, np.tile(self.residual_std_, (mean.shape[0], 1))


class MLPBaseline(TorchBaseline):
    """Flattened-window MLP: mean-pools the window, ignores ordering."""

    name = "mlp"

    def _build(self, input_dim: int, output_dim: int):
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        )

    def _features(self, data: "BaselineArrays"):
        """Mean-pool the window: the MLP has no notion of ordering."""
        X = torch.from_numpy(np.nan_to_num(data.X, nan=0.0).astype(np.float32)).mean(1)
        P = torch.from_numpy(np.nan_to_num(data.P, nan=0.0).astype(np.float32))
        return torch.cat([X, P], dim=-1)


class GRUBaseline(TorchBaseline):
    """Recurrent baseline: the temporal reference point for the TCN encoder."""

    name = "gru"

    def _build(self, input_dim: int, output_dim: int):
        class _GRU(torch.nn.Module):
            def __init__(self, dim: int, out: int):
                super().__init__()
                self.gru = torch.nn.GRU(dim, 64, batch_first=True)
                self.head = torch.nn.Linear(64, out)

            def forward(self, x):
                _, hidden = self.gru(x)
                return self.head(hidden[-1])

        return _GRU(input_dim, output_dim)


def build_baselines(epochs: int = 20) -> dict[str, object]:
    """Default baseline suite used by the synthetic benchmark."""
    return {
        "mean": MeanBaseline(),
        "ridge": RidgeBaseline(alpha=1.0),
        "mlp": MLPBaseline(epochs=epochs),
        "gru": GRUBaseline(epochs=epochs),
    }


__all__ = [
    "BaselineArrays",
    "GRUBaseline",
    "MLPBaseline",
    "MeanBaseline",
    "RidgeBaseline",
    "TorchBaseline",
    "build_baselines",
    "dataset_arrays",
]
