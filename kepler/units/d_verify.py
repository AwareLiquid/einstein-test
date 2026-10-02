"""D — 观测校验：训练段拟合 + 留出段评分（可证伪性的实测量）。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .c_derive import predict

# 参数边界（几何/物理合法性；不含任何未来知识）
_BOUNDS = {
    "circle": ([-3, -3, 0.05, 1e-3, -10], [3, 3, 3.0, 10.0, 10.0]),
    "epicycle": ([-3, -3, 0.05, 1e-3, -10, 0.0, 1e-3, -10],
                 [3, 3, 3.0, 10.0, 10.0, 1.5, 10.0, 10.0]),
    "conic": ([0.3, 0.0, -10, -20], [2.0, 0.85, 10, 20]),
}


def fit(hyp: dict, t: np.ndarray, xy: np.ndarray) -> dict:
    """在训练段最小二乘精修参数。"""
    lo, hi = _BOUNDS[hyp["type"]]
    p0 = np.clip(np.array(hyp["params"], dtype=float), lo, hi)

    def resid(p):
        h = {"type": hyp["type"], "params": p}
        return (predict(h, t) - xy).ravel()

    sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=400)
    return {"type": hyp["type"], "params": sol.x.tolist()}


def train_holdout_split(n: int, seed: int = 0, frac: float = 0.3
                        ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_hold = int(n * frac)
    return idx[n_hold:], idx[:n_hold]         # train(70%), holdout(30%)


def score(hyp: dict, obs: dict, split: tuple[np.ndarray, np.ndarray]) -> dict:
    t = np.asarray(obs["t"])
    xy = np.stack([obs["x"], obs["y"]], 1)
    tr, ho = split
    fitted = fit(hyp, t[tr], xy[tr])
    pred_tr = predict(fitted, t[tr])
    pred_ho = predict(fitted, t[ho])
    rmse_tr = float(np.sqrt(((pred_tr - xy[tr]) ** 2).sum(1).mean()))
    rmse_ho = float(np.sqrt(((pred_ho - xy[ho]) ** 2).sum(1).mean()))
    return {"hyp": fitted, "rmse_train": rmse_tr, "rmse_holdout": rmse_ho}
