"""A — 异常检测：拟合旧范式（正圆/偏心圆），输出残差与反常标记。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def fit_circle(t: np.ndarray, xy: np.ndarray) -> dict:
    """旧范式 v0：正圆 + 匀速角运动  p(t)=C+R[cos(nt+phi), sin(nt+phi)]。"""
    cx0, cy0 = xy.mean(axis=0)
    r0 = np.sqrt(((xy - [cx0, cy0]) ** 2).sum(1)).mean()
    # 初相位/角速度：atan2 展开后线性拟合
    ang = np.unwrap(np.arctan2(xy[:, 1] - cy0, xy[:, 0] - cx0))
    n0, phi0 = np.polyfit(t, ang, 1)

    def resid(p):
        cx, cy, r, n, ph = p
        pred = np.stack([cx + r * np.cos(n * t + ph),
                         cy + r * np.sin(n * t + ph)], 1)
        return (pred - xy).ravel()

    sol = least_squares(resid, [cx0, cy0, r0, n0, phi0],
                        method="lm", max_nfev=2000)
    rms = float(np.sqrt((sol.fun ** 2).mean()))
    return {"params": sol.x.tolist(), "rms": rms}


def anomalies_for_planet(obs: dict, noise_sigma: float = 0.01) -> dict:
    t = np.asarray(obs["t"])
    xy = np.stack([obs["x"], obs["y"]], 1)
    fit = fit_circle(t, xy)
    cx, cy, r, n, ph = fit["params"]
    pred = np.stack([cx + r * np.cos(n * t + ph),
                     cy + r * np.sin(n * t + ph)], 1)
    res = np.sqrt(((pred - xy) ** 2).sum(1))
    flagged = res > 3 * noise_sigma
    return {"planet": obs["name"], "circle_rms": fit["rms"],
            "flag_frac": float(flagged.mean()),
            "residual_median": float(np.median(res)),
            "circle_params": fit["params"],
            "anomalies": list(np.where(flagged)[0][:50])}


def run(observations: dict) -> dict:
    out = {"planets": {}}
    for obs in observations["planets"]:
        out["planets"][obs["name"]] = anomalies_for_planet(obs)
    return out
