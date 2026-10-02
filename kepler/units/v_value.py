"""V — 全局价值调制：V(h) = w1*C + w2*S + w3*F - w4*K（每代 z-score 归一）。"""
from __future__ import annotations

import numpy as np

from .b_hypothesis import NPARAMS

WEIGHTS = {"C": 1.0, "S": 1.0, "F": 1.0, "K": 1.0}
SURVIVORS = 8


def components(scores: list[dict]) -> dict:
    """四条分量（原始值，越大越好）：
    C 一致性 = -训练 RMSE            （与观测无矛盾）
    F 可证伪性 = -留出 RMSE          （对未见时刻的可检验预言）
    S 简洁性 = -(参数数 + 3×本轮项)   （奥卡姆；本轮重罚）
    K 冗余 = 本轮幅度比 (R2/R1)       （冗余项惩罚；非本轮=0）
    """
    C = np.array([-s["rmse_train"] for s in scores])
    F = np.array([-s["rmse_holdout"] for s in scores])
    S, K = [], []
    for s in scores:
        typ = s["hyp"]["type"]
        p = s["hyp"]["params"]
        n_epi = 1 if typ == "epicycle" else 0
        S.append(-(NPARAMS[typ] + 3 * n_epi))
        K.append((min(abs(p[5]) / (abs(p[2]) + 1e-9), 1.0)) if typ == "epicycle" else 0.0)
    return {"C": C, "S": np.array(S, float), "F": F, "K": np.array(K)}


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else x * 0.0


def value(scores: list[dict], weights: dict | None = None,
          use_v: bool = True) -> np.ndarray:
    w = weights or WEIGHTS
    c = components(scores)
    if not use_v:                        # 消融：关 V（随机价值 = 不选择）
        return np.zeros(len(scores))
    return (w["C"] * _z(c["C"]) + w["S"] * _z(c["S"])
            + w["F"] * _z(c["F"]) - w["K"] * _z(c["K"]))
