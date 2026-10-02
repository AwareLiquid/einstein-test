"""C — 数理推导：从假说推导任意时刻的可观测预言（位置序列）。"""
from __future__ import annotations

import numpy as np

import orbits


def predict(hyp: dict, t: np.ndarray) -> np.ndarray:
    """hypothesis -> (N,2) positions at times t."""
    typ = hyp["type"]
    p = hyp["params"]
    if typ == "circle":
        cx, cy, r, n, ph = p
        return np.stack([cx + r * np.cos(n * t + ph),
                         cy + r * np.sin(n * t + ph)], 1)
    if typ == "epicycle":
        cx, cy, r1, n1, ph1, r2, n2, ph2 = p
        return np.stack([cx + r1 * np.cos(n1 * t + ph1) + r2 * np.cos(n2 * t + ph2),
                         cy + r1 * np.sin(n1 * t + ph1) + r2 * np.sin(n2 * t + ph2)], 1)
    if typ == "conic":
        a, e, om, m0 = p
        return orbits.conic_position(t, a, e, om, m0)
    raise ValueError(typ)
