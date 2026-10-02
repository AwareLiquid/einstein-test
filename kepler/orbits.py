"""Shared orbital math for the Kepler prototype (t0 前知识：圆锥曲线+代数).

Used by: data generation (truth), C-derivation (predictions), truth grader.
No future knowledge: this module knows conic geometry and Newton iteration,
NOT that planets follow these curves.
"""
from __future__ import annotations

import math

import numpy as np


def kepler_solve(M: np.ndarray, e: float, iters: int = 60) -> np.ndarray:
    """Solve M = E - e sin E for E (Newton, vectorised)."""
    E = M.copy()
    for _ in range(iters):
        d = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - d
        if np.max(np.abs(d)) < 1e-12:
            break
    return E


def conic_position(t: np.ndarray, a: float, e: float, omega: float,
                   M0: float, mu: float = 1.0) -> np.ndarray:
    """Position (N,2) of a conic trajectory; focus at the origin."""
    e = float(np.clip(e, 0.0, 0.95))
    n = math.sqrt(mu / a ** 3)
    M = (M0 + n * t) % (2 * math.pi)
    E = kepler_solve(M, e)
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                        np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1 - e * np.cos(E))
    x = r * np.cos(nu + omega)
    y = r * np.sin(nu + omega)
    return np.stack([x, y], axis=-1)
