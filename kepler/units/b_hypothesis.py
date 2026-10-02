"""B — 假说生成：从几何原语族采样候选假说（含大量错误项）。"""
from __future__ import annotations

import numpy as np

# 原语族（t0 前数学：圆、偏心圆、本轮、圆锥曲线）
TYPES = ("circle", "epicycle", "conic")
NPARAMS = {"circle": 5, "epicycle": 8, "conic": 4}


def random_candidate(rng: np.random.Generator, typ: str,
                     prior: dict | None = None) -> dict:
    prior = prior or {}
    if typ == "circle":
        p = [rng.normal(0, 0.5), rng.normal(0, 0.5),
             prior.get("r", 1.0) * rng.uniform(0.6, 1.4),
             prior.get("n", 1.0) * rng.uniform(0.7, 1.3),
             rng.uniform(0, 2 * np.pi)]
    elif typ == "epicycle":
        p = [rng.normal(0, 0.3), rng.normal(0, 0.3),
             prior.get("r", 1.0) * rng.uniform(0.6, 1.4),
             prior.get("n", 1.0) * rng.uniform(0.8, 1.2), rng.uniform(0, 2 * np.pi),
             prior.get("r", 1.0) * rng.uniform(0.02, 0.5),
             prior.get("n", 1.0) * rng.uniform(0.8, 1.5), rng.uniform(0, 2 * np.pi)]
    elif typ == "conic":
        p = [prior.get("r", 1.0) * rng.uniform(0.5, 1.6),
             rng.uniform(0.0, 0.7), rng.uniform(0, 2 * np.pi),
             rng.uniform(0, 2 * np.pi)]
    else:
        raise ValueError(typ)
    return {"type": typ, "params": [float(v) for v in p]}


def generate(rng: np.random.Generator, gen: int, n: int,
             prior: dict | None, survivors: list[dict] | None = None,
             fresh_frac: float = 0.4) -> list[dict]:
    """Gen1：纯先验采样；后续：幸存者变异 + 新随机（保多样性）。"""
    cands: list[dict] = []
    n_fresh = max(1, int(n * fresh_frac)) if gen > 1 else n
    weights = [0.5, 0.3, 0.2]  # circle / epicycle / conic 的基础比例
    for _ in range(n_fresh):
        typ = rng.choice(TYPES, p=weights)
        cands.append(random_candidate(rng, typ, prior))
    if gen > 1 and survivors:
        base = survivors
        while len(cands) < n:
            parent = base[rng.integers(0, len(base))]
            typ = parent["type"]
            if rng.random() < 0.10:                    # 结构变异
                typ = rng.choice([t for t in TYPES if t != typ])
                cands.append(random_candidate(rng, typ, prior))
                continue
            p = np.array(parent["params"])
            scale = 0.30 * (0.85 ** gen)               # 扰动随代数收缩
            p = p + rng.normal(0, 1, p.shape) * scale * (np.abs(p) + 0.05)
            if typ == "conic":
                p[1] = float(np.clip(p[1], 0.0, 0.8))
            if typ == "circle":
                p[2] = abs(p[2])
            if typ == "epicycle":
                p[2], p[5] = abs(p[2]), abs(p[5])
            cands.append({"type": typ, "params": [float(v) for v in p]})
    return cands
