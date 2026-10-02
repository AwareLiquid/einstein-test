"""E — 重组：幸存假说的参数交叉 + 变异 + 结构互换，生成下一代候选。"""
from __future__ import annotations

import numpy as np

from .b_hypothesis import NPARAMS, random_candidate


def recombine(rng: np.random.Generator, survivors: list[dict], n: int,
              prior: dict | None = None) -> list[dict]:
    out: list[dict] = []
    for _ in range(n):
        a, b = survivors[rng.integers(0, len(survivors))], \
               survivors[rng.integers(0, len(survivors))]
        if a["type"] == b["type"]:
            pa, pb = np.array(a["params"]), np.array(b["params"])
            w = rng.uniform(0.3, 0.7)
            p = w * pa + (1 - w) * pb
            typ = a["type"]
        else:                                  # 结构不同 -> 取其一并微扰
            src = a if rng.random() < 0.5 else b
            typ, p = src["type"], np.array(src["params"])
        scale = 0.15 * (np.abs(p) + 0.05)
        p = p + rng.normal(0, 1, p.shape) * scale
        if typ == "conic":
            p[1] = float(np.clip(p[1], 0.0, 0.8))
            p[0] = float(np.clip(p[0], 0.3, 2.0))
        if typ in ("circle", "epicycle"):
            p[2] = abs(p[2])
        if typ == "epicycle":
            p[5] = abs(p[5])
        out.append({"type": typ, "params": [float(v) for v in p]})
    # 注入少量全新随机（防早熟）
    for _ in range(max(1, n // 8)):
        out[rng.integers(0, n)] = random_candidate(
            rng, rng.choice(["circle", "epicycle", "conic"]), prior)
    assert all(len(h["params"]) == NPARAMS[h["type"]] for h in out)
    return out
