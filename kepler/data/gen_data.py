"""Kepler prototype — data generation (M1).

Generates the frozen corpus O_t0: positions-only observations of 3 planets on
eccentric orbits, irregular sparse timestamps, gaussian noise. This is the
"Tycho data" the cluster must explain WITHOUT any knowledge of ellipses.

Also writes AUDIT.md: the frozen-corpus declaration (no t>t0 theory/terms).

Run:
    python kepler/data/gen_data.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orbits import conic_position  # noqa: E402

HERE = Path(__file__).resolve().parent
MU = 1.0                       # GM of the central body (origin)
orbit_position = conic_position


def main() -> None:
    rng = np.random.default_rng(20261002)
    planets = []
    truth = {}
    for i in range(3):
        a = float(rng.uniform(0.8, 1.2))
        e = float(rng.uniform(0.05, 0.6))
        omega = float(rng.uniform(0, 2 * math.pi))
        M0 = float(rng.uniform(0, 2 * math.pi))
        period = 2 * math.pi * math.sqrt(a ** 3 / MU)
        # ~200 observations over >=3 periods, irregular timestamps
        span = period * rng.uniform(3.0, 3.6)
        t = np.sort(rng.uniform(0, span, 200))
        t = t + rng.normal(0, 0.02, t.shape)   # timing jitter
        t = np.clip(t, 0, None)
        pos = orbit_position(t, a, e, omega, M0)
        pos = pos + rng.normal(0, 0.01, pos.shape)   # measurement noise
        planets.append({"name": f"P{i+1}", "t": t.tolist(),
                        "x": pos[:, 0].tolist(), "y": pos[:, 1].tolist()})
        truth[f"P{i+1}"] = {"a": a, "e": e, "omega": omega, "M0": M0,
                            "period": period}

    (HERE / "observations.json").write_text(
        json.dumps({"mu": MU, "planets": planets}, indent=1), encoding="utf-8")
    # truth kept OUT of the corpus dir (leak discipline) — for the grader only
    (HERE.parent / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")

    audit = [
        "# 冻结语料审计（Phase 0）", "",
        "- 生成时间：2026-10-02；随机种子 20261002（可复现）",
        "- 语料内容：3 颗行星的位置观测 (t, x, y)；速度/质量/轨道参数**不在语料内**",
        "- 观测噪声 σ=0.01；时间戳抖动 σ=0.02；每行星 200 点，跨 ≥3 个周期",
        "- 真值参数写入 `kepler/truth.json`（评审员专用；**任何单元/提示词不得读取**）",
        "- 泄漏审计：输入/输出词表禁止出现 ellipse/椭圆/focus/焦点/Kepler/开普勒/"
        "equal-area/等面积；圆锥曲线作为数学对象（Apollonius，t0 前）允许",
        "- 冻结红线：语料生成后只读；任何改动必须重新生成 + 重审",
    ]
    (HERE / "AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(f"wrote {HERE / 'observations.json'} "
          f"({sum(len(p['t']) for p in planets)} points)")
    print("truth written to kepler/truth.json (grader-only)")
    for name, tp in truth.items():
        print(f"  {name}: a={tp['a']:.3f} e={tp['e']:.3f} "
              f"omega={tp['omega']:.2f} period={tp['period']:.2f}")


if __name__ == "__main__":
    main()
