"""残差曲线图（迭代轨迹证据）：best-holdout 随代数下降 + 圆锥曲线幸存占比。

纯 numpy 手写 SVG（无 matplotlib 依赖）。读 blackboard/trace*.jsonl。
用法: python kepler/eval/plot_trace.py [suffix]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
W, H, PAD = 640, 400, 60


def load(suffix: str) -> dict:
    path = HERE / "blackboard" / f"trace{suffix}.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    out = {}
    for r in rows:
        out.setdefault(r["planet"], []).append(r)
    return out


def svg_for(series: list[dict], planet: str) -> str:
    gens = [r["gen"] for r in series]
    best = [r["best_holdout"] for r in series]
    frac = [sum(1 for t in r["survivor_types"] if t == "conic")
            / max(len(r["survivor_types"]), 1) for r in series]
    x = lambda g: PAD + (g - min(gens)) / max(max(gens) - min(gens), 1) * (W - 2 * PAD)
    y = lambda v: H - PAD - v / max(max(best) * 1.1, 1e-9) * (H - 2 * PAD)
    yf = lambda v: H - PAD - v * (H - 2 * PAD)

    pts = " ".join(f"{x(g):.1f},{y(v):.1f}" for g, v in zip(gens, best))
    pts_f = " ".join(f"{x(g):.1f},{yf(v):.1f}" for g, v in zip(gens, frac))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">
<rect width="100%" height="100%" fill="#0e0e12"/>
<text x="{PAD}" y="24" fill="#ebebeb" font-size="14">{planet} — best holdout RMSE（红） vs 圆锥曲线幸存占比（青）</text>
<polyline points="{pts}" fill="none" stroke="#ff5a5a" stroke-width="2"/>
<polyline points="{pts_f}" fill="none" stroke="#3ad6c8" stroke-width="1.5" stroke-dasharray="5 3"/>
<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#555"/>
<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#555"/>
<text x="{PAD-8}" y="{PAD+8}" fill="#888" font-size="10" text-anchor="end">1.0</text>
<text x="{PAD-8}" y="{H-PAD}" fill="#888" font-size="10" text-anchor="end">0</text>
<text x="{W-PAD}" y="{H-PAD+16}" fill="#888" font-size="10" text-anchor="end">gen {max(gens)}</text>
</svg>"""


def main() -> None:
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    data = load(suffix)
    for planet, series in data.items():
        outp = HERE / "blackboard" / f"residual_curve_{planet}{suffix}.svg"
        outp.write_text(svg_for(series, planet), encoding="utf-8")
        h0, h1 = series[0]["best_holdout"], series[-1]["best_holdout"]
        print(f"{planet}: gen1 {h0:.4f} -> gen{series[-1]['gen']} {h1:.4f} "
              f"({(h0-h1)/h0*100:.1f}% 下降)  -> {outp.name}")


if __name__ == "__main__":
    main()
