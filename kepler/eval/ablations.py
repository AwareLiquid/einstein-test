"""消融三件套：关 V / 关重组 / 单体集中式 baseline（证明集体不可替代）。

Runs the driver in 3 configs + a single-agent black-box baseline, then reports:
- 关 V:   幸存者类型分布（预期：本轮/复杂项爆炸，无法收敛到简单圆锥曲线）
- 关重组: 收敛代数与最终类型（预期：卡在偏心圆/本轮局部最优）
- 单体:   直接最小二乘高阶傅里叶拟合（拟合误差可低，但无方程先于形态轨迹）

Run:  python kepler/eval/ablations.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent.parent


def run_driver(tag: str, extra: list[str]) -> dict:
    out = HERE / "blackboard" / f"convergence_{tag}.json"
    cmd = [sys.executable, str(HERE / "driver.py"),
           "--out", f"_{tag}"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"driver {tag} failed")
    return json.loads(out.read_text(encoding="utf-8"))


def single_agent_baseline(k_harm: int = 6) -> dict:
    """单模型直推：p(t)=Σ_k a_k cos(k w t)+b_k sin(k w t)；黑箱高容量拟合。"""
    obs = json.loads((HERE / "data" / "observations.json").read_text(encoding="utf-8"))
    res = {}
    for o in obs["planets"]:
        t = np.asarray(o["t"])
        xy = np.stack([o["x"], o["y"]], 1)
        n = len(t)
        rng = np.random.default_rng(0)
        hold = rng.permutation(n)[: int(0.3 * n)]
        tr = np.setdiff1d(np.arange(n), hold)

        def pred(p, tt):
            w = np.exp(p[-1])
            out = np.zeros((len(tt), 2))
            for k in range(1, k_harm + 1):
                out += np.outer(np.cos(k * w * tt), p[2 * (k - 1):2 * k])
                out += np.outer(np.sin(k * w * tt), p[2 * k_harm + 2 * (k - 1):
                                                      2 * k_harm + 2 * k])
            return out + p[-3:-1]

        p0 = np.zeros(2 * k_harm * 2 + 3)
        p0[-1] = np.log(1.0)

        def resid(p):
            return (pred(p, t[tr]) - xy[tr]).ravel()

        sol = least_squares(resid, p0, max_nfev=3000)
        ho_rmse = float(np.sqrt(((pred(sol.x, t[hold]) - xy[hold]) ** 2)
                                .sum(1).mean()))
        res[o["name"]] = {"holdout_rmse": ho_rmse,
                          "n_params": len(sol.x),
                          "interpretable_equation": False}
    return res


def _survivor_conic_fraction(suffix: str) -> dict:
    """最终一代幸存者中圆锥曲线的占比（V 的直接作用面）。"""
    out = {}
    for p in ("P1", "P2", "P3"):
        # 从 trace 文件读取（driver 写的 trace{suffix}.jsonl）
        path = HERE / "blackboard" / f"trace{suffix}.jsonl"
        if not path.exists():
            out[p] = None
            continue
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
                if json.loads(ln)["planet"] == p]
        if not rows:
            out[p] = None
            continue
        last = rows[-1]["survivor_types"]
        out[p] = sum(1 for t in last if t == "conic") / max(len(last), 1)
    return out


def main() -> None:
    print("=== 消融：关 V ===")
    no_v = run_driver("nov", ["--no-v", "--generations", "12"])
    print("=== 消融：关重组 ===")
    no_e = run_driver("noe", ["--no-recombine", "--generations", "12"])
    print("=== 单体 baseline ===")
    single = single_agent_baseline()

    report = {"no_v": no_v["verdicts"], "no_recombine": no_e["verdicts"],
              "single_agent": single,
              "survivor_conic_fraction": {
                  "normal": _survivor_conic_fraction(""),
                  "no_v": _survivor_conic_fraction("_nov"),
                  "no_recombine": _survivor_conic_fraction("_noe"),
              }}
    out_path = HERE / "blackboard" / "ablations.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    print("\n--- 消融结果 ---")
    for tag, key in (("关 V", "no_v"), ("关重组", "no_recombine")):
        conv = report[key]
        n_conic = sum(1 for v in conv if v.get("equation") is not None)
        print(f"{tag}: 收敛到圆锥曲线 {n_conic}/3")
        for v in conv:
            print(f"   {v['planet']}: {v.get('winner_type', 'conic')} "
                  f"rmse_ho {v['rmse_holdout']:.4f} gen {v.get('gen', '-')}")
    print("单体 baseline (无方程先于形态轨迹):")
    for k, v in single.items():
        print(f"   {k}: holdout {v['holdout_rmse']:.4f} "
              f"({v['n_params']} 黑箱参数)")
    print("最终代幸存者 conic 占比 (V 的直接作用面):")
    for tag, fr in report["survivor_conic_fraction"].items():
        print(f"   {tag}: {fr}")


if __name__ == "__main__":
    main()
