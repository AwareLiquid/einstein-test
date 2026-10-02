"""Kepler prototype driver — 集体溯因闭环 (A→B→C→D→V→E), blackboard + trace.

Phase 0: 冻结审计（红线）
Loop per planet:  A(gen1) → B → C → D → V → E → ... → 收敛
输出: blackboard/trace.jsonl（全代际假说与分数）+ blackboard/convergence.json
      （先方程后形态的最终输出）

Run:
    python kepler/driver.py [--generations 12] [--no-v] [--no-recombine]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from units import a_anomaly, b_hypothesis, c_derive, d_verify, e_recombine, v_value  # noqa: E402

FORBIDDEN = ["ellipse", "椭圆", "focus", "焦点", "kepler", "开普勒",
             "equal-area", "等面积", "eccentricity", "偏心率"]


def audit(obs_path: Path) -> None:
    """Phase 0：语料泄漏扫描（输入侧）。"""
    text = obs_path.read_text(encoding="utf-8").lower()
    hits = [w for w in FORBIDDEN if w in text]
    if hits:
        raise SystemExit(f"[AUDIT FAIL] frozen corpus contains future terms: {hits}")
    print("[audit] frozen corpus clean: no future-theory terms in inputs")


def run_planet(name: str, obs: dict, rng: np.random.Generator, gens: int,
               use_v: bool, use_recombine: bool, n_cand: int = 48,
               noise_sigma: float = 0.01) -> dict:
    t = np.asarray(obs["t"])
    split = d_verify.train_holdout_split(len(t), seed=hash(name) % 999)
    # A — 旧范式残差（gen1 之前一次）
    anom = a_anomaly.anomalies_for_planet(obs, noise_sigma)
    print(f"  [{name}] A: circle rms {anom['circle_rms']:.4f} "
          f"(noise {noise_sigma}) flagged {anom['flag_frac']*100:.0f}%")
    prior = {"r": 1.0 / max(anom["circle_rms"], 1e-6) ** 0 * 1.0,
             "n": abs(anom["circle_params"][3])}

    survivors: list[dict] = []
    best_hist = []
    best = None
    trace = []
    for gen in range(1, gens + 1):
        if gen == 1:
            cands = b_hypothesis.generate(rng, gen, n_cand, prior)
        else:
            cands = (e_recombine.recombine(rng, survivors, n_cand, prior)
                     if use_recombine
                     else b_hypothesis.generate(rng, gen, n_cand, prior,
                                                survivors=survivors))
        scores = [d_verify.score(h, obs, split) for h in cands]
        v = v_value.value(scores, use_v=use_v)
        order = np.argsort(-v)
        survivors = [scores[i]["hyp"] for i in order[:v_value.SURVIVORS]]
        if not use_v:                       # 消融：不选择 → 随机幸存（假说爆炸）
            survivors = [s["hyp"] for s in scores[:v_value.SURVIVORS]]

        # 记录本代最优（按留出误差）
        rm = np.array([s["rmse_holdout"] for s in scores])
        i_best = int(np.argmin(rm))
        row = {"planet": name, "gen": gen,
               "best_holdout": float(rm[i_best]),
               "best_hyp": scores[i_best]["hyp"],
               "type_hist": {ty: sum(1 for s in scores if s["hyp"]["type"] == ty)
                             for ty in b_hypothesis.TYPES},
               "survivor_types": [s["type"] for s in survivors]}
        trace.append(row)
        best_hist.append(float(rm[i_best]))
        if best is None or rm[i_best] < best["rmse_holdout"]:
            best = {"hyp": scores[i_best]["hyp"],
                    "rmse_holdout": float(rm[i_best]),
                    "rmse_train": float(scores[i_best]["rmse_train"]),
                    "gen": gen}
        if gen % 3 == 0 or gen == 1:
            print(f"    [{name}] gen {gen}: best holdout {rm[i_best]:.4f} "
                  f"({scores[i_best]['hyp']['type']}) survivors={row['survivor_types']}")
        # 收敛：连续 3 代无 >1% 改善 且 误差 < 5×噪声
        if len(best_hist) >= 4 and rm[i_best] < 5 * noise_sigma:
            recent = best_hist[-3:]
            if min(recent) > 0.99 * best_hist[-4]:
                break
    return {"planet": name, "anomaly": anom, "best": best, "trace": trace}


def beam(planets_out: list[dict]) -> dict:
    """最终输出：先方程（参数），后形态（自然导出的结论）。"""
    lines = []
    verdicts = []
    for p in planets_out:
        h = p["best"]["hyp"]
        if h["type"] == "conic":
            a, e, om, m0 = h["params"]
            lines.append(f"{p['planet']}: conic  a={a:.4f} e={e:.4f} "
                         f"omega={om:.4f} M0={m0:.4f}")
            shape = "ellipse (e>0)" if e > 0.02 else "circle-equivalent (e~0)"
            verdicts.append({"planet": p["planet"], "equation":
                             {"a": a, "e": e, "omega": om, "M0": m0},
                             "derived_shape": shape,
                             "rmse_holdout": p["best"]["rmse_holdout"],
                             "gen": p["best"]["gen"]})
        else:
            lines.append(f"{p['planet']}: {h['type']} {h['params']} "
                         f"(non-conic winner)")
            verdicts.append({"planet": p["planet"], "equation": None,
                             "winner_type": h["type"],
                             "rmse_holdout": p["best"]["rmse_holdout"]})
    return {"equation_first": lines, "verdicts": verdicts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--no-v", action="store_true")
    ap.add_argument("--no-recombine", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    obs_path = HERE / "data" / "observations.json"
    audit(obs_path)
    observations = json.loads(obs_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    print(f"=== Kepler 集群闭环（{args.generations} 代; "
          f"V={'off' if args.no_v else 'on'}, "
          f"E={'off' if args.no_recombine else 'on'}）===")
    out = []
    for obs in observations["planets"]:
        out.append(run_planet(obs["name"], obs, rng, args.generations,
                              use_v=not args.no_v,
                              use_recombine=not args.no_recombine))
    conv = beam(out)
    conv["elapsed_sec"] = round(time.time() - t0, 1)
    conv["config"] = vars(args)
    # 黑板落盘
    bb = HERE / "blackboard"
    with (bb / f"trace{args.out}.jsonl").open("w", encoding="utf-8") as f:
        for p in out:
            for row in p["trace"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (bb / f"convergence{args.out}.json").write_text(
        json.dumps(conv, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n--- 最终输出（方程先于形态）---")
    for line in conv["equation_first"]:
        print(" ", line)
    print(f"elapsed {conv['elapsed_sec']}s")


if __name__ == "__main__":
    main()
