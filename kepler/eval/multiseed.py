"""多种子稳定性评测：驱动跑 5 seed，聚合收敛率与根数精度（v1 判据）。

PASS 标准（预注册）：≥4/5 seed 收敛到圆锥曲线，且 e 误差 < 0.05、
ω 误差 < 0.15 rad（对 3 行星全部成立才算该 seed 成功）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent


def run_seed(seed: int, gens: int = 12) -> dict:
    out = HERE / "blackboard" / f"convergence_ms{seed}.json"
    cmd = [sys.executable, str(HERE / "driver.py"), "--generations", str(gens),
           "--seed", str(seed), "--out", f"_ms{seed}"]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise SystemExit(f"seed {seed} driver failed")
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> None:
    truth = json.loads((HERE / "truth.json").read_text(encoding="utf-8"))
    n_seeds = 5
    ok_seeds = 0
    rows = []
    for seed in range(n_seeds):
        conv = run_seed(seed)
        seed_ok = True
        detail = []
        for v in conv["verdicts"]:
            tr = truth[v["planet"]]
            if v.get("equation") is None:
                seed_ok = False
                detail.append(f"{v['planet']}:non-conic")
                continue
            da = abs(v["equation"]["a"] - tr["a"])
            de = abs(v["equation"]["e"] - tr["e"])
            dw = abs(v["equation"]["omega"] - tr["omega"])
            dw = min(dw, 2 * np.pi - dw)          # 角度环绕
            good = de < 0.05 and dw < 0.15
            seed_ok &= good
            detail.append(f"{v['planet']}:da={da:.3f} de={de:.3f} dw={dw:.3f}"
                          f"{'' if good else ' ✗'}")
        ok_seeds += int(seed_ok)
        rows.append({"seed": seed, "ok": seed_ok, "detail": detail,
                     "verdicts": conv["verdicts"]})
        print(f"seed {seed}: {'OK ' if seed_ok else 'BAD'} " + " | ".join(detail))

    print(f"\n多 seed 成功 {ok_seeds}/{n_seeds}")
    print("VERDICT:", "PASS" if ok_seeds >= 4 else "FAIL")
    (HERE / "blackboard" / "multiseed.json").write_text(
        json.dumps({"ok": ok_seeds, "n": n_seeds, "rows": rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
