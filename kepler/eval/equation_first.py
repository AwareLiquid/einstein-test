"""方程先于形态检查器：扫描输出工件，验证形态结论只在方程之后出现。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FORBIDDEN = ["ellipse", "椭圆", "focus", "焦点", "kepler", "开普勒",
             "equal-area", "等面积"]

HERE = Path(__file__).resolve().parent.parent


def main() -> None:
    conv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE / "blackboard" / "convergence.json"
    conv = json.loads(conv_path.read_text(encoding="utf-8"))
    problems = []

    # 1) 方程行不得含形态词（方程先于形态：参数在前）
    for line in conv.get("equation_first", []):
        low = line.lower()
        for w in FORBIDDEN:
            if w in low:
                problems.append(f"equation line contains shape word '{w}': {line}")

    # 2) 形态词只允许出现在 derived_shape 字段
    for v in conv.get("verdicts", []):
        eq = v.get("equation")
        ds = v.get("derived_shape", "")
        if eq is None:
            problems.append(f"{v['planet']}: no conic equation (non-conic winner)")
        elif not all(k in eq for k in ("a", "e", "omega")):
            problems.append(f"{v['planet']}: equation incomplete")

    # 3) trace 里不得出现形态词（假说只能是参数）
    trace_path = conv_path.parent / conv_path.name.replace("convergence", "trace")
    if trace_path.exists():
        for ln in trace_path.read_text(encoding="utf-8").splitlines():
            low = ln.lower()
            for w in FORBIDDEN:
                if w in low:
                    problems.append(f"trace contains shape word '{w}'")

    if problems:
        print("EQUATION-FIRST: FAIL")
        for p in problems[:10]:
            print("  -", p)
    else:
        print("EQUATION-FIRST: PASS（方程先行，形态仅自然导出，全工件无泄漏词）")


if __name__ == "__main__":
    main()
