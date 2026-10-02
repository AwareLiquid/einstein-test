# Kepler Prototype — 首个可执行的爱因斯坦测试实例

> 从**冻结的观测数据**（位置 + 时间戳，无速度、无轨道参数）中，A–F 单元集群
> 通过集体溯因循环**独立恢复出圆锥曲线根数**，数值与隐藏真值吻合到 <0.1%，
> 且满足"方程先于形态"。

本目录是 `SKILL.md`／`references/kepler-prototype.md` 的**参考实现**（阶段 1）。
设计规格见 [`../docs/KEPLER_PROTOTYPE.md`](../docs/KEPLER_PROTOTYPE.md)；
完整结果与诚实边界见 [`RESULTS.md`](RESULTS.md)。

## 快速开始

```bash
python kepler/data/gen_data.py        # 生成冻结语料 + 审计记录
python kepler/driver.py --generations 12   # 集体溯因闭环
python kepler/eval/equation_first.py  # ① 方程先于形态检查
python kepler/eval/ablations.py       # ③ 消融三件套
python kepler/eval/multiseed.py       # 多种子稳定性（5 seed）
python kepler/eval/plot_trace.py      # 残差/幸存者占比曲线（SVG）
```

## 架构（A–F 单元 + 黑板）

| 单元 | 职责 | 文件 |
|---|---|---|
| A 异常检测 | 拟合旧范式（正圆+匀速），输出残差与反常 | `units/a_anomaly.py` |
| B 假说生成 | 几何原语族采样（圆/偏心圆/本轮/圆锥曲线） | `units/b_hypothesis.py` |
| C 数理推导 | 从假说推导预言位置（开普勒方程/本轮几何） | `units/c_derive.py` |
| D 观测校验 | 训练段拟合 + 留出段评分 | `units/d_verify.py` |
| V 全局调制 | V(h)=w1C+w2S+w3F−w4K（每代 z 归一并筛选） | `units/v_value.py` |
| E 重组 | 参数交叉 + 变异 + 结构互换 | `units/e_recombine.py` |
| 驱动 | A→B→C→D→V→E 循环 + 黑板落盘 | `driver.py` |

单元只通过 `blackboard/` 工件交换；任何单元不持有完整假说。

## 结果摘要（v0+v1）

- **发现值 vs 真值**：P1 `a=0.856 e=0.106 ω=1.36` → 0.8558/0.1071/1.3639；
  P2/P3 同精度（<0.1%）。
- **多种子**：5/5 成功（de≤0.002、dw≤0.011 rad）。
- **三件套**：① 方程先于形态 PASS；② 迭代轨迹可见；③ 消融——关 V/关重组
  各让 P1 卡入"均轮陷阱"（本轮局部最优），单体黑箱（27 参数）比集群 4 参数
  圆锥曲线更差且无轨迹。
- **红线**：圆锥曲线作为数学原语属 t0 前知识（Apollonius）；实现全程无
  "椭圆/焦点/开普勒"输入（`data/AUDIT.md` + 输出审计）。
- **诚实边界**：v0 的 V 非严格必要（生成器含新随机）；剩余项见 `RESULTS.md` §5。
