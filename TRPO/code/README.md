# TRPO 数值实验

Python 3.10+，仅标准库。所有命令从仓库根目录执行。

## 1. 快速运行

```bash
python TRPO/code/trpo_demo.py
python TRPO/code/trpo_demo.py --problem contextual
python TRPO/code/trpo_demo.py --initial-p 0.01
python TRPO/code/trpo_demo.py --initial-p 0.01 --max-backtracks 1
python TRPO/code/trpo_demo.py --problem contextual --iterations 15 --damping 0.1
python -m unittest discover -s TRPO/code -p test_trpo_demo.py -v
```

输出 JSON 到标准输出；需要保存时可使用 `> result.json`。程序不自动写训练文件，不依赖其他章节代码。

## 2. 范围与目标

- `single`：一个状态、两个动作，奖励 `[0,2]`，一步结束。
- `contextual`：两个上下文各以 0.5 概率出现，奖励分别为 `[0,2]`、`[2,0]`，一步结束。

每个上下文有独立参数 $\theta_i$，动作 1 概率为 $\sigma(\theta_i)$。上下文分布固定，不受动作影响，因此精确代理增益恰等于真实价值增益。这能核对优化器，却不能用来证明一般 MDP 中代理目标与真实性能一致。

演示直接枚举动作与上下文期望，不抽样，没有 Critic、GAE、Gymnasium 或深层网络。它实现 TRPO 的局部求解与回溯核心；完整采样流程见[讲义](../README.md#第-4-讲回溯线搜索与完整训练流程)。

## 3. 参数与输出

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--problem` | `single` | 单状态或 `contextual` |
| `--initial-p` | 0.5 | 每个上下文初始动作 1 概率，严格在 `(0,1)` |
| `--delta` | 0.01 | 旧到新平均 KL 预算，正有限数 |
| `--damping` | 0 | Fisher 阻尼，非负；默认零便于复现手算 |
| `--iterations` | 1 | 更新轮数，正整数 |
| `--cg-iters` | 10 | 每轮 CG 上限 |
| `--max-backtracks` | 10 | 含完整步的候选总数 |

回溯因子为 0.5，最低增益比为 0.1；可在函数接口中设置。JSON 的 `attempts` 保存每个候选的比例、完整 KL、实际与预期代理增益、是否接受；顶层 `kl` 和 `gain` 是最终应用的步，更新失败时均为零。失败时原样返回旧参数，`reason` 说明原因。

## 4. 公式对照

| 函数 | 含义 |
| --- | --- |
| `objective` | 精确平均奖励 J |
| `surrogate` | 冻结旧优势，精确计算 $\sum_s w_s\sum_a\pi_{new}(a\mid s)A_{old}(s,a)$ |
| `mean_kl` | $\sum_s w_s D_{KL}(\pi_{old}\|\pi_{new})$ |
| `gradient` | $g_i=w_i p_i(1-p_i)[R_i(1)-R_i(0)]$ |
| `fisher_vector_product` | $(F+\eta I)v$，$F_{ii}=w_i p_i(1-p_i)$ |
| `conjugate_gradient` | 矩阵向量积接口求解线性系统 |
| `trpo_step` | 求方向、按阻尼二次型缩放、回溯检查、失败恢复 |

为数值稳定，精确动作求和时先代数约去旧概率，`surrogate` 不显式计算比率；KL 用稳定 log probability 计算。Fisher 因独立参数而为对角形式，但求解仍用 CG，方便替换为神经网络 Hessian 向量积。高维模型不能照抄这个对角解析式。

## 5. 观察与解释

默认完整步应通过：动作 1 概率约 0.570243，J 从 1 增到 1.140486，KL 约 0.009966843。上下文实验中两个参数方向相反，但好动作概率都增加。

初始概率 0.01 时，完整步二阶预测 KL 为 0.01，完整分布 KL 约 0.016729512，所以回退到半步。这是本实验最关键的观察：二阶近似不能代替非线性检查。

增大阻尼后步长通常更保守。多轮时，每轮重新固定旧参数与优势；预算限制的是相邻两轮策略，不能把每轮 KL 与初始策略到最终策略的 KL 混为一谈。单次测试看到 J 单调，来自本例特殊的精确一步期望结构，不是普遍的 TRPO 收敛结论。

验收：复现[例题 1–4](../examples.md)，解释每个失败候选，展示零梯度保留旧参数，以及超预算且无后续候选时的恢复行为。

[验证记录](validation.md) · [讲义](../README.md) · [练习](../exercises.md)
