# MDP 动态规划编程实验

使用 **Python 3.10+ 标准库**，无需安装第三方依赖。所有命令在仓库根目录执行。这里直接读取已知模型并计算概率期望，不训练神经网络，也不随机采样，所以相同配置的结果可直接复现，无需随机种子。

## 1. 快速开始

```bash
# 完成宝箱环境的全部算法
python MDP/code/mdp.py

# 持续型任务：基础 → 训练 → 工作
python MDP/code/mdp.py --env career

# 随机开箱任务
python MDP/code/mdp.py --env stochastic

# 帮助
python MDP/code/mdp.py --help
```

`--algorithm all` 默认执行固定均匀策略的两种评估、价值迭代、策略迭代、有限时域规划。它们求的目标不同，请按输出字段阅读。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--env` | `treasure` | 另可选 `career`、`stochastic` |
| `--algorithm` | `all` | 另可选 `evaluate`、`value`、`policy`、`finite` |
| `--gamma` | 0.9 | 无限时域要求 `[0,1)`；仅 `finite` 可取 1 |
| `--tolerance` | 1e-10 | 无限时域迭代的贝尔曼残差容差 |
| `--max-iterations` | 100000 | 迭代上限；未达到标准会报错 |
| `--horizon` | 3 | 有限时域允许的动作数，非负整数 |
| `--output` | 自动目录 | 自定义结果位置 |

`--horizon` 仅在 `all` / `finite` 的有限时域部分生效；`--tolerance` 和 `--max-iterations` 不改变有限时域的边界。线性方程评估直接求解，不使用迭代容差。

## 2. 模型表示与扩展

```python
model = {
    "s": {
        "a": [Outcome(0.75, "end", 8.0), Outcome(0.25, "end", -4.0)],
    },
    "end": {},
}
```

每个合法状态动作对拥有一个联合结果列表。结果包含概率、下一状态与奖励，允许相同下一状态对应不同奖励。终止状态字典为空，它没有动作，后续价值为 0；进入终点时的奖励照常计入。

`validate_model` 检查概率范围、每对概率和、下一状态是否存在、奖励是否有限。策略在每个非终止状态列出所有合法动作及概率（包括概率 0 的动作），不为终止状态列策略。

代码中的 `make_model` 提供三个教学模型，扩展时可以直接构造相同结构并调用各个算法函数。模型只支持有限状态、有限动作、有限个联合结果；未处理连续空间或未知动力学。

## 3. 宝箱与随机开箱

```bash
python MDP/code/mdp.py --env treasure --gamma 0.9
python MDP/code/mdp.py --env treasure --gamma 0.3
python MDP/code/mdp.py --env stochastic --gamma 0.9
```

宝箱模型与 [MC](../../MonteCarlo/code/README.md) 和 [DQN](../../DQN/code/README.md) 一致。随机版本仅将开箱改为 0.75 概率得 8、0.25 概率得 −4，两个结果都终止。

| 环境 / 折扣 | 均匀策略起点值 | 最优起点值 | 最优山洞值 | 起点最优动作 |
| --- | ---: | ---: | ---: | --- |
| 宝箱 / 0.9 | 1.9 | 6.2 | 8 | cave |
| 宝箱 / 0.3 | 1.3 | 3 | 8 | coin |
| 随机开箱 / 0.9 | 1.225 | 3.5 | 5 | cave |

`evaluation` 评价每个状态均匀选动作的固定策略，因此其值与 `value_iteration` / `policy_iteration` 的最优值不同。

修改随机开箱概率的独立实验，无需改变源文件：

```python
# 在仓库根目录进入 python 后逐行执行
import sys
sys.path.insert(0, "MDP/code")
from mdp import Outcome, make_model, value_iteration
model = make_model("stochastic")
model["cave"]["chest"] = [Outcome(0.7, "win", 8), Outcome(0.3, "lose", -4)]
result = value_iteration(model, gamma=0.9)
print(result["values"], result["actions"], result["q_values"])
```

p=0.7 时，山洞值 4.4，起点山洞动作价值 2.96，低于金币 3。该片段只修改当前内存里的模型；原课程设置仍为 p=0.75。

## 4. 持续型模型与有限时域

```bash
python MDP/code/mdp.py --env career --algorithm all --gamma 0.9
python MDP/code/mdp.py --env career --algorithm finite --horizon 1
python MDP/code/mdp.py --env career --algorithm finite --horizon 2
python MDP/code/mdp.py --env career --algorithm finite --horizon 3
python MDP/code/mdp.py --env career --algorithm finite --horizon 10
python MDP/code/mdp.py --env career --algorithm finite --horizon 3 --gamma 1
```

基础状态选择 rest 得 1 并留在原处，invest 得 0 并到熟练状态；熟练状态选择 work 得 2 并留在原处，reset 得 0 并回到基础。

$\gamma=0.9$ 时无限时域最优值为 `[18,20]`，策略 invest/work。均匀随机策略的值则为 `[7.25,7.75]`。

有限时域 `layers[h]` 对应“还剩 h 次动作”，边界 `layers[0]` 的价值全为 0、无动作。实际决策先用 H 层，随后 H−1 层，依次减少，而不是始终使用 H 层策略。

基础状态在 h=1、2 时选择 rest，在 h≥3 时选择 invest（默认折扣）。有限时域可取 $\gamma=1$；无限时域入口拒绝该值，以免对本持续奖励任务误用折扣收敛结论。

## 5. 输出与数值精度

默认保存到已被忽略的 `runs/mdp/<环境>_<算法>_gamma<折扣>_h<时域>/`：

| 文件或字段 | 内容 |
| --- | --- |
| `config.json` | 完整运行参数 |
| `result.json` → `model` | 实际采用的模型 |
| `evaluation.iterative` | 固定均匀策略的同步迭代评估、轮次、策略残差和误差界 |
| `evaluation.linear` | 同一策略的线性方程解、状态顺序、矩阵与奖励向量 |
| `value_iteration` | 最优价值近似、Q、贪心动作、轮次、最优残差与误差界 |
| `value_iteration.extracted_policy_values` | 另解最终贪心策略的价值，核对策略表现 |
| `policy_iteration` | 策略迭代结果、每轮评估与改进历史 |
| `finite_horizon.layers` | 从 0 到 H 的所有剩余步数价值与策略 |

相同目录再次运行会覆盖旧结果，容差不在默认目录名中；比较精度设置请用不同 `--output`。没有断点续算功能。

残差为 $\|TV-V\|_\infty$，值误差界为残差除以 $1-\gamma$。实际浮点计算有舍入误差，输出残差恰为 0 不代表数学上精确无误。代码的高斯消元仅用于小型教学系统，含主元选择与接近奇异检查；大型或病态系统应使用成熟线性代数库与误差分析。

策略迭代从每个状态的第一个合法动作开始；并列时按模型中动作顺序选择第一个最大值。价值迭代从 0 开始，全部状态同步更新。停止检查使用**更新后的 V 的实际残差**，不是只比较两次旧值的差。不同教材的停止标准或计数约定可能产生不同轮次。

## 6. 公式与代码对应

| 原理 | 对应函数 |
| --- | --- |
| $\sum p(s',r\mid s,a)[r+\gamma V(s')]$ | `action_values` |
| 按固定策略求动作均值 / 对动作求最大值 | `bellman_backup` 的 `policy` 参数 |
| 同步策略评估 | `iterative_evaluation` |
| $(I-\gamma P_\pi)V=r_\pi$ | `linear_evaluation`、`solve_linear` |
| 贝尔曼残差 | `bellman_residual` |
| 贪心策略改进 | `greedy_actions`、`deterministic_policy` |
| 评估与改进交替 | `policy_iteration` |
| 最优算子迭代 | `value_iteration` |
| $V_h$ 使用 $V_{h-1}$ | `finite_horizon` |

这些方法依赖已知模型中的完整期望。MC 从完整回报采样，DQN 从单步经验构造自举目标，不能把它们与这里的精确模型备份混为一谈。

## 7. 测试与报告

```bash
python -m unittest discover -s MDP/code -p test_mdp.py -v
```

15 项测试使用手算轮次和解析解验证：两种策略评估、矩阵终止处理、同步备份、两类控制算法、随机期望、持续型固定点与误差界、有限时域、策略改进、并列规则、求解器主元、模型验证、迭代上限和输入不变性。

已运行数据见 [验证记录](validation.md)。实验报告应包括模型表、求解目标、算法与初始条件、解析结果、残差、误差界及策略含义；本章不需要多随机种子，因为没有采样。

[返回讲义](../README.md) · [课后练习](../exercises.md) · [参考答案](../answers.md)
