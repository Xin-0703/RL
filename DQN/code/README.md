# DQN 编程实验

本目录使用 PyTorch 实现标准 DQN，包含在线网络、目标网络、均匀经验回放、$\epsilon$-greedy、批量 TD 损失、梯度裁剪、独立评估和模型保存。默认 CPU 运行，无需 GPU。

## 1. 安装

建议 Python 3.10 或更新版本。在仓库根目录执行：

```bash
python -m venv .venv
```

激活虚拟环境：

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

安装全部实验依赖：

```bash
python -m pip install -r DQN/code/requirements.txt
```

如果只做宝箱实验和机制测试，安装 `numpy`、`torch` 即可；Gymnasium 只在运行 CartPole 时导入。若平台无法从默认源安装 PyTorch，可按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择适合的安装命令。

## 2. 实验一：宝箱环境

```bash
python DQN/code/train.py --env treasure --seed 0
```

默认训练 600 回合，$\gamma=0.9$。两个输入是状态独热向量，两个输出对应当前状态的两个动作：

| 状态 | 动作 0 | 动作 1 |
| --- | --- | --- |
| 起点 $S_0$ | 拿金币，+3，结束 | 进山洞，−1，到 $S_1$ |
| 山洞 $S_1$ | 开宝箱，+8，结束 | 踩陷阱，−4，结束 |

理论 Q 值为起点 `[3, 6.2]`、山洞 `[8, −4]`。默认设置应学会“进山洞 → 开宝箱”。实际 Q 值由训练产生，不在网络内写入答案。

查看最终输出：

- `return_mean`：不折扣的奖励和，最优路径为 $-1+8=7$。
- `discounted_return_mean`：从初始状态算起的折扣回报，最优路径为 $-1+0.9\times8=6.2$。
- `q_values`：网络对起点和山洞两个状态的动作价值预测。
- `first_episode_trace`：第一条贪心评估轨迹，动作应依次为 1、0。

改变折扣因子：

```bash
python DQN/code/train.py --env treasure --gamma 0.3 --seed 0 --output runs/treasure_gamma03
```

此时进山洞的理论价值为 1.4，低于金币的 3，正确的贪心策略变为拿金币。

## 3. 实验二：CartPole

```bash
python DQN/code/train.py --env cartpole --seed 0 --output runs/cartpole_seed0
```

环境为 `CartPole-v1`，状态是 4 维连续向量，动作是向左或向右推动。默认训练 400 回合，$\gamma=0.99$，保留环境的时间限制，不打开图形界面。训练效果受种子、训练预算和依赖版本影响，不承诺固定分数。

快速检查安装和运行流程可缩短回合数；这不代表已完成学习：

```bash
python DQN/code/train.py --env cartpole --episodes 5 --eval-episodes 2 --output runs/cartpole_smoke
```

默认 1,000 条经验后才开始训练，因此极短运行可能完全没有梯度更新；应当检查 `gradient_updates`。正式学习使用默认预算或在报告中说明调整。

## 4. 输出与重新加载

默认输出目录为 `runs/<环境>_seed<种子>/`，已加入 Git 忽略。相同目录再次运行会覆盖此前的日志与模型；比较实验应使用不同的 `--output`。

| 文件 | 内容 |
| --- | --- |
| `training.csv` | 各回合的交互步数、训练次数、探索概率、两种回报与损失 |
| `config.json` | 完整超参数 |
| `evaluation.json` | 独立贪心评估结果、首条轨迹；宝箱另含 Q 值 |
| `model.pt` | 在线网络权重、网络维度和训练配置 |

重新加载评估：

```bash
python DQN/code/train.py --evaluate runs/treasure_seed0/model.pt
python DQN/code/train.py --evaluate runs/cartpole_seed0/model.pt --eval-episodes 20
```

评估模式从模型恢复环境、折扣因子和种子；训练参数 `--env`、`--seed`、`--output` 不用于覆盖模型配置，`--episodes` 和 `--gamma` 不允许在该模式传入。需要调整评价回合数时使用 `--eval-episodes`。

模型是推理检查点，不包含恢复训练需要的优化器、回放池与随机数状态。代码未提供断点续训功能。

评价总是使用独立环境、$\epsilon=0$ 和保存种子加 10,000 后的初始种子序列。均值和标准差来自评价回合，标准差采用 `ddof=0`。默认不在训练中插入评估，以免混淆训练预算。

## 5. 对照公式阅读代码

| 课程知识 | 对应实现 |
| --- | --- |
| $Q(s,\cdot;\theta)$ | `QNetwork`：64 → 64 的两层隐藏层，线性输出 |
| 经验池 $\mathcal D$ | `ReplayBuffer`，复制观测并均匀无放回抽样 |
| $\epsilon$-greedy | `epsilon_at` 与 `select_action` |
| $y=r+\gamma(1-d)\max Q^-$ | `compute_targets`，使用 `torch.no_grad()` |
| 实际动作的在线预测 | `gather(1, actions.unsqueeze(1)).squeeze(1)` |
| $L=\frac1{2B}\sum(y-q)^2$ | `optimize` 中的平方损失 |
| 参数学习 | `zero_grad → backward → clip_grad_norm_ → step` |
| 目标同步 | `updates % target_every == 0` 后复制参数 |
| 终止与截断 | `train` 只向经验池存 `terminated`；两者均结束回合 |

本实现使用普通 Python 环境循环，没有自动重置与向量环境。若改成这类环境，要确认保存的是重置前的最终观测。对真正终止状态，返回的观测形状仍须有效，其价值由掩码屏蔽。

## 6. 检查关键机制

在仓库根目录执行：

```bash
python -m unittest discover -s DQN/code -p test_dqn.py -v
```

测试覆盖终止屏蔽、外部时间截断保留自举、目标停止梯度、选中动作的批量梯度、训练时目标参数不变、回放数据复制及容量、宝箱奖励和回合边界，以及保存模型所需的环境维度类型。测试使用确定性小网络验证公式，不以某个随机训练分数作为单元测试。

已执行的运行版本和实测结果见 [验证记录](validation.md)。

## 7. 实验报告建议

记录依赖版本、训练配置、每个种子的训练步数、贪心评估均值和标准差。至少比较 3 个种子，区分普通奖励和与折扣回报。宝箱可以对照解析最优值；CartPole 应根据实际评估报告结果。

[返回课程讲义](../README.md) · [完成课后练习](../exercises.md)
