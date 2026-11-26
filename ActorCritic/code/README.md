# Actor-Critic 编程实验

Python 3.10+ **标准库即可运行**，无需额外安装依赖。使用表格 softmax Actor、表格 V Critic、同策略冻结批次；默认一步 TD，另提供完整 MC 回报减学习基线对照。所有命令从仓库根目录执行。

## 1. 快速运行

```bash
python ActorCritic/code/actor_critic.py
python ActorCritic/code/actor_critic.py --env bandit
python ActorCritic/code/actor_critic.py --estimator mc
python ActorCritic/code/actor_critic.py --gamma 0.3
python ActorCritic/code/actor_critic.py --help
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--env` | `treasure` | 宝箱，另可选单步随机任务 `bandit` |
| `--estimator` | `td0` | 一步 TD；`mc` 使用完整回报 |
| `--episodes` | 10000 | 回合预算 |
| `--batch-size` | 20 | 同一冻结策略采样的回合数 |
| `--actor-lr` | 0.1 | Actor 梯度上升步长 |
| `--critic-lr` | 0.2 | Critic 半梯度步长 |
| `--gamma` | 0.9 | 有限回合折扣，允许 `[0,1]` |
| `--seed` | 0 | 训练局部随机数种子 |
| `--eval-episodes` | 5000 | 每种独立评价的回合数 |
| `--log-every` | 10 | 每多少次批量更新记录一次 |
| `--output` | 自动 | 指定独立输出目录 |

输出目录默认 `runs/ac/<环境>_<估计器>_seed<种子>_gamma<折扣>/`，已由仓库 `.gitignore` 忽略。目录名不包含学习率、预算、批量大小，比较这些参数时必须另设 `--output`；同名运行会覆盖。

## 2. 两个环境与理论参照

宝箱环境与前章一致：

| 状态 | 动作 0 | 动作 1 |
| --- | --- | --- |
| 起点 0 | 金币，+3，终止 | 山洞，−1，到状态 1 |
| 山洞 1 | 宝箱，+8，终止 | 陷阱，−4，终止 |

起点选山洞的概率记作 p，山洞选宝箱的概率记作 q：

$$
V^\pi(S_1)=12q-4,\qquad J=V^\pi(S_0)=3(1-p)+p[-1+\gamma(12q-4)].
$$

均匀策略、$\gamma=0.9$ 时真实 V 为 `[1.9,2]`；最优策略 V 为 `[6.2,8]`。低折扣 0.3 时金币最优，起点价值为 3。

bandit 只有一个状态、一步终止：动作 0 的奖励 1 成功率为 0.2，动作 1 为 0.8，否则奖励 0。策略价值 $V^\pi=0.2\pi(0)+0.8\pi(1)$，最优值 0.8。单步任务中 TD 与 MC 的目标都等于当前奖励，**相同种子、配置下两者应产生相同训练参数**。

训练只使用抽样经验，不将解析 V 或最优值写入 Actor/Critic。`analytic_values` 只用于日志和最终诊断。

## 3. 更新时序与公式对照

每批先用当前 Actor 收集 B 个完整回合，期间不更新任何参数。随后用旧 V 算完本批全部方向，再一起应用更新：

| 数学意义 | 函数 |
| --- | --- |
| 稳定 softmax | `softmax` |
| 同策略完整回合，保存采样概率 | `sample_episode` |
| $y=r+\gamma(1-d)V(s')$ | `td_target` |
| MC 回报 $G_t$ | `mc_returns` |
| $\sum\gamma^t\delta(one\_hot-\pi)/B$ | `batch_directions` 的 Actor 方向 |
| $\sum\delta\,one\_hot(state)/B$ | `batch_directions` 的 Critic 方向 |
| 两个参数应用旧方向 | `apply_updates` |
| 当前策略解析 V，仅诊断 | `analytic_values` |
| 独立随机评价与贪心诊断 | `evaluate` |

默认 10,000 回合、每批 20 回合，共 500 次更新；末批按实际回合数平均。Actor 保留时刻折扣 $\gamma^t$；Critic 价值拟合不乘它。程序用显式导数，因此“停止梯度”体现为只把旧目标、误差数值作为方向权重，不沿下一 V 或 Actor 权重再次求导。

程序没有回放、目标网络、熵奖励、优势标准化或梯度裁剪。它不是完整 A2C、A3C 或 PPO，也不是神经网络实现；神经网络损失路径见 [讲义第 4 讲](../README.md#第-4-讲actor-怎样根据评价改变策略)。

## 4. 记录与评价

| 文件/字段 | 含义 |
| --- | --- |
| `config.json` | 全部运行参数 |
| `learning.csv` | 已用回合、更新次数、解析 J、旧批 Actor/价值损失、更新后的 Critic 误差和起点动作 1 概率 |
| `result.json` → `logits` | 最终 Actor 参数 |
| `policy_probabilities` | 学到的原随机策略 |
| `critic_values` | 学到的状态 V |
| `analytic_policy_values` | **同一最终策略**的解析 V |
| `analytic_objective` | 最终策略的精确 J |
| `analytic_optimal_value` | 玩具模型最优 J 的参照 |
| `stochastic_evaluation` | 按原随机策略独立评价，种子=训练种子+10000 |
| `greedy_diagnostic` | 概率最大动作策略的独立诊断，种子=训练种子+20000 |

评价不改变参数，也不消耗训练 RNG 状态。贪心概率并列时均匀选择。回报标准差使用总体标准差（`ddof=0` 口径）。`return_mean` 为不折扣奖励和，`discounted_return_mean` 对应目标 J。

默认宝箱应主要选择“山洞 → 宝箱”，原随机策略 J 通常略低于 6.2；贪心诊断可达到折扣 6.2、普通回报 7。不要以贪心结果替代原随机策略结果。Critic 有限步误差和采样噪声是正常现象，应观察它是否跟随当前策略 V。

保存最终参数与日志，但没有加载继续训练或神经网络权重接口。若增加加载功能，必须保存所需配置，并区分推理加载与完整恢复训练的 RNG 等状态。

## 5. 三个种子与 MC 对照

PowerShell：

```powershell
foreach ($seed in 0..2) {
    foreach ($envName in @('treasure', 'bandit')) {
        foreach ($mode in @('td0', 'mc')) {
            python ActorCritic/code/actor_critic.py --env $envName --estimator $mode --seed $seed
        }
    }
}
python ActorCritic/code/actor_critic.py --gamma 0.3
python ActorCritic/code/actor_critic.py --critic-lr 0.01 --output runs/ac/slow_critic
```

Bash：

```bash
for seed in 0 1 2; do
  for env in treasure bandit; do
    for mode in td0 mc; do
      python ActorCritic/code/actor_critic.py --env "$env" --estimator "$mode" --seed "$seed"
    done
  done
done
python ActorCritic/code/actor_critic.py --gamma 0.3
python ActorCritic/code/actor_critic.py --critic-lr 0.01 --output runs/ac/slow_critic
```

MC 对照使用同一套 Actor、Critic 与批次时序，仅目标换为完整回报。训练过程中策略不同会使后续动作采样不同，因此不是每条轨迹都能一一配对；不能据此直接计算固定策略梯度方差。

## 6. 终止边界与实现范围

两个环境都最多两步自然终止，不存在外部时间截断；`Step` 和 `td_target` 保留 `terminated`/`truncated` 的区分，测试用合成转移核对截断自举。`mc_returns` 拒绝不完整自然终止轨迹。

若扩展环境，必须保存 reset 前的最后观测，外部截断停止采样但保留自举；MC 对照要决定如何补足尾部，不能把缺失回报默认为零。本程序没有通用 Gymnasium 环境循环或自动重置支持。

S 个状态、A 个动作、B 回合、长度上限 T 时，本实现采样和 Actor 计算约 $O(BTA)$，轨迹保存每步概率约 $O(BTA)$，参数表约 $O(SA+S)$；当前 A=2、T≤2。大量连续状态应改用函数逼近。

## 7. 测试

```bash
python -m unittest discover -s ActorCritic/code -p test_actor_critic.py -v
```

测试核对课堂数值、冻结时序、半梯度、时刻折扣、终止/截断、末批归一化、独立评价、确定性重复；还枚举全部宝箱轨迹，检查精确 Critic 时 TD 梯度期望与 MC、解析 J 有限差分一致，并验证错误 Critic 的偏差。

实验报告至少包含三个种子、相同预算、Actor 概率、Critic 与当前策略解析 V、随机评估和贪心诊断，勿仅比较一次代理损失。

[实测记录](validation.md) · [返回讲义](../README.md) · [练习](../exercises.md) · [答案](../answers.md)
