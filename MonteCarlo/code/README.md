# 蒙特卡洛编程实验

本目录把讲义中的公式实现为 3 个可重复运行的实验。只需 **Python 3.10 或更新版本**，全部使用标准库，无需安装 NumPy、PyTorch 或 Gymnasium。所有命令都在仓库根目录执行。

## 1. 快速开始

```bash
# 首次访问 MC：固定策略的随机游走价值预测
python MonteCarlo/code/monte_carlo.py --task predict

# 首次访问 MC 控制：学习宝箱环境中的动作选择
python MonteCarlo/code/monte_carlo.py --task control

# 异策略 MC：用随机策略的数据评价固定目标策略
python MonteCarlo/code/monte_carlo.py --task offpolicy

# 查看全部参数
python MonteCarlo/code/monte_carlo.py --help
```

参数作用域会检查：`--visit` 只用于预测；`--epsilon`、`--eval-episodes` 只用于控制。控制固定采用首次访问状态动作对；异策略实验只估计固定起点的 V。

| 参数 | 预测 | 控制 | 异策略评估 |
| --- | --- | --- | --- |
| `--episodes` 默认值 | 10,000 | 50,000 | 10,000 |
| `--gamma` 默认值 | 1，当前实验仅支持此值 | 0.9，可调整 | 0.9，可调整 |
| `--seed` 默认值 | 0 | 0 | 0 |
| `--visit` | `first` 或 `every`；默认 `first` | 固定首次访问 | 固定起点 |
| `--epsilon` | 固定等概率左右移动 | 默认 0.1 | 行为策略固定等概率 |
| `--eval-episodes` | 不适用 | 默认 100 | 不适用 |
| `--max-steps` | 默认 10,000，异常保护 | 同左 | 同左 |
| `--log-every` | 默认约记录 100 个检查点 | 同左 | 同左 |
| `--output` | 可自定义输出目录 | 同左 | 同左 |

## 2. 实验一：随机游走策略评估

```text
左终点 0 ←→ 1 ←→ 2 ←→ 3（起点）←→ 4 ←→ 5 ←→ 6 右终点
```

每步以 0.5 的概率向左或向右；只有进入右终点的那一步奖励为 1，其余奖励为 0，两个终点都结束回合。固定 $\gamma=1$，每个决策状态的价值等于先到右终点的概率，解析值为：

$$
V^\pi(i)=i/6,\qquad i=1,2,3,4,5.
$$

比较两种访问规则：

```bash
python MonteCarlo/code/monte_carlo.py --task predict --visit first --seed 0
python MonteCarlo/code/monte_carlo.py --task predict --visit every --seed 0
```

输出中 `states` 对应 `values`、`counts` 和 `analytic_values` 的顺序。`rmse` 衡量五个状态的估计误差，公式见 [练习答案 14](../answers.md#14-随机游走实验参考)。未访问状态输出 `null`；若存在未覆盖状态，RMSE 也输出 `null`。

同种子下这两次运行生成相同轨迹，差别仅在回报样本如何选取。每次访问计数通常更大，但同回合样本相关，不能直接得出它一定更准确的结论。

预测命令只接受 $\gamma=1$，避免把折扣价值错误地与 $i/6$ 对比。若扩展到其他折扣，应同步修改解析基准。

## 3. 实验二：宝箱 MC 控制

| 状态 | 动作 0 | 动作 1 |
| --- | --- | --- |
| 起点 0 | 金币，+3，结束 | 山洞，−1，到状态 1 |
| 山洞 1 | 宝箱，+8，结束 | 陷阱，−4，结束 |

```bash
python MonteCarlo/code/monte_carlo.py --task control --gamma 0.9 --epsilon 0.1 --seed 0
python MonteCarlo/code/monte_carlo.py --task control --gamma 0.3 --epsilon 0.1 --seed 0
```

每回合结束后，按首次状态动作对的回报更新 Q；下一回合用更新后 Q 的 $\epsilon$-贪心策略采样。并列最大动作均匀分配贪心概率。

| 输出 | 阅读方式 |
| --- | --- |
| `q_values` | 两行对应起点与山洞，两列对应动作 0、1；未访问的对显示 `null` |
| `counts` | 与 Q 表对应的样本计数 |
| `greedy_action_sets` | 各状态的全部最大价值动作，可能并列 |
| `analytic_optimal_q` | 解析最优 Q，仅用于对照；未写入学习表 |
| `training_return_mean` | 全部训练回合的不折扣奖励和均值，包含探索成本 |
| `training_discounted_return_mean` | 全部训练回合的初始折扣回报均值 |
| `evaluation` | 独立环境和独立随机数、$\epsilon=0$ 下的评价结果 |

未知动作在选动作时采用初始化 Q=0，但结果用 `null` 标识未学习，避免误以为它的真实价值就是 0。输出的 `greedy_action_sets` 会考虑这些初始化值，阅读时需结合 `counts`。

排序正确且 $\epsilon=0.1$ 时，进入山洞后的期望奖励为 $0.95\times8+0.05\times(-4)=7.4$，所以所学山洞动作价值趋近 $-1+7.4\gamma$，并非最优值 $-1+8\gamma$。本实验的有限样本还混合了早期策略，需容许误差。

独立评价使用新环境、种子 `训练种子 + 10000`，不更新 Q。评价标准差采用总体标准差；本玩具环境在学到唯一贪心策略后是确定性的，标准差为 0 不代表一般任务没有不确定性。

| 折扣 | 预期贪心轨迹 | 不折扣回报 | 折扣回报 |
| --- | --- | ---: | ---: |
| 0.9 | 山洞 → 宝箱 | 7 | 6.2 |
| 0.3 | 金币 | 3 | 3 |

观察取消探索造成的失败：

```bash
python MonteCarlo/code/monte_carlo.py --task control --epsilon 0 --seed 1
```

实测该种子首先选金币，此后一直选金币，山洞相关计数为 0。随机打破初始并列不等于持续探索。

## 4. 实验三：异策略评估

```bash
python MonteCarlo/code/monte_carlo.py --task offpolicy --seed 0 --episodes 10000
```

行为策略 $b$ 每个状态均匀选动作；固定目标策略 $\pi$ 选山洞、宝箱。这里只估计固定起点的 $V^\pi(0)$，默认解析值为 6.2；不进行策略改进。

输出含 `ordinary`（普通 IS）、`weighted`（加权 IS）、`positive_weight_episodes`、`weight_sum` 和 `analytic_target_value`。若成功路径数为 K：

$$
C=4K,\qquad V_{\mathrm{ordinary}}=24.8K/N.
$$

默认 $\gamma=0.9$ 且 $K>0$ 时加权值恰好为 6.2；$K=0$ 时为 `null`。这是确定性路径的特殊性质，详见 [讲义第 4 讲](../README.md#第-4-讲异策略评估与重要性采样进阶)。

代码保存采样时实际使用的行为动作概率，不在事后用新策略重新计算旧概率。通用 `trajectory_weight` 辅助函数支持指定权重起始位置；实验入口只使用起点 V 的权重，未实现异策略 Q 学习或控制。

## 5. 文件输出与复现

默认写入仓库已忽略的 `runs/mc/`，目录名包含任务、种子、折扣，以及适用的访问规则或探索率。

| 文件 | 内容 |
| --- | --- |
| `config.json` | 完整参数与输出路径 |
| `result.json` | 最终估计、计数、解析对照；控制还含独立评价 |
| `learning.csv` | 检查点数据：预测价值/RMSE、控制累计训练回报、或两类 IS 估计 |

相同配置再次运行会覆盖同名输出；不同回合数不自动进入目录名，比较预算时请指定不同 `--output`。这些文件是实验记录，程序没有断点续训功能。日志默认始终包含第一个和最后一个回合，CSV 空单元格表示尚未定义或尚未覆盖。

按课后题一次运行三个种子（Windows PowerShell）：

```powershell
foreach ($seed in 0..2) {
    foreach ($visit in @('first', 'every')) {
        python MonteCarlo/code/monte_carlo.py --task predict --seed $seed --visit $visit --episodes 10000
    }
    foreach ($gamma in @('0.9', '0.3')) {
        python MonteCarlo/code/monte_carlo.py --task control --seed $seed --gamma $gamma --episodes 50000
    }
    python MonteCarlo/code/monte_carlo.py --task offpolicy --seed $seed --episodes 10000
}
```

macOS/Linux Bash：

```bash
for seed in 0 1 2; do
  for visit in first every; do
    python MonteCarlo/code/monte_carlo.py --task predict --seed "$seed" --visit "$visit" --episodes 10000
  done
  for gamma in 0.9 0.3; do
    python MonteCarlo/code/monte_carlo.py --task control --seed "$seed" --gamma "$gamma" --episodes 50000
  done
  python MonteCarlo/code/monte_carlo.py --task offpolicy --seed "$seed" --episodes 10000
done
```

## 6. 对照公式阅读实现

| 知识点 | 函数或类 | 关键行为 |
| --- | --- | --- |
| 完整回合 | `sample_episode`、`Episode` | 只有自然终止才返回可学习轨迹 |
| $G_t=R_{t+1}+\gamma G_{t+1}$ | `discounted_returns` | 从后往前计算 |
| 首次访问/每次访问 | `update_episode` | 按正向时间筛选；每回合清空集合 |
| 样本均值增量更新 | `MeanTable.update` | 计数先增加，步长取计数倒数 |
| Q 的访问键 | `update_episode(action_value=True)` | 使用 `(state, action)` |
| 并列 ε-贪心 | `epsilon_greedy_probabilities` | 均匀分配并列贪心概率 |
| 同策略控制 | `control` | 完整回合生成后才更新 Q |
| 独立评价 | `evaluate_treasure` | 新环境、新随机数、无训练更新 |
| $\prod\pi/b$ | `trajectory_weight` | 使用轨迹中保存的行为概率 |
| 普通/加权 IS | `ImportanceMean.update` | 普通计数包括零权重；总权重为 0 时加权值未定义 |

随机游走轨迹和回报占用 $O(T)$ 临时空间，处理一个回合为 $O(T)$；表格额外占用 $O(|\mathcal S|)$ 或 $O(|\mathcal S||\mathcal A|)$ 空间。日志占用的内存由检查点数量决定。这些结论针对本实验有限动作集。

## 7. 机制测试与实验报告

```bash
python -m unittest discover -s MonteCarlo/code -p test_monte_carlo.py -v
```

测试用确定性轨迹及枚举的完整路径核对数学恒等式，包括首次访问陷阱、重复状态不同动作、零权重、V/Q 权重边界、环境终止、截断保护、独立评价与参数限制。多种子训练结果另记在 [实测记录](validation.md)，不使用某个随机训练分数作为单元测试标准。

报告须包含任务、访问规则、种子、回合数、折扣、探索率、覆盖计数、解析基准、估计误差和独立评价。分析异常前先看覆盖，再看实现和统计误差。

[返回讲义](../README.md) · [课后练习](../exercises.md) · [参考答案](../answers.md)
