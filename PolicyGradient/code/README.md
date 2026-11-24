# 策略梯度编程实验

本目录用表格 softmax 实现 REINFORCE，直接展示 logits、策略概率与梯度。**Python 3.10+ 标准库即可运行**，不依赖 PyTorch、NumPy 或 Gymnasium。所有命令在仓库根目录执行。

## 1. 快速运行

```bash
# 默认：宝箱、历史状态基线、10,000 回合
python PolicyGradient/code/reinforce.py

# 随机单步任务
python PolicyGradient/code/reinforce.py --env bandit

# 无基线版本
python PolicyGradient/code/reinforce.py --env treasure --baseline none

# 低折扣对照
python PolicyGradient/code/reinforce.py --env treasure --gamma 0.3

# 帮助
python PolicyGradient/code/reinforce.py --help
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--env` | `treasure` | 另可选 `bandit` |
| `--episodes` | 10000 | 训练完整回合数 |
| `--batch-size` | 20 | 同一冻结策略采样的回合数 |
| `--lr` | 0.1 | 梯度上升步长 |
| `--gamma` | 0.9 | 有限回合折扣，允许 `[0,1]` |
| `--baseline` | `history` | 仅历史批次的状态均值；`none` 为零基线 |
| `--seed` | 0 | 训练随机种子 |
| `--eval-episodes` | 5000 | 每种独立评价的回合数 |
| `--max-steps` | 10 | 保护上限，超出会拒绝更新并报错 |
| `--log-every` | 10 | 每多少次参数更新记录一次 |
| `--output` | 自动目录 | 可指定独立结果目录 |

训练总更新次数为 $\lceil M/B\rceil$，最后一批不足 B 时除以实际回合数。本实现没有回报标准化、熵正则、梯度裁剪或重放旧数据；代码对应讲义的原始估计器。

## 2. 单步随机奖励实验

只有一个状态，动作 0 成功概率 0.2，动作 1 成功概率 0.8；成功奖励 1，失败 0，一步结束。

$$
J=0.2\pi(0)+0.8\pi(1)=0.2+0.6\pi(1).
$$

初始 logits 全零，对应均匀策略，J=0.5；最优动作 1 对应 J=0.8。`policy_probabilities[0]` 的顺序就是动作 0、1。

即使策略完全选择优动作，回报也有随机性，标准差为 0.4。评价均值接近 0.8 但不必精确相等；这可用于区分策略随机性与环境随机性。

## 3. 多步宝箱实验

| 状态 | 动作 0 | 动作 1 |
| --- | --- | --- |
| 0：起点 | 金币 +3，结束 | 山洞 −1，到状态 1 |
| 1：山洞 | 宝箱 +8，结束 | 陷阱 −4，结束 |

令 p 为起点山洞概率、q 为山洞宝箱概率：

$$
J=3(1-p)+p[-1+\gamma(12q-4)].
$$

输出概率的第一行是 `[金币,山洞]`，第二行是 `[宝箱,陷阱]`。高折扣最优方向为山洞、宝箱，最优折扣回报 6.2，不折扣奖励和 7。低折扣 0.3 下应偏向金币，最优回报 3。

采样梯度不使用已知奖励概率或解析 J。解析模型仅在训练后的记录、评价对照与导数检查中调用。策略训练仍由真实抽样得到的动作和奖励驱动。

## 4. 历史基线与梯度时序

每批开始先冻结 logits 与历史状态基线：

1. 采样完整一批，期间不更新参数。
2. 反向计算每个时刻的 G。
3. 用冻结的基线构造优势，并使用采样时保存的概率算分数梯度。
4. 按回合数平均，进行一次 actor 梯度上升。
5. 最后把本批回报加入历史状态均值，下一批才能使用。

历史基线统计所有访问，不按首次访问筛选；在一般轨迹中，策略梯度本身也要累计所有决策时刻。基线不是用来替代回报的 Q 目标，而是从回报中减去一个合法参照。

当历史数据来自不同策略，均值不必精确等于当前 V；其是否降低方差需要实测。不能在同一批次里先给每个样本取 b=自己的 G，否则优势恒为零。

## 5. 记录、独立评价与导数检查

默认输出在已忽略的 `runs/pg/<环境>_<基线>_seed<种子>_gamma<折扣>/`：

| 文件或字段 | 含义 |
| --- | --- |
| `config.json` | 全部配置 |
| `learning.csv` | 更新次数、已用回合数、解析 J、本批回报、梯度范数、起点动作 1 概率 |
| `result.json` → `policy_probabilities` | 学到的原随机策略 |
| `baseline_values`、`baseline_counts` | 历史状态基线及样本计数；无基线时为 `null` |
| `analytic_objective` | 最终随机策略的精确 J，仅作对照 |
| `analytic_optimal_value` | 玩具模型的最优值基准 |
| `stochastic_evaluation` | 原随机策略的独立评价，种子为训练种子+10000 |
| `greedy_diagnostic` | 概率最大动作策略的独立诊断，种子为训练种子+20000 |
| `gradient_check` | 固定测试 logits 下的解析导数与中心有限差分结果 |

评价不更新 actor 或基线。并列贪心动作均匀抽取。`action_counts` 可看每个状态访问与动作频率；标准差使用总体标准差。`return_mean` 为不折扣奖励和，`discounted_return_mean` 与目标 J 的定义一致。

有限差分使用步长 1e-6、测试 logits 每行为 `[0.2,-0.3]`，并非对有噪声的采样训练目标直接差分；机制测试还枚举全部轨迹，检查采样估计的期望是否等于解析导数。二者一起核对数学与实现。

同名目录重复运行会覆盖，学习率、回合数和批量大小不在目录名中；比较超参数请指定不同 `--output`。保存的是最终参数与记录，程序未提供继续训练加载入口。

## 6. 固定策略的方差实验

```bash
python PolicyGradient/code/variance.py --env treasure --seed 0 --batches 1000 --batch-size 20
python PolicyGradient/code/variance.py --env bandit --seed 0 --batches 1000 --batch-size 20
```

程序固定均匀策略，为每批轨迹同时计算无基线和**已知解析状态价值基线**的梯度，不进行任何学习。宝箱解析基线 `[1.9,2]`、单步任务 `[0.5]`，与当前抽样动作无关。

这是诊断实验，不能声称实际训练已经拥有解析 V；`reinforce.py` 的 `history` 基线仍只用历史样本。输出各坐标平均梯度、均值与真梯度的差、坐标样本方差之和（每坐标用 n−1 分母），以及对真梯度的均方误差。相同轨迹配对比较，减少数据差异带来的干扰。

默认写入 `runs/pg/variance_<环境>_seed<种子>.json`，折扣和批量配置不在文件名中，比较时请另设 `--output`。

## 7. 三个种子的完整实验

PowerShell：

```powershell
foreach ($seed in 0..2) {
    foreach ($envName in @('bandit', 'treasure')) {
        foreach ($baselineName in @('none', 'history')) {
            python PolicyGradient/code/reinforce.py --env $envName --baseline $baselineName --seed $seed
        }
    }
    python PolicyGradient/code/variance.py --env treasure --seed $seed
}
python PolicyGradient/code/reinforce.py --env treasure --gamma 0.3 --seed 0
```

Bash：

```bash
for seed in 0 1 2; do
  for env in bandit treasure; do
    for baseline in none history; do
      python PolicyGradient/code/reinforce.py --env "$env" --baseline "$baseline" --seed "$seed"
    done
  done
  python PolicyGradient/code/variance.py --env treasure --seed "$seed"
done
python PolicyGradient/code/reinforce.py --env treasure --gamma 0.3 --seed 0
```

## 8. 对照代码与公式

| 数学含义 | 函数或位置 |
| --- | --- |
| 稳定 softmax | `softmax`，先减最大值 |
| 完整同策略轨迹 | `sample_episode`，保存每步冻结概率 |
| $G_t=R_{t+1}+\gamma G_{t+1}$ | `returns` |
| $\gamma^t(G_t-b)(one\_hot-\pi)$ | `batch_gradient` |
| 梯度上升 | `update_logits`，使用加号 |
| 冻结历史基线 | `HistoricalBaseline` 与 `train` 中更新顺序 |
| $J(\theta)$ 与精确梯度 | `analytic_objective`、`analytic_gradient`，仅作诊断 |
| 中心有限差分 | `gradient_check` |
| 原策略与贪心化分别评价 | `evaluate` 的 `greedy` 参数 |

有限离散动作数为 A、批量 B、最大轨迹长 T 时，采样和梯度约需 $O(BTA)$ 工作，轨迹保存每步概率约需 $O(BTA)$ 空间；策略表为 $O(|S|A)$。本实现 A=2；日志空间取决于检查点数量。

## 9. 测试与报告

```bash
python -m unittest discover -s PolicyGradient/code -p test_reinforce.py -v
```

测试覆盖 softmax 数值、得分导数、折扣、批量归一化、梯度上升、轨迹枚举、基线独立性与冻结时序、有限差分、不完整回合、末批、评价隔离和参数校验。

报告应写明目标 J、折扣、批量大小、学习率、基线来源、种子、预算、概率、解析 J 和随机策略评价。比较基线时不要只看最终一次代理损失或某一条贪心轨迹。

[实测记录](validation.md) · [返回讲义](../README.md) · [练习](../exercises.md) · [答案](../answers.md)
