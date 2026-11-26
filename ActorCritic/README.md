# Actor-Critic：演员—评论家算法

> 课程主题：Actor 决定怎么行动，Critic 学习如何评价，二者通过优势估计合作。
>
> 建议学时：4 次 45 分钟理论课，另加 1 次 90 分钟实验。
>
> 先修知识：[MDP](../MDP/README.md)、[策略梯度](../PolicyGradient/README.md)、期望、梯度下降；建议对照 [DQN](../DQN/README.md) 理解 TD 自举。

## 课程资料与教学安排

| 资料 | 内容 |
| --- | --- |
| [本讲义](#第-1-讲为什么需要演员和评论家) | 动机、原理推导、更新公式、算法流程与方法比较 |
| [课堂例题](examples.md) | 6 道逐步计算题：价值、优势、两个参数更新、批量梯度、偏差和多步估计 |
| [实验说明](code/README.md) | 安装、运行、结果解释、公式对应 |
| [完整实现](code/actor_critic.py) | Python 标准库、表格 softmax、一步 TD 与 MC 对照 |
| [验证记录](code/validation.md) | 测试、精确期望核对与多种子实测 |
| [课后练习](exercises.md) | 16 道概念、计算、推导与编程题 |
| [参考答案](answers.md) | 逐题过程和实验验收标准 |

| 课时 | 教师展开 | 课堂活动 |
| --- | --- | --- |
| 1 | 回顾策略梯度；两种角色；V、Q、A | 计算同一策略下的状态价值与优势 |
| 2 | TD 自举；优势估计；Actor 和 Critic 梯度 | 手算一条经验的两个更新 |
| 3 | 在线算法；冻结批次；终止、截断和评价 | 写伪代码、检查旧参数的使用顺序 |
| 4 | 偏差与方差；n 步、GAE；A2C/A3C/PPO | 比较不同权重来源，完成推导题 |
| 实验 | 宝箱 30 分钟；单步任务 20 分钟；MC 对照 20 分钟；报告 20 分钟 | 三个种子、解析 V 核对、随机策略评价 |

学习目标：能解释 Actor-Critic 与 REINFORCE、DQN 的区别；从策略梯度推导 Actor 的更新；从贝尔曼期望方程推导 Critic 的 TD 目标；知道优势何时应当停止梯度；独立计算并实现一次批量更新。

---

## 第 1 讲：为什么需要演员和评论家？

### 1.1 从前两条路线出发

- DQN 学习最优动作价值 Q，根据 Q 值选择动作。
- REINFORCE 直接学习策略，用完整回合回报评价所选动作。
- Actor-Critic 直接学习策略，同时学习价值函数，用价值估计构造策略更新的权重。

演员（Actor）不是在执行固定剧本，而是参数化策略 $\pi_\theta(a\mid s)$。评论家（Critic）不是外部裁判，而是从奖励中学习的价值估计 $V_\phi(s)$。环境提供奖励，Critic 提供估计；二者不能混淆。

### 1.2 延续课堂宝箱环境

```text
起点 S₀
├─ 动作 0：拿金币 → +3，结束
└─ 动作 1：进山洞 → −1，进入 S₁
                               ├─ 动作 0：开宝箱 → +8，结束
                               └─ 动作 1：踩陷阱 → −4，结束
```

取 $\gamma=0.9$。如果策略总是进山洞、开宝箱，起点折扣回报为 6.2；如果山洞经常踩陷阱，进山洞未必好。

Actor 需要问：“进山洞是否比当前策略在起点的平均表现更好？”Critic 通过 $V^\pi$ 提供这个参照，并估计山洞未来能获得多少奖励。

### 1.3 三种输出各表示什么？

| 对象 | 典型输出 | 含义 |
| --- | --- | --- |
| Actor | 两个动作概率 | 接下来以多大概率选择各动作 |
| 状态 Critic | 一个标量 $V_\phi(s)$ | 当前策略从该状态出发的预计回报 |
| DQN 的 Q 网络 | 每个动作一个 Q 值 | 当前各动作的预计最优后续回报 |

本章 Actor 用 softmax；Critic 预测一个可正可负的数值，不用 softmax。Actor-Critic 是算法家族，Critic 也可以估计 Q；本课专讲 **基于 V 的离散动作同策略 Actor-Critic**，不等同于 DDPG、SAC 等离策略变体。

---

## 第 2 讲：从策略梯度到优势

### 2.1 先固定目标与折扣约定

沿用策略梯度章：从固定初始分布采样有限回合，优化

$$
J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^tR_{t+1}\right],
\qquad G_t=\sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1}.
$$

本章每回合都从真实起点开始，Actor 梯度显式保留 $\gamma^t$。一些实用实现将它吸收进状态分布，或采用不同的代理目标；不能未经说明把它删掉。代码允许有限回合 $0\leq\gamma\leq1$。

### 2.2 策略梯度定理

回顾轨迹对数导数与因果性，得到：

$$
\nabla J=\mathbb E\left[\sum_t\gamma^t Q^{\pi_\theta}(S_t,A_t)
\nabla_\theta\log\pi_\theta(A_t\mid S_t)\right].
$$

环境模型不依赖 $\theta$，动作按当前策略采样；无需对环境求导。若想复习轨迹概率推导，请看 [策略梯度第 2 讲](../PolicyGradient/README.md#第-2-讲策略梯度推导与-reinforce)。

### 2.3 减去状态基线

对动作采样时已固定、与本次动作无关的状态基线 $b(s)$：

$$
\mathbb E[b(s)\nabla\log\pi_\theta(A\mid s)\mid s]
=b(s)\sum_a\nabla\pi_\theta(a\mid s)=0.
$$

取 $b=V^\pi$，定义优势：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

因此：

$$
\boxed{\nabla J=\mathbb E\left[\sum_t\gamma^t A^\pi(S_t,A_t)
\nabla_\theta\log\pi_\theta(A_t\mid S_t)\right].}
$$

正优势表示这个动作比当前策略平均表现好，应当沿其分数梯度增加概率；负优势表示更差，应当降低概率。这是一次局部梯度方向，不保证任意学习率更新后真实 J 必然增加。

---

## 第 3 讲：Critic 怎样学习，TD 误差怎样评价动作？

### 3.1 Critic 评价的是当前策略

当前策略的贝尔曼期望方程是：

$$
V^\pi(s)=\mathbb E_{a\sim\pi,\,s',r}[r+\gamma(1-d)V^\pi(s')\mid s].
$$

$d=1$ 仅表示真正终止。**没有下一动作的 max**，因为 Critic 估计当前策略的 V，而不是直接求最优 Q。

### 3.2 一步 TD 目标与误差

对实际经验 $(s,a,r,s',d)$，使用更新前 Critic：

$$
y=r+\gamma(1-d)V_\phi(s'),\qquad
\boxed{\delta=y-V_\phi(s).}
$$

不需要等未来全部奖励出现，就能用后续价值估计构造目标，这就是自举。真正终止时 $y=r$。

### 3.3 为什么它可以作为优势估计？

假设策略固定且 Critic 精确等于 $V^\pi$：

$$
\begin{aligned}
\mathbb E[\delta\mid s,a]
&=\mathbb E[r+\gamma(1-d)V^\pi(s')\mid s,a]-V^\pi(s)\\
&=Q^\pi(s,a)-V^\pi(s)=A^\pi(s,a).
\end{aligned}
$$

所以一步 TD 误差在这个条件下是优势的条件无偏估计。单条经验的 $\delta$ 仍可能受随机奖励与转移影响，不等于精确优势。

若 $V_\phi=V^\pi+e$，则：

$$
\boxed{\mathbb E[\delta\mid s,a]
=A^\pi(s,a)+\gamma\mathbb E[(1-d)e(s')\mid s,a]-e(s).}
$$

当前状态误差 $-e(s)$ 是动作无关的基线项，其得分梯度期望为零；下一状态误差经过动作相关转移传回，**一般会使 Actor 梯度产生偏差**。这是 TD 自举与完整 MC 回报减基线的关键区别。

### 3.4 Critic 的半梯度更新

目标 $y$ 作为常数，最小化：

$$
L_C(\phi)=\frac12[\operatorname{stopgrad}(y)-V_\phi(s)]^2.
$$

于是：

$$
\boxed{\phi\leftarrow\phi+\alpha_C\delta\nabla_\phi V_\phi(s).}
$$

不沿 $V_\phi(s')$ 求导，故称半梯度 TD。它不是对完整贝尔曼残差平方求全梯度；也不能只因有平方损失就保证所有 TD 训练收敛。

表格 Critic 中每个状态一个独立参数：

$$
V(s)\leftarrow V(s)+\alpha_C\delta.
$$

Actor 改变策略后，Critic 的学习目标也随之改变。通常需要让价值估计足够及时，但“Critic 学习率越大越好”并不成立；太大也会振荡。

---

## 第 4 讲：Actor 怎样根据评价改变策略？

### 4.1 单条样本的更新

用 $\hat A_t=\delta_t$ 近似优势：

$$
\boxed{\theta\leftarrow\theta+
\alpha_A\gamma^t\delta_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).}
$$

采用最小化优化器时，Actor 的代理损失为：

$$
L_A(\theta)=-\gamma^t\operatorname{stopgrad}(\delta_t)
\log\pi_\theta(a_t\mid s_t).
$$

**负号**将梯度下降变为策略收益的梯度上升；**停止梯度**让优势只作为权重，不让 Actor 损失反向修改 Critic。

### 4.2 表格 softmax 便于手算

每状态两个 logits：

$$
\pi(a\mid s)=\frac{e^{z(s,a)}}{\sum_b e^{z(s,b)}},\qquad
\frac{\partial\log\pi(a\mid s)}{\partial z(s,b)}=\mathbf1\{b=a\}-\pi(b\mid s).
$$

因此：

$$
\boxed{z(s,b)\leftarrow z(s,b)+\alpha_A\gamma^t\delta_t
[\mathbf1\{b=a_t\}-\pi(b\mid s)].}
$$

两个 logits 一增一减，总和增量为零；真正决定概率的是它们的差。深层网络需要进一步应用链式法则，共享参数会同时影响其他状态的输出。

### 4.3 自动微分实现应有的梯度路径

神经网络版的单批关键语义可写为（代码片段不是本章标准库程序的依赖）：

```python
# states/next_states: [N,state_dim]；rewards/done/time_weights: [N]
with torch.no_grad():
    targets = rewards + gamma * (1 - done) * critic(next_states).squeeze(-1)
values = critic(states).squeeze(-1)
advantages = (targets - values).detach()
distribution = torch.distributions.Categorical(logits=actor(states))
actor_loss = -(time_weights * advantages * distribution.log_prob(actions)).sum() / B
critic_loss = 0.5 * (targets - values).square().sum() / B
# 独立 actor/critic 参数；分别 zero_grad、backward、step。
```

这里 N 是一批的总步数，B 是回合数。`actions` 为 `[N]` 整数索引。先用旧参数算好目标与优势，再更新两者。Actor 与 Critic 可以共享特征，但必须明确联合损失和梯度路径；本章使用独立参数。

---

## 第 5 讲：完整算法流程

### 5.1 在线一步 Actor-Critic

```text
初始化 Actor θ、Critic φ
对每回合：
    s ← reset；I ← 1
    重复：
        a ~ πθ(·|s)，保存采样时的 log 概率或得分梯度
        执行 a，得到 r、s′、terminated、truncated
        用旧 Critic 算 y = r + γ(1-terminated)Vφ(s′)
        用旧 Critic 算 δ = y - Vφ(s)
        从旧参数算好两个方向：
            actor_direction = I * δ * ∇θ log πθ(a|s)
            critic_direction = δ * ∇φ Vφ(s)
        θ += αA * actor_direction
        φ += αC * critic_direction
        I ← γI；s ← s′
        若 terminated 或 truncated：结束当前采样回合
```

这是可边交互边更新的基础形式。每次更新都改变后续采样策略，不能把整回合所有样本声称为来自完全相同的冻结策略；小步长下是常用的在线近似。

### 5.2 本章代码：冻结批次的一步 TD Actor-Critic

为与上一章衔接并便于严格核对，本章实现采用 **批量同策略采样**：

1. 冻结 Actor 和 Critic，采样 B 个完整短回合。
2. 用同一组旧 V 为每条经验计算一步 TD 目标与误差。
3. Actor 累积 $\sum_{i,t}\gamma^t\delta_{it}\nabla\log\pi_{it}/B$。
4. Critic 累积 $\sum_{i,t}\delta_{it}\nabla V(S_{it})/B$。
5. 同时应用已计算好的两个方向，丢弃该批数据。
6. 重复到回合预算结束；末批按实际回合数归一化。

对应批量损失：

$$
L_A=-\frac1B\sum_{i,t}\gamma^t\operatorname{stopgrad}(\delta_{it})\log\pi_\theta(a_{it}\mid s_{it}),
$$

$$
L_C=\frac1{2B}\sum_{i,t}[\operatorname{stopgrad}(y_{it})-V_\phi(s_{it})]^2.
$$

Critic 此处按访问状态拟合，并未乘 $\gamma^t$；Actor 则保留该因子。回合长度不等时不能把 B 随意改成总步数，再声称是同一个估计器。

本代码虽然等短回合收齐才更新，**权重来源仍是一步 TD，而不是完整回报**。选择 `--estimator mc` 才使用完整 $G_t$，成为学习状态基线的 REINFORCE 对照。

### 5.3 为什么这里没有经验回放和目标网络？

这个基础 AC 使用当前策略数据，一批只更新一次。直接混入旧策略数据会改变状态、动作分布，通常需要重要性采样等修正。不能把 DQN 的经验回放原样移植进来。

此处用 Critic 自己的旧值自举，但并未设置长期独立的目标网络；“计算时停止梯度”与“维护目标网络”是两件事。离策略 Actor-Critic 的其他变体可能同时使用回放和目标网络，不应从本课推断所有 AC 都禁止它们。

### 5.4 终止与时间截断

| 情况 | 是否结束当前采样 | 下一状态是否自举 |
| --- | --- | --- |
| 真正终止 | 是 | 否，$y=r$ |
| 外部时间限制截断 | 是 | 是，用重置前最后状态的 V |
| 普通转移 | 否 | 是 |

如果时间上限是任务定义本身，应作为真正终止，且状态通常要包含剩余时间。本课玩具环境最多两步自然结束，不产生截断；`td_target` 的接口保留截断语义并有机制测试。若扩展到截断环境，不能用 reset 后的初始观测当作上一回合的 $s'$。

---

## 第 6 讲：例题、评价与实验

### 6.1 均匀策略下 Critic 应学到什么？

两个状态都以 0.5 概率选动作，则：

$$
V^\pi(S_1)=0.5\times8+0.5\times(-4)=2,
$$

$$
V^\pi(S_0)=0.5\times3+0.5[-1+0.9\times2]=1.9.
$$

这不是最优策略的 `[6.2,8]`，而是**当前均匀策略**的 `[1.9,2]`。随着 Actor 改变，Critic 的正确答案也变化。

见 [课堂例题](examples.md)，逐步完成一次 Critic 更新和 Actor 更新，再核对优势条件期望。

### 6.2 实验顺序

```bash
python ActorCritic/code/actor_critic.py
python ActorCritic/code/actor_critic.py --env bandit
python ActorCritic/code/actor_critic.py --estimator mc
python ActorCritic/code/actor_critic.py --gamma 0.3
```

不需要额外依赖。先观察 Actor 动作概率，再比较 Critic 估计与同一最终策略的解析 V，最后查看独立随机策略评估。完整命令和参数见 [实验说明](code/README.md)。

### 6.3 正确解释结果

- 学到的是随机策略，主要评价应按该策略继续采样；argmax 后的贪心策略是另一项诊断。
- 宝箱最优普通奖励和为 7，$\gamma=0.9$ 的折扣回报为 6.2；不要混用口径。
- Critic 应与当前策略的 V 对照，不能无条件与最优 V 比较。
- 代理损失由样本和旧估计组成，Actor 损失为负或变小都不直接证明策略更好。
- 训练随机性与环境随机性不同：单步 bandit 的优动作奖励成功率为 0.8，即使动作固定，回报仍会波动。
- 同预算、多种子报告才能观察稳定性；少量成功运行不能证明一般收敛。

---

## 第 7 讲：偏差、方差与扩展

### 7.1 MC 与一步 TD 的取舍

| 权重来源 | 优势估计 | 特点 |
| --- | --- | --- |
| MC + 基线 | $G_t-V_\phi(s_t)$ | 需完整回报；合法冻结基线不改变策略梯度期望 |
| 一步 TD | $r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$ | 可提前更新；通常减少远期回报噪声，但不准确 Critic 可能引入偏差 |
| n 步 | n 步奖励 + 尾部 V − 当前 V | 在更长实际回报与更多自举之间取舍 |

“TD 方差一定更小”或“MC 一定更准确”都不是无条件结论；受奖励、转移、Critic 误差和样本预算影响。

### 7.2 n 步优势

未经过真正终止时：

$$
\hat A_t^{(n)}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}
+\gamma^nV_\phi(S_{t+n})-V_\phi(S_t).
$$

若先到真正终止，就在终止处停止求和且尾部为 0；若仅到采样边界，尾部使用最后观测的 V。对同一组冻结 V，沿途不终止时有望远镜恒等式：

$$
\hat A_t^{(n)}=\delta_t+\gamma\delta_{t+1}+\cdots+\gamma^{n-1}\delta_{t+n-1}.
$$

### 7.3 GAE 概念

GAE 用 $\lambda\in[0,1]$ 平衡不同长度的 TD 残差。对单条连续采样片段反向递推：

$$
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}.
$$

其中 $\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)$；$c_t$ 表示下一条残差是否属于同一条连续片段。真正终止、时间截断后重置或本次片段结束都要停止残差串接。**截断的 TD 目标可以自举，但不能把重置后新回合的残差串到上一回合。**

在完整自然终止回合、同一冻结 V 下：$\lambda=0$ 还原一步 TD；$\lambda=1$ 望远镜展开为 $G_t-V(s_t)$。截断片段的 $\lambda=1$ 则仍含尾部自举，不等于完整 MC。本章不实现 GAE，作为后续实验方向。

### 7.4 A2C、A3C、PPO 与连续动作

| 方法 | 与本课的关系 |
| --- | --- |
| 基础 Actor-Critic | 策略学习与价值学习结合的家族 |
| A2C | 通常同步收集多个环境的短 rollout，再批量更新优势 Actor-Critic |
| A3C | 多个工作者异步交互与更新共享参数 |
| PPO | 通常使用 Actor、Critic 和优势，但引入概率比和裁剪代理目标，可多轮使用一批数据 |
| DDPG / SAC | 离策略 AC，通常有 Q Critic、回放等机制，与本课流程不同 |

本章是**单进程、冻结批次、表格一步 TD AC**，不将其冒称为完整 A2C 或 A3C。连续动作可用高斯 Actor，Critic 仍可估计 V；还需正确处理标准差、动作边界和 log 密度。

### 7.5 可选的熵正则

为鼓励探索，有时在 Actor 最小化损失中加入 $-\beta H(\pi(\cdot\mid s))$，其中 $H=-\sum_a\pi(a)\log\pi(a)$。正的 $\beta$ 倾向提高熵，但会改变优化目标。代码默认没有熵正则、优势标准化、梯度裁剪或回放，便于逐式核对；有限预算仍可能探索不足。

## 算法与公式总结

| 内容 | 本课约定 |
| --- | --- |
| Actor | softmax 策略 $\pi_\theta(a\mid s)$ |
| Critic | 当前策略状态价值 $V_\phi(s)$ |
| 一步目标 | $y=r+\gamma(1-d)V_\phi(s')$ |
| 优势近似 | $\delta=y-V_\phi(s)$ |
| Actor 方向 | $\gamma^t\delta\nabla\log\pi$，优势停止梯度 |
| Critic 方向 | $\delta\nabla V$，目标停止梯度 |
| 批量分母 | 实际回合数 B |
| 更新时序 | 采样、目标与优势均基于旧参数，再应用两个方向 |
| 数据使用 | 同策略、一批一次，不无修正复用旧经验 |
| 核心限制 | 不准确尾部价值可能使 Actor 梯度有偏 |

完成 [例题](examples.md)，独立做 [练习](exercises.md)，再查看 [答案](answers.md)。

## 参考资料

1. Sutton & Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), 2nd ed., Chapters 6 and 13.
2. Mnih et al., [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), 2016.
3. Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), 2015.
4. Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.

[返回首页](../README.md) · [复习策略梯度](../PolicyGradient/README.md) · [对照 DQN](../DQN/README.md)
