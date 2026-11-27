# TRPO：信赖域策略优化

> 课程主题：沿着提高收益的方向更新策略，同时用 KL 散度控制策略变化。
>
> 先修：[策略梯度](../PolicyGradient/README.md)、[Actor-Critic](../ActorCritic/README.md)、泰勒展开、矩阵与二次型。
>
> 建议学时：4 次 45 分钟理论课，另加 1 次 90 分钟实验。

## 资料与学习目标

| 资料 | 内容 |
| --- | --- |
| 本讲义 | 性能差异引理、代理目标、KL 约束、自然梯度、共轭梯度与线搜索 |
| [课堂例题](examples.md) | 6 道完整例题：概率比、单参数更新、二维更新、回溯、状态分布、GAE |
| [实验说明](code/README.md) | 无依赖运行、公式与函数对应、实验范围 |
| [数值实现](code/trpo_demo.py) | 精确期望上下文 bandit，Fisher 向量积、CG 和回溯 |
| [验证记录](code/validation.md) | 公式检查、数值测试和运行结果 |
| [课后练习](exercises.md) · [答案](answers.md) | 10 道概念、计算、推导与实验题 |

学习后应能从性能差异推导代理目标，从 KL 的二阶展开推导更新步长，手算一次 TRPO 更新，并解释实用 TRPO 为什么没有逐次真实回报必增的保证。

| 课时 | 内容 | 活动 |
| --- | --- | --- |
| 1 | 大步更新的风险、性能差异、旧状态分布近似 | 例题 1、5 |
| 2 | KL、Fisher、自然梯度、约束优化 | 例题 2、3 |
| 3 | Fisher 向量积、CG、阻尼、回溯 | 例题 4、运行实验 |
| 4 | Actor-Critic、GAE、采样误差、PPO 对比 | 例题 6、练习与讨论 |

## 第 1 讲：为什么普通策略梯度还需要约束？

Actor-Critic 给出方向 $g$，常见更新为 $\theta' = \theta+\alpha g$。梯度只描述局部变化；同一学习率在不同状态、参数尺度与训练阶段，可能造成完全不同的动作概率变化。若更新破坏了原有好策略，下一批轨迹质量会明显下降。

TRPO（Trust Region Policy Optimization）用“新旧动作分布差多少”衡量步长。直观上：目标函数决定往哪里走，KL 预算决定允许走多远。

### 1.1 先固定数学约定

本讲理论采用固定初始分布、无限折扣目标，$0\le\gamma<1$：

$$
J(\pi)=\mathbb E_\pi\left[\sum_{t=0}^{\infty}\gamma^tR_{t+1}\right],\qquad
d_\pi(s)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t\Pr_\pi(S_t=s).
$$

$d_\pi$ 是归一化的折扣状态分布。有限回合可补零奖励吸收状态；吸收状态的优势和 KL 贡献均为零。不能将这里的 $d_\pi$ 与任意长度轨迹直接按步平均得到的状态分布视为完全相同。

用 $\pi_0=\pi_{\theta_0}$ 表示本次冻结的旧策略，$A_0=Q^{\pi_0}-V^{\pi_0}$ 表示旧优势。候选策略改变时，旧策略、旧优势与采样数据均保持固定。

### 1.2 性能差异引理

沿新策略轨迹展开旧优势：

$$
\begin{aligned}
\mathbb E_{\pi'}\left[\sum_t\gamma^t A_0(S_t,A_t)\right]
&=\mathbb E_{\pi'}\left[\sum_t\gamma^t(R_{t+1}+\gamma V^{\pi_0}(S_{t+1})-V^{\pi_0}(S_t))\right]\\
&=J(\pi')-J(\pi_0).
\end{aligned}
$$

中间价值项望远镜消去；假设价值有界，尾部极限为零。于是

$$
\boxed{J(\pi')-J(\pi_0)=\frac1{1-\gamma}\mathbb E_{s\sim d_{\pi'},a\sim\pi'}[A_0(s,a)].}
$$

困难在于右边使用**新策略的状态分布**，而我们手里只有旧策略采样的数据。

### 1.3 代理目标与动作重要性采样

先将 $d_{\pi'}$ 近似替换为 $d_{\pi_0}$，定义

$$
L_0(\pi')=J(\pi_0)+\frac1{1-\gamma}\mathbb E_{s\sim d_{\pi_0},a\sim\pi'}[A_0(s,a)].
$$

若旧策略覆盖新策略会使用的动作，则

$$
\mathbb E_{a\sim\pi'} A_0
=\mathbb E_{a\sim\pi_0}\left[\frac{\pi'(a\mid s)}{\pi_0(a\mid s)}A_0\right].
$$

忽略与候选参数无关的常数及正比例因子，可最大化

$$
\boxed{S(\theta)=\mathbb E_{d_{\pi_0},\pi_0}[r_\theta(s,a)A_0(s,a)],\qquad
r_\theta=\frac{\pi_\theta(a\mid s)}{\pi_0(a\mid s)}.}
$$

精确优势下 $S(\theta_0)=0$，并且 $\nabla S(\theta_0)=(1-\gamma)\nabla J(\theta_0)$。重要性比率只修正动作分布，**没有修正状态分布**。候选策略离旧策略太远时，代理目标可能不再可靠。

采样实现把 $A_0$ 换成冻结的 $\hat A$；有限样本下其均值未必为零，所以线搜索应检查 $\hat S(\theta')-\hat S(\theta_0)$。

## 第 2 讲：KL 约束与理论保证的边界

### 2.1 实用约束

本章统一使用旧到新的方向：

$$
K(\theta)=\mathbb E_{s\sim d_{\pi_0}}\left[D_{\mathrm{KL}}(\pi_0(\cdot\mid s)\|\pi_\theta(\cdot\mid s))\right],
$$

$$
\boxed{\max_\theta S(\theta)\quad\text{s.t.}\quad K(\theta)\le\delta.}
$$

离散动作下，状态 KL 为 $\sum_a\pi_0(a\mid s)\log[\pi_0(a\mid s)/\pi_\theta(a\mid s)]$。对每个采样状态，通常对所有动作求和，或使用分布库的解析 KL。只保存被执行动作的旧 log probability，足以算比率，却不总足以计算完整分布 KL。

KL 不对称；反向 KL 的远处数值可能不同。两种方向在相等分布处的二阶局部度量相同，但实现必须固定方向。

### 2.2 为什么限制分布差异有帮助？

令 $\epsilon=\max_{s,a}|A_0(s,a)|$，$\alpha=\max_s D_{\mathrm{TV}}(\pi_0(\cdot\mid s),\pi'(\cdot\mid s))$。在有限/适当有界条件下，经典 TRPO 下界为

$$
J(\pi')\ge L_0(\pi')-\frac{4\gamma\epsilon}{(1-\gamma)^2}\alpha^2.
$$

Pinsker 不等式给出 $D_{\mathrm{TV}}^2\le D_{\mathrm{KL}}/2$（自然对数）。因此约束**所有状态上的最大 KL**可以控制这个误差项。

但实用算法使用采样状态上的**平均 KL**，再进行有限精度优化，优势也来自估计。平均值小不代表每个状态的 KL 都小；只增加代理目标也不代表增加上述下界。因此不能声称实用 TRPO 每次更新都保证真实 J 上升。例题 5 给出平均约束的盲点。

## 第 3 讲：从约束问题推导自然梯度更新

令 $x=\theta-\theta_0$。在旧参数处作近似：

$$
S(\theta_0+x)\approx S(\theta_0)+g^Tx,\qquad
K(\theta_0+x)\approx\tfrac12x^TFx.
$$

其中 $g=\nabla S(\theta_0)$。KL 在旧参数处取零且一阶导为零；在可交换求导、期望等常规正则条件下，其 Hessian 为 Fisher：

$$
F=\nabla^2K(\theta_0)
=\mathbb E_{d_{\pi_0},\pi_0}\left[\nabla\log\pi_0\nabla\log\pi_0^T\right].
$$

局部问题变成

$$
\max_x g^Tx\quad\text{s.t.}\quad\tfrac12x^TFx\le\delta.
$$

若 F 正定且 g 非零，拉格朗日驻点满足

$$
g-\lambda Fx=0\quad\Rightarrow\quad x=\lambda^{-1}F^{-1}g.
$$

令约束取等号，得到

$$
\boxed{x_* = \sqrt{\frac{2\delta}{g^TF^{-1}g}}F^{-1}g.}
$$

$F^{-1}g$ 是自然梯度方向；前面的因子把该方向缩放到局部 KL 预算边界。g 的整体正比例缩放在无阻尼理想解中会抵消，但一般的非线性优势变换不会保持方向。

### 3.1 共轭梯度与 Fisher 向量积

神经网络参数多，不能显式构造或求逆 F。先用共轭梯度（CG）近似解 $Fx=g$，只需能够计算 $Fv$：

$$
Fv=\left.\nabla_\theta\big(\nabla_\theta K(\theta)^T v\big)\right|_{\theta=\theta_0}.
$$

自动微分实现需要保留一阶导数的计算图，再做一次向量积求导；不是把 $v$ 或旧分布也当作待优化参数。

CG 从 $x=0,r=g,p=r$ 开始，重复 $\alpha=r^Tr/(p^TFp)$、$x\leftarrow x+\alpha p$、$r_{new}\leftarrow r-\alpha Fp$、$p\leftarrow r_{new}+\frac{r_{new}^Tr_{new}}{r^Tr}p$，直到残差足够小或达到迭代上限。

### 3.2 奇异、阻尼与数值保护

softmax 参数可能有平移冗余，采样 Fisher 也可能秩不足。通常用 $B=F+\eta I$、$\eta>0$，求解 $Bx=g$，并取

$$
x_{full}=\sqrt{\frac{2\delta}{x^TBx}}x.
$$

这是在阻尼二次型下保守缩放的一种实现约定；有些实现用未阻尼 F 做缩放，需在代码中说明。阻尼是数值稳定项，不是 KL 的定义。遇到零梯度、非正曲率、非有限数值或求解失败时，应放弃本次更新，保留旧参数。

## 第 4 讲：回溯线搜索与完整训练流程

### 4.1 为什么缩放后还要回溯？

KL 的二阶模型只在附近准确；即使 $\frac12x^TFx=\delta$，完整步的真实 K 也可能超过预算。

依次试 $\theta_j=\theta_0+\beta^j x_{full}$，$j=0,1,\ldots,m-1$，例如 $\beta=0.5$。在**同一冻结批次**检查：

1. 实际 KL 有限且 $K(\theta_j)\le\delta$；
2. 实际代理增益 $\Delta S=\hat S(\theta_j)-\hat S(\theta_0)>0$；
3. 增益达到预期的一定比例：$\Delta S\ge c\,\beta^j g^Tx_{full}$，例如 $c=0.1$。

接受首个满足条件的候选；全部失败则恢复 $\theta_0$。这里“实际”是相对于二阶模型而言，指完整非线性 KL 和代理目标，仍不等于环境真实回报。

### 4.2 与 Critic、GAE 配合

冻结旧策略收集轨迹，保存观测、动作、奖励、终止标志、旧 log probability 及计算 KL 所需的旧分布参数。用冻结价值预测构造

$$
\delta_t=R_{t+1}+\gamma(1-\mathrm{terminated}_t)V(S_{t+1})-V(S_t),
$$

$$
\hat A_t=\delta_t+\gamma\lambda\,\mathrm{continue}_t\hat A_{t+1}.
$$

真正终止时不自举。单纯时间截断需用 reset **之前**的最后观测自举；优势递推在采样片段边界停止，不能跨 reset 接到下一回合。$\lambda<1$ 的自举通常用偏差换方差。

```text
重复：
    固定 Actor 与 Critic 的旧快照，采集当前策略轨迹
    从旧价值计算 GAE 与价值目标，随后冻结这些数值
    定义新旧概率比代理 S 和旧到新平均 KL
    计算 g；用 Fisher 向量积与 CG 求解方向
    缩放至局部预算，回溯检查完整 KL 和代理增益
    接受候选，否则恢复旧 Actor
    用固定目标拟合 Critic
    丢弃本批，进入下一轮同策略采样
```

若 Actor 与 Critic 共享参数，单独拟合 Critic 也会改变策略，可能破坏刚满足的 KL；应采用独立参数，或显式控制共享部分的变化。停止优势梯度不等于解决共享参数更新的影响。

理论使用折扣状态权重；很多工程实现对有限采样步直接平均。二者应说明区别。若按完整回合实现起点折扣目标，应保留 $\gamma^t$ 并正确归一化；不能一边声称精确对应理论，一边默默改用别的状态权重。

## 第 5 讲：比较与实验

| 方法 | 策略更新约束/控制 | 计算特点 |
| --- | --- | --- |
| 普通 Actor-Critic | 学习率，可能附梯度裁剪 | 一阶更新；参数梯度小不等于 KL 小 |
| 自然策略梯度 | Fisher 度量方向 | 不自动包含 TRPO 的非线性 KL 检查 |
| TRPO | KL 二阶近似、CG、回溯 | 多次 Fisher 向量积与候选评估 |
| PPO-Clip | 裁剪概率比代理目标 | 常用一阶优化、多轮小批次；裁剪不构成严格 KL 上界 |

[实验](code/README.md) 使用一步 bandit 的精确期望，便于隔离和验证优化器本身。它不训练 Critic、不采样轨迹，也不等同于完整神经网络 TRPO。完成后再把精确期望替换为冻结轨迹与 GAE，才能构建采样版算法。

建议顺序：例题 1–3 → 默认实验 → 例题 4 与稀有动作实验 → 例题 5–6 → [练习](exercises.md)。

## 参考资料

- Schulman et al., 2015, [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)：性能下界、最大 KL 与实用平均 KL 近似。
- Schulman et al., 2016, [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)：GAE。
- [OpenAI Spinning Up：TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html)：目标、更新过程和实现思路。

[返回总目录](../README.md) · [课堂例题](examples.md) · [课后练习](exercises.md)
