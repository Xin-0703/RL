# Actor-Critic 课后练习

先独立作答，再核对 [参考答案](answers.md)。除特别说明外，采用表格 softmax、独立状态价值参数、普通梯度更新；$\delta=y-V$，Actor 上升、Critic 半梯度下降。

## 一、概念题

### 1. 两种角色

Actor、Critic 分别输出什么、学习什么？为什么 Critic 的输出不应当经过 softmax？Critic 是否就是环境奖励函数？

### 2. 与前面算法的联系

比较 DQN、REINFORCE、一步 Actor-Critic 的动作选择方式与更新权重。一步 AC 的 Critic 目标为什么没有 $\max$？

### 3. 判断并解释

1. 只要 Critic 不准确，MC 回报减去冻结状态基线的策略梯度就一定有偏。
2. 一步 TD 误差对任意 Critic 都是精确优势。
3. 基础同策略 AC 可以无修正地重复使用任意旧策略经验。
4. `detach` 优势与维护独立目标网络是同一件事。
5. Actor、Critic 两个方向都应从这次更新前的参数计算。
6. Actor-Critic 一定需要两个完全独立的神经网络。

## 二、计算题

### 4. 当前策略的 V、Q、A

使用宝箱环境：金币 +3 终止；山洞 −1 到 $S_1$；宝箱 +8 终止；陷阱 −4 终止。起点山洞概率为 0.6，山洞宝箱概率为 0.75，$\gamma=0.9$。

求两个 V、四个 Q、四个优势，并检查每状态优势的策略加权均值是否为零。

### 5. 正优势的一次更新

起点 $t=0$，logits `[0,0]`，旧值 $V(S_0)=1,V(S_1)=4$。抽到山洞，奖励 −1，未终止。$\gamma=0.9,\alpha_A=0.2,\alpha_C=0.1$。

求 y、δ、得分梯度、两个参数的新值和新的山洞概率。说明下一状态的 V 是否也被本次 Critic 更新改变。

### 6. 负优势与时刻折扣

在 $t=1$ 的山洞抽到陷阱，奖励 −4 并终止。山洞旧值为 2，Actor logits `[0,0]`，$\gamma=0.9,\alpha_A=0.1,\alpha_C=0.2$。

求目标、误差、Actor 方向、新 logits、宝箱概率与山洞新 V。为什么负奖励的动作仍可能出现正优势？

### 7. 一批两个回合

冻结两状态 logits 为零、V 为 `[2,5]`，$\gamma=0.9$。回合 A 拿金币结束，回合 B 进山洞后踩陷阱。求每条经验的 y、δ，批量 Actor 与 Critic 方向以及 $L_C$。分母是 2 还是 3？

### 8. 终止与外部截断

奖励 2，当前 V=3，最后观测 V=10，$\gamma=0.9$。分别计算真正终止和单纯外部截断时的 y、δ。若随后 reset 到初始状态 V=1，能否将 1 替代最后观测的 10？

### 9. n 步与 GAE

冻结价值 $V(S_0)=1,V(S_1)=3,V(S_2)=4$。奖励依次为 0、2、5，第三步真正终止，$\gamma=0.9$。

求三个 TD 误差；起点两步优势、三步优势；$\lambda=0.8$ 的起点 GAE。用 TD 误差求和核对三步结果。

## 三、推导题

### 10. TD 作为优势的条件

证明 Critic 精确为 $V^\pi$ 时 $\mathbb E[\delta\mid s,a]=A^\pi(s,a)$。若价值误差为 e，写出额外偏差项，并解释当前状态项与下一状态项对策略梯度的不同影响。

### 11. 停止梯度

考虑一条转移的 $\delta=r+\gamma V_\phi(s')-V_\phi(s)$ 和损失 $-\delta\log\pi_\theta(a\mid s)$。如果优势未停止梯度，这个损失对 Critic 参数多出了什么导数？它是否就是我们希望的 TD 价值拟合方向？

### 12. 精确期望与策略梯度

均匀策略宝箱环境，$\gamma=0.9$，真实 V 为 `[1.9,2]`。

1. 枚举三条完整路径及概率。
2. 用精确 Critic 算起点和山洞两个 logit 向量的 TD Actor 梯度期望。
3. 直接对 $J=3(1-p)+p[-1+0.9(12q-4)]$ 求导核对，其中 p 为山洞概率、q 为宝箱概率。

## 四、编程与实验题

### 13. 多种子训练

分别用种子 0、1、2 训练默认宝箱 TD AC。记录两个动作概率、Critic V、当前策略解析 V、解析 J、随机策略评估及贪心诊断。为何不直接把 Critic 和最优 `[6.2,8]` 比较？

### 14. MC 与 TD、学习率对照

相同预算和种子下比较 `--estimator td0` 与 `--estimator mc`。再将 Critic 学习率改为 0.01，观察学习过程。能否从一次运行断言某个方法方差更低或更稳定？需要怎样的固定策略实验才能直接研究梯度方差？

### 15. 找出实现问题

指出下面片段的问题并给出修正思路。假设数据来自当前策略，时间限制为外部截断，Actor 与 Critic 参数独立，t 不一定为 0。

```python
done = terminated or truncated
target = reward + gamma * critic(next_state)
advantage = target - critic(state)
actor_loss = advantage * actor(state).softmax(-1)[action]
critic_loss = (target - critic(state)).square()
(actor_loss + critic_loss).backward()
actor_optimizer.step()
critic_optimizer.step()
```

### 16. 选做：n 步优势

在独立分支或副本增加 n 步优势估计，先实现两步情形。要求：真正终止尾部为零；外部片段边界保留 V 自举；不得跨 reset 拼接奖励；Actor 优势和 Critic 目标均使用本批冻结旧值。用确定性轨迹测试 n=1 与现有 TD 一致、完整终止轨迹 n 足够大时与 MC 一致。

[参考答案](answers.md) · [返回讲义](README.md)
