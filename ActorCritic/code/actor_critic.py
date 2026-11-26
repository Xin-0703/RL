"""表格 softmax Actor-Critic：Python 3.10+ 标准库，冻结批次一步 TD / MC 对照。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    state: int
    action: int
    reward: float
    next_state: int
    terminated: bool
    truncated: bool
    probabilities: tuple[float, ...]


def softmax(logits):
    if not logits or not all(math.isfinite(z) for z in logits):
        raise ValueError("logits must be nonempty and finite")
    weights = [math.exp(z - max(logits)) for z in logits]
    return [w / sum(weights) for w in weights]


def sample_action(probabilities, rng):
    draw, total = rng.random(), 0.0
    for a, p in enumerate(probabilities):
        total += p
        if draw < total:
            return a
    return len(probabilities) - 1


def initial_parameters(env):
    if env not in ("treasure", "bandit"):
        raise ValueError("unknown environment")
    count = 2 if env == "treasure" else 1
    return [[0.0, 0.0] for _ in range(count)], [0.0] * count


def sample_episode(env, logits, rng, greedy=False):
    state, episode = 0, []
    for _ in range(2):
        probabilities = softmax(logits[state])
        if greedy:
            tied = [a for a, p in enumerate(probabilities) if p == max(probabilities)]
            probabilities = [float(a in tied) / len(tied) for a in range(2)]
        action = sample_action(probabilities, rng)
        if env == "bandit":
            reward = float(rng.random() < (0.2, 0.8)[action])
            terminated = True
        elif env == "treasure":
            if state == 0:
                reward, terminated = (3.0, True) if action == 0 else (-1.0, False)
            else:
                reward, terminated = (8.0 if action == 0 else -4.0), True
        else:
            raise ValueError("unknown environment")
        next_state = 0 if terminated else 1
        episode.append(Step(state, action, reward, next_state, terminated,
                            False, tuple(probabilities)))
        if terminated:
            return episode
        state = next_state
    raise RuntimeError("toy environment failed to terminate")


def td_target(step, values, gamma):
    # 不读取终点值；truncated 不屏蔽后续价值。
    return step.reward if step.terminated else step.reward + gamma * values[step.next_state]


def mc_returns(episode, gamma):
    if not episode or not episode[-1].terminated:
        raise ValueError("MC requires a complete naturally terminated episode")
    result, total = [0.0] * len(episode), 0.0
    for t in reversed(range(len(episode))):
        total = episode[t].reward + gamma * total
        result[t] = total
    return result


def batch_directions(batch, values, gamma, estimator="td0"):
    """旧策略、旧 critic 固定；返回梯度上升方向，而非更新后的参数。

    actor: Σ γ^t δ (one_hot - π) / B
    critic: Σ δ one_hot(state) / B，为半梯度负损失方向。
    """
    if not batch or estimator not in ("td0", "mc"):
        raise ValueError("nonempty batch and valid estimator required")
    actor = [[0.0, 0.0] for _ in values]
    critic = [0.0] * len(values)
    actor_loss, critic_loss = 0.0, 0.0
    for episode in batch:
        if not episode:
            raise ValueError("empty trajectory")
        targets = (mc_returns(episode, gamma) if estimator == "mc" else
                   [td_target(step, values, gamma) for step in episode])
        for t, (step, target) in enumerate(zip(episode, targets)):
            delta = target - values[step.state]
            weight = gamma ** t * delta
            for action in range(2):
                actor[step.state][action] += weight * (
                    float(action == step.action) - step.probabilities[action]
                ) / len(batch)
            critic[step.state] += delta / len(batch)
            actor_loss -= weight * math.log(step.probabilities[step.action]) / len(batch)
            critic_loss += 0.5 * delta ** 2 / len(batch)
    return actor, critic, actor_loss, critic_loss


def apply_updates(logits, values, actor_direction, critic_direction, actor_lr, critic_lr):
    # 两个方向已从更新前的参数算好，避免先改 V 再计算 actor 优势。
    new_logits = [[z + actor_lr * g for z, g in zip(row, grad)]
                  for row, grad in zip(logits, actor_direction)]
    new_values = [v + critic_lr * g for v, g in zip(values, critic_direction)]
    if not all(math.isfinite(x) for row in new_logits for x in row) or not all(
            math.isfinite(x) for x in new_values):
        raise FloatingPointError("nonfinite update; reduce learning rates")
    return new_logits, new_values


def analytic_values(env, logits, gamma):
    """仅训练后/日志诊断使用，不供 TD 训练读取。"""
    p = [softmax(row) for row in logits]
    if env == "bandit":
        return [0.2 * p[0][0] + 0.8 * p[0][1]]
    v1 = 8 * p[1][0] - 4 * p[1][1]
    return [3 * p[0][0] + p[0][1] * (-1 + gamma * v1), v1]


def evaluate(env, logits, gamma, episodes, seed, greedy=False):
    rng, raw, discounted = random.Random(seed), [], []
    counts = [[0, 0] for _ in logits]
    for _ in range(episodes):
        episode = sample_episode(env, logits, rng, greedy)
        raw.append(sum(step.reward for step in episode))
        discounted.append(sum(gamma ** t * s.reward for t, s in enumerate(episode)))
        for step in episode:
            counts[step.state][step.action] += 1
    return {"episodes": episodes, "seed": seed, "greedy": greedy,
            "return_mean": statistics.mean(raw), "return_std": statistics.pstdev(raw),
            "discounted_return_mean": statistics.mean(discounted),
            "discounted_return_std": statistics.pstdev(discounted), "action_counts": counts}


def train(args):
    logits, values = initial_parameters(args.env)
    rng = random.Random(args.seed)
    completed, updates, curve = 0, 0, []
    while completed < args.episodes:
        size = min(args.batch_size, args.episodes - completed)
        # 整批采样与方向计算均不改变 actor/critic；数据只使用一次。
        batch = [sample_episode(args.env, logits, rng) for _ in range(size)]
        ag, cg, aloss, closs = batch_directions(batch, values, args.gamma, args.estimator)
        logits, values = apply_updates(logits, values, ag, cg, args.actor_lr, args.critic_lr)
        completed += size
        updates += 1
        if updates == 1 or updates % args.log_every == 0 or completed == args.episodes:
            exact = analytic_values(args.env, logits, args.gamma)
            curve.append({"episodes": completed, "updates": updates,
                          "analytic_objective": exact[0], "actor_surrogate_loss": aloss,
                          "critic_loss": closs, "critic_max_error": max(
                              abs(v - x) for v, x in zip(values, exact)),
                          "p_start_action1": softmax(logits[0])[1]})
    exact = analytic_values(args.env, logits, args.gamma)
    result = {"episodes": completed, "updates": updates, "logits": logits,
              "policy_probabilities": [softmax(row) for row in logits],
              "critic_values": values, "analytic_policy_values": exact,
              "analytic_objective": exact[0],
              "analytic_optimal_value": 0.8 if args.env == "bandit" else max(3.0, -1 + args.gamma * 8),
              "stochastic_evaluation": evaluate(args.env, logits, args.gamma,
                  args.eval_episodes, args.seed + 10000),
              "greedy_diagnostic": evaluate(args.env, logits, args.gamma,
                  args.eval_episodes, args.seed + 20000, True)}
    return result, curve


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("treasure", "bandit"), default="treasure")
    parser.add_argument("--estimator", choices=("td0", "mc"), default="td0")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--actor-lr", type=float, default=0.1)
    parser.add_argument("--critic-lr", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(args.episodes, args.batch_size, args.eval_episodes, args.log_every) <= 0:
        parser.error("counts must be positive")
    if not math.isfinite(args.gamma) or not 0 <= args.gamma <= 1:
        parser.error("gamma must be finite and in [0,1]")
    if any(not math.isfinite(x) or x <= 0 for x in (args.actor_lr, args.critic_lr)):
        parser.error("learning rates must be positive and finite")
    if args.output is None:
        args.output = Path("runs") / "ac" / f"{args.env}_{args.estimator}_seed{args.seed}_gamma{args.gamma:g}"
    return args


def main(argv=None):
    args = parse_args(argv)
    result, curve = train(args)
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), output=str(args.output))
    for name, data in (("config.json", config), ("result.json", result)):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False,
            indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "learning.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
