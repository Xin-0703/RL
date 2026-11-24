"""表格 softmax REINFORCE：仅用 Python 3.10+ 标准库。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class IncompleteEpisodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Step:
    state: int
    action: int
    reward: float
    probabilities: tuple[float, ...]  # 采样时冻结策略的概率


@dataclass(frozen=True)
class Episode:
    steps: tuple[Step, ...]
    terminated: bool = True


def softmax(logits: Sequence[float]) -> list[float]:
    if not logits or not all(math.isfinite(z) for z in logits):
        raise ValueError("Logits must be nonempty and finite")
    maximum = max(logits)
    weights = [math.exp(z - maximum) for z in logits]
    return [weight / sum(weights) for weight in weights]


def sample_action(probabilities: Sequence[float], rng: random.Random) -> int:
    if (not probabilities or any(not math.isfinite(p) or p < 0 for p in probabilities)
            or not math.isclose(sum(probabilities), 1, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Probabilities must be finite, nonnegative and sum to 1")
    draw, cumulative = rng.random(), 0.0
    for action, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return action
    return max(a for a, p in enumerate(probabilities) if p > 0)


def validate_gamma(gamma: float) -> None:
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and in [0, 1]")


def returns(episode: Episode, gamma: float) -> list[float]:
    validate_gamma(gamma)
    if not episode.terminated or not episode.steps:
        raise IncompleteEpisodeError("A complete nonempty episode is required")
    result, total = [0.0] * len(episode.steps), 0.0
    for t in range(len(episode.steps) - 1, -1, -1):
        total = episode.steps[t].reward + gamma * total
        result[t] = total
    return result


def initial_logits(env: str) -> list[list[float]]:
    if env not in ("bandit", "treasure"):
        raise ValueError("Unknown environment")
    return [[0.0, 0.0] for _ in range(1 if env == "bandit" else 2)]


def sample_episode(env: str, logits: list[list[float]], rng: random.Random,
                   max_steps: int = 10, greedy: bool = False) -> Episode:
    if env not in ("bandit", "treasure") or max_steps <= 0:
        raise ValueError("Unknown environment or invalid max_steps")
    state, steps = 0, []
    for _ in range(max_steps):
        probabilities = softmax(logits[state])
        if greedy:
            maximum = max(probabilities)
            tied = [a for a, p in enumerate(probabilities) if p == maximum]
            probabilities = [float(a in tied) / len(tied) for a in range(2)]
        action = sample_action(probabilities, rng)
        if env == "bandit":
            reward = float(rng.random() < (0.2, 0.8)[action])
            terminated = True
        elif state == 0:
            reward, terminated = (3.0, True) if action == 0 else (-1.0, False)
        else:
            reward, terminated = (8.0 if action == 0 else -4.0), True
        steps.append(Step(state, action, reward, tuple(probabilities)))
        if terminated:
            return Episode(tuple(steps))
        state = 1
    raise IncompleteEpisodeError("Protection limit reached: no update uses incomplete episodes")


def batch_gradient(episodes: Sequence[Episode], gamma: float,
                   baselines: Sequence[float]) -> list[list[float]]:
    """按回合平均；基线必须来自采样本批之前的数据，当前动作不能影响它。"""
    if not episodes or not baselines or not all(math.isfinite(b) for b in baselines):
        raise ValueError("Nonempty episodes and finite baselines are required")
    gradient = [[0.0, 0.0] for _ in baselines]
    for episode in episodes:
        totals = returns(episode, gamma)  # 不完整时在更新 actor 前报错
        for t, (step, total) in enumerate(zip(episode.steps, totals)):
            if not 0 <= step.state < len(baselines) or len(step.probabilities) != 2:
                raise ValueError("Invalid state or action space")
            probabilities = step.probabilities
            if (step.action not in (0, 1) or any(not math.isfinite(p) or p < 0 for p in probabilities)
                    or not math.isclose(sum(probabilities), 1, rel_tol=0, abs_tol=1e-12)
                    or probabilities[step.action] <= 0):
                raise ValueError("Invalid sampled policy probabilities")
            weight = gamma ** t * (total - baselines[step.state])
            for action in range(2):
                gradient[step.state][action] += weight * (
                    float(action == step.action) - probabilities[action]
                ) / len(episodes)
    return gradient


def update_logits(logits: list[list[float]], gradient: list[list[float]], lr: float) -> list[list[float]]:
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be positive and finite")
    updated = [[z + lr * g for z, g in zip(row, grad)] for row, grad in zip(logits, gradient)]
    if not all(math.isfinite(z) for row in updated for z in row):
        raise FloatingPointError("Nonfinite update; reduce the learning rate")
    return updated


class HistoricalBaseline:
    def __init__(self, states: int):
        self.values = [0.0] * states
        self.counts = [0] * states

    def fit_after_batch(self, episodes: Sequence[Episode], gamma: float) -> None:
        samples = [(step.state, total) for episode in episodes
                   for step, total in zip(episode.steps, returns(episode, gamma))]
        for state, total in samples:
            self.counts[state] += 1
            self.values[state] += (total - self.values[state]) / self.counts[state]


def analytic_objective(env: str, logits: list[list[float]], gamma: float) -> float:
    """仅用于解析诊断，不参与采样训练的梯度。"""
    probabilities = [softmax(row) for row in logits]
    if env == "bandit":
        return 0.2 * probabilities[0][0] + 0.8 * probabilities[0][1]
    if env == "treasure":
        cave = -1 + gamma * (8 * probabilities[1][0] - 4 * probabilities[1][1])
        return 3 * probabilities[0][0] + cave * probabilities[0][1]
    raise ValueError("Unknown environment")


def analytic_gradient(env: str, logits: list[list[float]], gamma: float) -> list[list[float]]:
    probabilities = [softmax(row) for row in logits]
    if env == "bandit":
        gap = 0.6 * probabilities[0][0] * probabilities[0][1]
        return [[-gap, gap]]
    p, q = probabilities
    cave = -1 + gamma * (8 * q[0] - 4 * q[1])
    gap0 = (cave - 3) * p[0] * p[1]
    gap1 = p[1] * gamma * 12 * q[0] * q[1]
    return [[-gap0, gap0], [gap1, -gap1]]


def gradient_check(env: str, gamma: float, step_size: float = 1e-6) -> dict:
    logits = [[0.2, -0.3] for _ in initial_logits(env)]
    exact, numeric = analytic_gradient(env, logits, gamma), []
    for state, row in enumerate(logits):
        estimated = []
        for action in range(len(row)):
            plus, minus = [r[:] for r in logits], [r[:] for r in logits]
            plus[state][action] += step_size
            minus[state][action] -= step_size
            estimated.append((analytic_objective(env, plus, gamma) -
                              analytic_objective(env, minus, gamma)) / (2 * step_size))
        numeric.append(estimated)
    return {"logits": logits, "analytic": exact, "finite_difference": numeric,
            "max_abs_error": max(abs(a - b) for erow, nrow in zip(exact, numeric)
                                 for a, b in zip(erow, nrow))}


def evaluate(env: str, logits: list[list[float]], gamma: float, episodes: int,
             seed: int, greedy: bool = False) -> dict:
    rng, raw, discounted = random.Random(seed), [], []
    action_counts = [[0, 0] for _ in logits]
    for _ in range(episodes):
        episode = sample_episode(env, logits, rng, greedy=greedy)
        raw.append(sum(step.reward for step in episode.steps))
        discounted.append(returns(episode, gamma)[0])
        for step in episode.steps:
            action_counts[step.state][step.action] += 1
    return {"episodes": episodes, "seed": seed, "greedy": greedy,
            "return_mean": statistics.mean(raw), "return_std": statistics.pstdev(raw),
            "discounted_return_mean": statistics.mean(discounted),
            "discounted_return_std": statistics.pstdev(discounted), "action_counts": action_counts}


def train(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    logits, rng = initial_logits(args.env), random.Random(args.seed)
    baseline = HistoricalBaseline(len(logits))
    completed, updates, curve = 0, 0, []
    while completed < args.episodes:
        batch_size = min(args.batch_size, args.episodes - completed)
        # actor 与基线在整批数据采样和梯度构造期间均固定。
        frozen_baseline = baseline.values[:] if args.baseline == "history" else [0.0] * len(logits)
        batch = [sample_episode(args.env, logits, rng, args.max_steps) for _ in range(batch_size)]
        gradient = batch_gradient(batch, args.gamma, frozen_baseline)
        logits = update_logits(logits, gradient, args.lr)
        if args.baseline == "history":
            baseline.fit_after_batch(batch, args.gamma)
        completed += batch_size
        updates += 1
        if updates == 1 or updates % args.log_every == 0 or completed == args.episodes:
            curve.append({"episodes": completed, "updates": updates,
                          "analytic_objective": analytic_objective(args.env, logits, args.gamma),
                          "batch_discounted_return_mean": statistics.mean(returns(e, args.gamma)[0] for e in batch),
                          "gradient_norm": math.sqrt(sum(g * g for row in gradient for g in row)),
                          "p_start_action1": softmax(logits[0])[1]})
    result = {
        "episodes": completed, "updates": updates, "logits": logits,
        "policy_probabilities": [softmax(row) for row in logits],
        "baseline_values": baseline.values if args.baseline == "history" else None,
        "baseline_counts": baseline.counts if args.baseline == "history" else None,
        "analytic_objective": analytic_objective(args.env, logits, args.gamma),
        "analytic_optimal_value": 0.8 if args.env == "bandit" else max(3.0, -1 + args.gamma * 8),
        "stochastic_evaluation": evaluate(args.env, logits, args.gamma, args.eval_episodes, args.seed + 10000),
        "greedy_diagnostic": evaluate(args.env, logits, args.gamma, args.eval_episodes, args.seed + 20000, True),
        "gradient_check": gradient_check(args.env, args.gamma),
    }
    return result, curve


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("bandit", "treasure"), default="treasure")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--baseline", choices=("none", "history"), default="history")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10, help="Update interval for logging")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_gamma(args.gamma)
        if min(args.episodes, args.batch_size, args.eval_episodes, args.max_steps, args.log_every) <= 0:
            raise ValueError("Counts and limits must be positive")
        if not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError("lr must be positive and finite")
    except ValueError as error:
        parser.error(str(error))
    if args.output is None:
        args.output = Path("runs") / "pg" / f"{args.env}_{args.baseline}_seed{args.seed}_gamma{args.gamma:g}"
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result, curve = train(args)
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output"] = str(args.output)
    for filename, data in (("config.json", config), ("result.json", result)):
        (args.output / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "learning.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
