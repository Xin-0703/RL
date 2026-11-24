"""回合式蒙特卡洛教学实验，仅依赖 Python 3.10+ 标准库。

包含首次/每次访问预测、首次访问同策略控制、固定起点异策略评估。
环境都自然终止；保护性步数上限触发时抛错，不返回不完整回合。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Hashable, Sequence


Policy = Callable[[int], Sequence[float]]


@dataclass(frozen=True)
class Step:
    state: int
    action: int
    reward: float  # R_{t+1}: 执行当前动作后收到的奖励
    behavior_probability: float


@dataclass(frozen=True)
class Episode:
    steps: tuple[Step, ...]
    terminated: bool


class IncompleteEpisodeError(RuntimeError):
    """未拿到完整回合，不能把缺失的尾部回报当作 0。"""


class RandomWalk:
    """0 和 6 为终点，1..5 为决策状态，从 3 出发。

    动作 0 向左，动作 1 向右；仅进入右终点时奖励为 1。
    """

    def reset(self) -> int:
        self.state = 3
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        if action not in (0, 1):
            raise ValueError("Action must be 0 or 1")
        if self.state in (0, 6):
            raise RuntimeError("Reset the environment after termination")
        self.state += -1 if action == 0 else 1
        return self.state, float(self.state == 6), self.state in (0, 6)


class Treasure:
    """与 DQN 章节相同的宝箱环境；2 是终止状态。"""

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        if action not in (0, 1):
            raise ValueError("Action must be 0 or 1")
        if self.state == 2:
            raise RuntimeError("Reset the environment after termination")
        if self.state == 0 and action == 1:
            self.state = 1
            return 1, -1.0, False
        reward = 3.0 if self.state == 0 else (8.0 if action == 0 else -4.0)
        self.state = 2
        return 2, reward, True


def check_gamma(gamma: float) -> None:
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")


def checked_probabilities(probabilities: Sequence[float]) -> list[float]:
    probabilities = list(probabilities)
    if not probabilities or any(
        not math.isfinite(p) or p < 0.0 or p > 1.0 for p in probabilities
    ):
        raise ValueError("Policy probabilities must be finite and in [0, 1]")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Policy probabilities must sum to 1")
    return probabilities


def sample_action(probabilities: Sequence[float], rng: random.Random) -> int:
    probabilities = checked_probabilities(probabilities)
    draw, cumulative = rng.random(), 0.0
    for action, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return action
    # 仅用于浮点累加导致的末端舍入；不选概率为 0 的尾部动作。
    return max(a for a, p in enumerate(probabilities) if p > 0.0)


def sample_episode(
    env: RandomWalk | Treasure,
    policy: Policy,
    rng: random.Random,
    max_steps: int = 10_000,
) -> Episode:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    state = env.reset()
    steps = []
    for _ in range(max_steps):
        probabilities = checked_probabilities(policy(state))
        action = sample_action(probabilities, rng)
        next_state, reward, terminated = env.step(action)
        steps.append(Step(state, action, reward, probabilities[action]))
        if terminated:
            return Episode(tuple(steps), terminated=True)
        state = next_state
    raise IncompleteEpisodeError(
        f"No natural termination within {max_steps} steps; "
        "the incomplete episode was not used for learning."
    )


def discounted_returns(rewards: Sequence[float], gamma: float) -> list[float]:
    """G_T=0，从后往前计算 G_t=R_{t+1}+gamma*G_{t+1}。"""
    check_gamma(gamma)
    result = [0.0] * len(rewards)
    total = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        total = rewards[t] + gamma * total
        result[t] = total
    return result


def episode_returns(episode: Episode, gamma: float) -> list[float]:
    if not episode.terminated:
        raise IncompleteEpisodeError("A complete, naturally terminated episode is required")
    if not episode.steps:
        raise ValueError("Teaching environments require at least one step")
    return discounted_returns([step.reward for step in episode.steps], gamma)


@dataclass
class MeanTable:
    values: dict[Hashable, float] = field(default_factory=dict)
    counts: dict[Hashable, int] = field(default_factory=dict)

    def update(self, key: Hashable, value: float) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1
        old = self.values.get(key, 0.0)
        self.values[key] = old + (value - old) / self.counts[key]

    def value(self, key: Hashable) -> float:
        return self.values.get(key, 0.0)


def update_episode(
    table: MeanTable,
    episode: Episode,
    gamma: float,
    visit: str = "first",
    action_value: bool = False,
) -> None:
    if visit not in ("first", "every"):
        raise ValueError("visit must be 'first' or 'every'")
    returns = episode_returns(episode, gamma)
    seen = set()  # 每个回合重新清空。
    for step, total in zip(episode.steps, returns):  # 正向认定首次访问。
        key = (step.state, step.action) if action_value else step.state
        if visit == "first" and key in seen:
            continue
        seen.add(key)
        table.update(key, total)


def epsilon_greedy_probabilities(q_values: Sequence[float], epsilon: float) -> list[float]:
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if not q_values or not all(math.isfinite(q) for q in q_values):
        raise ValueError("q_values must be nonempty and finite")
    best = max(q_values)
    greedy = [a for a, q in enumerate(q_values) if q == best]
    probabilities = [epsilon / len(q_values)] * len(q_values)
    for action in greedy:
        probabilities[action] += (1.0 - epsilon) / len(greedy)
    return probabilities


def treasure_policy(table: MeanTable, epsilon: float) -> Policy:
    return lambda state: epsilon_greedy_probabilities(
        [table.value((state, action)) for action in (0, 1)], epsilon
    )


def target_treasure_policy(state: int) -> Sequence[float]:
    """固定目标策略：起点选山洞，山洞选宝箱。"""
    return (0.0, 1.0) if state == 0 else (1.0, 0.0)


def trajectory_weight(
    episode: Episode, target_policy: Policy, start: int = 0
) -> float:
    """给定条件状态后修正动作分布；V 从 t 开始，Q 从 t+1 开始。

    调用者必须确保完整目标策略被行为策略覆盖；仅从已观测动作无法
    检查所有未观测动作的覆盖。长轨迹更适合专门的稳定权重算法。
    """
    if not episode.terminated:
        raise IncompleteEpisodeError("Importance sampling requires a complete episode")
    if not 0 <= start <= len(episode.steps):
        raise ValueError("start must be inside the trajectory or at its end")
    weight = 1.0
    for step in episode.steps[start:]:
        behavior = step.behavior_probability
        if not math.isfinite(behavior) or not 0.0 < behavior <= 1.0:
            raise ValueError("An observed action must have positive behavior probability")
        probabilities = checked_probabilities(target_policy(step.state))
        if not 0 <= step.action < len(probabilities):
            raise ValueError("Observed action is not in the target policy action space")
        weight *= probabilities[step.action] / behavior
        if not math.isfinite(weight):
            raise OverflowError("Importance weight overflow; use a stable long-horizon method")
    return weight


@dataclass
class ImportanceMean:
    count: int = 0
    weight_sum: float = 0.0
    ordinary: float = 0.0
    weighted: float | None = None

    def update(self, total: float, weight: float) -> None:
        if not math.isfinite(total) or not math.isfinite(weight) or weight < 0.0:
            raise ValueError("Return and nonnegative weight must be finite")
        product = weight * total
        new_weight_sum = self.weight_sum + weight
        if not math.isfinite(product) or not math.isfinite(new_weight_sum):
            raise OverflowError("Importance-sampling accumulator overflow")
        self.count += 1  # 零权重回合也进入普通估计的分母。
        self.ordinary += (product - self.ordinary) / self.count
        self.weight_sum = new_weight_sum
        if weight > 0.0:
            if self.weighted is None:
                self.weighted = total
            else:
                self.weighted += (weight / self.weight_sum) * (total - self.weighted)


def evaluate_treasure(
    table: MeanTable, gamma: float, episodes: int, seed: int
) -> dict:
    if episodes <= 0:
        raise ValueError("Evaluation episode count must be positive")
    env, rng = Treasure(), random.Random(seed)
    policy = treasure_policy(table, epsilon=0.0)
    totals, discounted, first_trace = [], [], []
    for index in range(episodes):
        episode = sample_episode(env, policy, rng)
        totals.append(sum(step.reward for step in episode.steps))
        discounted.append(episode_returns(episode, gamma)[0])
        if index == 0:
            first_trace = [
                {"state": step.state, "action": step.action, "reward": step.reward}
                for step in episode.steps
            ]
    return {
        "episodes": episodes,
        "seed": seed,
        "epsilon": 0.0,
        "return_mean": statistics.mean(totals),
        "return_std": statistics.pstdev(totals),
        "discounted_return_mean": statistics.mean(discounted),
        "discounted_return_std": statistics.pstdev(discounted),
        "first_episode_trace": first_trace,
    }


def prediction_metrics(table: MeanTable) -> dict:
    states = list(range(1, 6))
    values = [table.values.get(state) for state in states]
    truth = [state / 6 for state in states]
    # 未覆盖时不把初始化值当成已有估计。
    rmse = None if None in values else math.sqrt(
        sum((value - exact) ** 2 for value, exact in zip(values, truth)) / 5
    )
    return {
        "states": states,
        "values": values,
        "counts": [table.counts.get(state, 0) for state in states],
        "analytic_values": truth,
        "rmse": rmse,
    }


def predict(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    env, rng, table = RandomWalk(), random.Random(args.seed), MeanTable()
    curve = []
    for n in range(1, args.episodes + 1):
        episode = sample_episode(env, lambda _: (0.5, 0.5), rng, args.max_steps)
        update_episode(table, episode, args.gamma, args.visit)
        if n == 1 or n % args.log_every == 0 or n == args.episodes:
            metrics = prediction_metrics(table)
            curve.append({
                "episode": n, "rmse": metrics["rmse"],
                **{f"v_{s}": table.values.get(s) for s in range(1, 6)},
            })
    return prediction_metrics(table), curve


def control(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    env, rng, table = Treasure(), random.Random(args.seed), MeanTable()
    policy = treasure_policy(table, args.epsilon)
    totals, discounted, curve = 0.0, 0.0, []
    for n in range(1, args.episodes + 1):
        # Q 在 sample_episode 内不变，因此一回合内的策略固定。
        episode = sample_episode(env, policy, rng, args.max_steps)
        update_episode(table, episode, args.gamma, visit="first", action_value=True)
        totals += sum(step.reward for step in episode.steps)
        discounted += episode_returns(episode, args.gamma)[0]
        if n == 1 or n % args.log_every == 0 or n == args.episodes:
            curve.append({
                "episode": n, "epsilon": args.epsilon,
                "training_return_mean": totals / n,
                "training_discounted_return_mean": discounted / n,
                "q_start_cave": table.values.get((0, 1)),
            })
    result = {
        "states": [0, 1], "actions": [0, 1],
        "q_values": [[table.values.get((s, a)) for a in (0, 1)] for s in (0, 1)],
        "counts": [[table.counts.get((s, a), 0) for a in (0, 1)] for s in (0, 1)],
        "greedy_action_sets": [
            [a for a, p in enumerate(treasure_policy(table, 0.0)(s)) if p > 0]
            for s in (0, 1)
        ],
        "analytic_optimal_q": [[3.0, -1.0 + args.gamma * 8.0], [8.0, -4.0]],
        "training_return_mean": totals / args.episodes,
        "training_discounted_return_mean": discounted / args.episodes,
        "evaluation": evaluate_treasure(table, args.gamma, args.eval_episodes, args.seed + 10_000),
    }
    return result, curve


def offpolicy(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    env, rng = Treasure(), random.Random(args.seed)
    estimator, positive, curve = ImportanceMean(), 0, []
    for n in range(1, args.episodes + 1):
        episode = sample_episode(env, lambda _: (0.5, 0.5), rng, args.max_steps)
        weight = trajectory_weight(episode, target_treasure_policy)
        estimator.update(episode_returns(episode, args.gamma)[0], weight)
        positive += int(weight > 0)
        if n == 1 or n % args.log_every == 0 or n == args.episodes:
            curve.append({
                "episode": n, "ordinary": estimator.ordinary,
                "weighted": estimator.weighted, "weight_sum": estimator.weight_sum,
            })
    return {
        "state": 0, "episodes": estimator.count,
        "positive_weight_episodes": positive, "weight_sum": estimator.weight_sum,
        "ordinary": estimator.ordinary, "weighted": estimator.weighted,
        "analytic_target_value": -1.0 + args.gamma * 8.0,
    }, curve


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("predict", "control", "offpolicy"), default="predict")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None, help="control only; default 0.1")
    parser.add_argument("--visit", choices=("first", "every"), default=None, help="predict only")
    parser.add_argument("--eval-episodes", type=int, default=None, help="control only; default 100")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.task != "control" and (args.epsilon is not None or args.eval_episodes is not None):
        parser.error("--epsilon and --eval-episodes apply only to control")
    if args.task != "predict" and args.visit is not None:
        parser.error("--visit applies only to predict; control uses first visits")
    args.episodes = args.episodes if args.episodes is not None else (50_000 if args.task == "control" else 10_000)
    args.gamma = args.gamma if args.gamma is not None else (1.0 if args.task == "predict" else 0.9)
    args.epsilon = args.epsilon if args.epsilon is not None else 0.1
    args.visit = args.visit or "first"
    args.eval_episodes = args.eval_episodes if args.eval_episodes is not None else 100
    args.log_every = args.log_every if args.log_every is not None else max(1, args.episodes // 100)
    if min(args.episodes, args.eval_episodes, args.max_steps, args.log_every) <= 0:
        parser.error("episode counts, max-steps and log-every must be positive")
    try:
        check_gamma(args.gamma)
        epsilon_greedy_probabilities([0.0, 0.0], args.epsilon)
    except ValueError as error:
        parser.error(str(error))
    if args.task == "predict" and args.gamma != 1.0:
        parser.error("predict uses gamma=1 so that analytic values i/6 remain valid")
    if args.output is None:
        suffix = f"{args.task}_seed{args.seed}_gamma{args.gamma:g}"
        if args.task == "predict":
            suffix += f"_{args.visit}"
        elif args.task == "control":
            suffix += f"_eps{args.epsilon:g}"
        args.output = Path("runs") / "mc" / suffix
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    runners = {"predict": predict, "control": control, "offpolicy": offpolicy}
    result, curve = runners[args.task](args)
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output"] = str(args.output)
    for filename, data in (("config.json", config), ("result.json", result)):
        (args.output / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    with (args.output / "learning.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
