"""固定均匀策略下比较梯度估计噪声；解析 V 只作诊断参照。"""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from reinforce import analytic_gradient, batch_gradient, initial_logits, sample_episode, validate_gamma


def compare(env: str, gamma: float, batches: int, batch_size: int, seed: int) -> dict:
    validate_gamma(gamma)
    if batches < 2 or batch_size <= 0:
        raise ValueError("At least two batches and a positive batch size are required")
    logits, rng = initial_logits(env), random.Random(seed)
    # 固定均匀策略的解析 V，独立于本次所有动作与回报。不是训练基线。
    baseline = [0.5] if env == "bandit" else [1 + gamma, 2.0]
    exact = [g for row in analytic_gradient(env, logits, gamma) for g in row]
    estimates = {"none": [], "oracle_state_value": []}
    for _ in range(batches):
        episodes = [sample_episode(env, logits, rng) for _ in range(batch_size)]
        for name, values in (("none", [0.0] * len(logits)), ("oracle_state_value", baseline)):
            gradient = batch_gradient(episodes, gamma, values)
            estimates[name].append([g for row in gradient for g in row])
    result = {}
    for name, samples in estimates.items():
        coordinates = list(zip(*samples))
        mean = [statistics.mean(column) for column in coordinates]
        result[name] = {
            "mean_gradient_flat": mean,
            "mean_gradient_error_norm": math.sqrt(sum((a - b) ** 2 for a, b in zip(mean, exact))),
            "sum_coordinate_sample_variances": sum(statistics.variance(column) for column in coordinates),
            "mean_squared_error_to_exact": statistics.mean(
                sum((a - b) ** 2 for a, b in zip(sample, exact)) for sample in samples),
        }
    return {"env": env, "gamma": gamma, "seed": seed, "batches": batches,
            "batch_size": batch_size, "policy_probabilities": [[0.5, 0.5] for _ in logits],
            "frozen_oracle_baseline": baseline, "analytic_gradient_flat": exact,
            "comparison": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("bandit", "treasure"), default="treasure")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.env, args.gamma, args.batches, args.batch_size, args.seed)
    except ValueError as error:
        parser.error(str(error))
    path = args.output or Path("runs") / "pg" / f"variance_{args.env}_seed{args.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"Saved to: {path.resolve()}")


if __name__ == "__main__":
    main()
