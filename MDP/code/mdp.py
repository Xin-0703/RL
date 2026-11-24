"""有限 MDP 与动态规划教学实验，仅依赖 Python 3.10+ 标准库。

model[state][action] 是 Outcome(概率, 下一状态, 奖励) 列表。
没有合法动作的状态是终止状态，其后续价值固定为 0。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Outcome:
    probability: float
    next_state: str
    reward: float


Model = dict[str, dict[str, list[Outcome]]]
Policy = dict[str, dict[str, float]]
Values = dict[str, float]


def validate_model(model: Model) -> None:
    if not model:
        raise ValueError("Model must contain at least one state")
    for state, actions in model.items():
        if not isinstance(state, str) or not state:
            raise ValueError("State names must be nonempty strings")
        for action, outcomes in actions.items():
            if not isinstance(action, str) or not action or not outcomes:
                raise ValueError("Each action must be named and have outcomes")
            for item in outcomes:
                if not math.isfinite(item.probability) or not 0 <= item.probability <= 1:
                    raise ValueError("Outcome probabilities must be finite and in [0, 1]")
                if item.next_state not in model or not math.isfinite(item.reward):
                    raise ValueError("Next states must exist and rewards must be finite")
            if not math.isclose(sum(item.probability for item in outcomes), 1.0,
                                rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Outcome probabilities for {state}/{action} must sum to 1")


def validate_policy(model: Model, policy: Policy) -> None:
    expected = {s for s, actions in model.items() if actions}
    if set(policy) != expected:
        raise ValueError("Policy must specify every decision state and no terminal state")
    for state in expected:
        probabilities = policy[state]
        if set(probabilities) != set(model[state]):
            raise ValueError("Policy must list exactly the legal actions, including zero probabilities")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError("Policy probabilities must be finite and in [0, 1]")
        if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Policy probabilities must sum to 1")


def validate_gamma(gamma: float, finite: bool = False) -> None:
    upper = gamma <= 1 if finite else gamma < 1
    if not math.isfinite(gamma) or gamma < 0 or not upper:
        domain = "[0, 1]" if finite else "[0, 1)"
        raise ValueError(f"gamma must be finite and in {domain}")


def validate_iterations(tolerance: float, max_iterations: int) -> None:
    if not math.isfinite(tolerance) or tolerance <= 0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive and finite")


def make_model(name: str) -> Model:
    if name in ("treasure", "stochastic"):
        if name == "treasure":
            chest = [Outcome(1.0, "terminal", 8.0)]
            terminals = {"terminal": {}}
            coin_end, trap_end = "terminal", "terminal"
        else:
            chest = [Outcome(0.75, "win", 8.0), Outcome(0.25, "lose", -4.0)]
            terminals = {"win": {}, "lose": {}}
            coin_end, trap_end = "win", "lose"
        model = {
            "start": {"coin": [Outcome(1.0, coin_end, 3.0)],
                      "cave": [Outcome(1.0, "cave", -1.0)]},
            "cave": {"chest": chest, "trap": [Outcome(1.0, trap_end, -4.0)]},
            **terminals,
        }
    elif name == "career":
        model = {
            "basic": {"rest": [Outcome(1.0, "basic", 1.0)],
                      "invest": [Outcome(1.0, "trained", 0.0)]},
            "trained": {"work": [Outcome(1.0, "trained", 2.0)],
                        "reset": [Outcome(1.0, "basic", 0.0)]},
        }
    else:
        raise ValueError(f"Unknown model: {name}")
    validate_model(model)
    return model


def deterministic_policy(model: Model, actions: dict[str, str]) -> Policy:
    expected = {s for s, legal in model.items() if legal}
    if set(actions) != expected or any(actions[s] not in model[s] for s in expected):
        raise ValueError("Choose exactly one legal action for every decision state")
    return {s: {a: float(a == actions[s]) for a in model[s]} for s in actions}


def initial_policy(model: Model) -> Policy:
    return deterministic_policy(model, {s: next(iter(actions)) for s, actions in model.items() if actions})


def uniform_policy(model: Model) -> Policy:
    return {s: {a: 1 / len(actions) for a in actions} for s, actions in model.items() if actions}


def action_values(model: Model, values: Values, gamma: float) -> dict[str, dict[str, float]]:
    """只做一次模型期望备份；进入终止的奖励照计，终止后续值为零。"""
    return {
        s: {a: sum(item.probability * (
            item.reward + gamma * (values[item.next_state] if model[item.next_state] else 0.0)
        ) for item in outcomes) for a, outcomes in actions.items()}
        for s, actions in model.items() if actions
    }


def bellman_backup(model: Model, values: Values, gamma: float, policy: Policy | None = None) -> Values:
    q_values = action_values(model, values, gamma)
    return {
        s: (sum(policy[s][a] * q for a, q in q_values[s].items()) if policy is not None
            else max(q_values[s].values())) if actions else 0.0
        for s, actions in model.items()
    }


def bellman_residual(model: Model, values: Values, gamma: float,
                     policy: Policy | None = None) -> float:
    updated = bellman_backup(model, values, gamma, policy)
    return max(abs(updated[s] - values[s]) for s in model)


def iterative_evaluation(model: Model, policy: Policy, gamma: float,
                         tolerance: float = 1e-10, max_iterations: int = 100_000) -> dict:
    validate_model(model)
    validate_policy(model, policy)
    validate_gamma(gamma)
    validate_iterations(tolerance, max_iterations)
    values = dict.fromkeys(model, 0.0)
    for iteration in range(1, max_iterations + 1):
        values = bellman_backup(model, values, gamma, policy)
        residual = bellman_residual(model, values, gamma, policy)
        if residual <= tolerance:
            return {"values": values, "iterations": iteration, "residual": residual,
                    "value_error_bound": residual / (1 - gamma)}
    raise RuntimeError("Policy evaluation did not reach its residual tolerance")


def solve_linear(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """小型教学求解器：高斯消元与部分主元选择，不显式计算逆矩阵。"""
    size = len(rhs)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("Linear system must be square")
    augmented = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, rhs)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ValueError("Singular or numerically ill-conditioned linear system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(column + 1, size):
            factor = augmented[row][column] / augmented[column][column]
            augmented[row][column] = 0.0
            for j in range(column + 1, size + 1):
                augmented[row][j] -= factor * augmented[column][j]
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        solution[row] = (augmented[row][-1] - sum(
            augmented[row][j] * solution[j] for j in range(row + 1, size)
        )) / augmented[row][row]
    return solution


def linear_evaluation(model: Model, policy: Policy, gamma: float) -> dict:
    validate_model(model)
    validate_policy(model, policy)
    validate_gamma(gamma)
    states = [s for s, actions in model.items() if actions]
    indices = {state: i for i, state in enumerate(states)}
    matrix = [[float(i == j) for j in range(len(states))] for i in range(len(states))]
    rhs = [0.0] * len(states)
    for state in states:
        row = indices[state]
        for action, probability in policy[state].items():
            for item in model[state][action]:
                mass = probability * item.probability
                rhs[row] += mass * item.reward
                if item.next_state in indices:
                    matrix[row][indices[item.next_state]] -= gamma * mass
    solution = solve_linear(matrix, rhs)
    values = {s: solution[indices[s]] if s in indices else 0.0 for s in model}
    residual = bellman_residual(model, values, gamma, policy)
    return {"values": values, "residual": residual,
            "value_error_bound": residual / (1 - gamma),
            "state_order": states, "matrix_I_minus_gamma_P": matrix, "expected_rewards": rhs}


def greedy_actions(model: Model, values: Values, gamma: float) -> dict[str, str]:
    # Python 的 max 在并列时保留字典中的第一个动作，保证一致的并列处理。
    return {s: max(q, key=q.get) for s, q in action_values(model, values, gamma).items()}


def value_iteration(model: Model, gamma: float, tolerance: float = 1e-10,
                    max_iterations: int = 100_000) -> dict:
    validate_model(model)
    validate_gamma(gamma)
    validate_iterations(tolerance, max_iterations)
    values = dict.fromkeys(model, 0.0)
    for iteration in range(1, max_iterations + 1):
        values = bellman_backup(model, values, gamma)
        residual = bellman_residual(model, values, gamma)
        if residual <= tolerance:
            actions = greedy_actions(model, values, gamma)
            evaluated = linear_evaluation(model, deterministic_policy(model, actions), gamma)
            return {"values": values, "actions": actions,
                    "q_values": action_values(model, values, gamma), "iterations": iteration,
                    "optimal_residual": residual, "value_error_bound": residual / (1 - gamma),
                    "extracted_policy_values": evaluated["values"],
                    "extracted_policy_residual": evaluated["residual"]}
    raise RuntimeError("Value iteration did not reach its residual tolerance")


def policy_iteration(model: Model, gamma: float, tolerance: float = 1e-10,
                     max_iterations: int = 100_000) -> dict:
    validate_model(model)
    validate_gamma(gamma)
    validate_iterations(tolerance, max_iterations)
    policy = initial_policy(model)
    history = []
    for iteration in range(1, max_iterations + 1):
        evaluation = linear_evaluation(model, policy, gamma)
        values = evaluation["values"]
        actions = greedy_actions(model, values, gamma)
        improved = deterministic_policy(model, actions)
        history.append({"iteration": iteration, "evaluated_policy": policy,
                        "values": values, "improved_actions": actions})
        if improved == policy:
            residual = bellman_residual(model, values, gamma)
            if residual > tolerance:
                raise RuntimeError("Policy is stable but numerical residual exceeds tolerance")
            return {"values": values, "actions": actions,
                    "q_values": action_values(model, values, gamma), "iterations": iteration,
                    "optimal_residual": residual, "policy_residual": evaluation["residual"],
                    "value_error_bound": residual / (1 - gamma), "history": history}
        policy = improved
    raise RuntimeError("Policy iteration did not stabilize within max_iterations")


def finite_horizon(model: Model, gamma: float, horizon: int) -> dict:
    validate_model(model)
    validate_gamma(gamma, finite=True)
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    values = dict.fromkeys(model, 0.0)
    layers = [{"remaining_steps": 0, "values": values, "actions": {}}]
    for remaining in range(1, horizon + 1):
        actions = greedy_actions(model, values, gamma)
        values = bellman_backup(model, values, gamma)
        layers.append({"remaining_steps": remaining, "values": values, "actions": actions})
    return {"horizon": horizon, "terminal_payoff": 0.0, "layers": layers}


def model_description(model: Model) -> dict:
    return {s: {a: [{"probability": item.probability, "next_state": item.next_state,
                    "reward": item.reward} for item in outcomes]
                for a, outcomes in actions.items()} for s, actions in model.items()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("treasure", "career", "stochastic"), default="treasure")
    parser.add_argument("--algorithm", choices=("all", "evaluate", "value", "policy", "finite"), default="all")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=100_000)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_gamma(args.gamma, finite=args.algorithm == "finite")
        validate_iterations(args.tolerance, args.max_iterations)
        if args.horizon < 0:
            raise ValueError("horizon must be nonnegative")
    except ValueError as error:
        parser.error(str(error))
    if args.output is None:
        args.output = Path("runs") / "mdp" / (
            f"{args.env}_{args.algorithm}_gamma{args.gamma:g}_h{args.horizon}"
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model = make_model(args.env)
    result = {"env": args.env, "gamma": args.gamma, "model": model_description(model)}
    if args.algorithm in ("all", "evaluate"):
        policy = uniform_policy(model)
        result["evaluation"] = {
            "policy": policy, "iterative": iterative_evaluation(
                model, policy, args.gamma, args.tolerance, args.max_iterations),
            "linear": linear_evaluation(model, policy, args.gamma),
        }
    if args.algorithm in ("all", "value"):
        result["value_iteration"] = value_iteration(model, args.gamma, args.tolerance, args.max_iterations)
    if args.algorithm in ("all", "policy"):
        result["policy_iteration"] = policy_iteration(model, args.gamma, args.tolerance, args.max_iterations)
    if args.algorithm in ("all", "finite"):
        result["finite_horizon"] = finite_horizon(model, args.gamma, args.horizon)
    config = vars(args).copy()
    config["output"] = str(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, data in (("config.json", config), ("result.json", result)):
        (args.output / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
