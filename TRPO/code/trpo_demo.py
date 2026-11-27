"""TRPO 优化器教学实验：精确期望二元上下文 bandit，Python 3.10+ 标准库。

每个上下文有一个独立 logit；不采样轨迹，不训练 Critic。
Fisher 用解析向量积，CG 与回溯流程可直接对照讲义。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Problem:
    weights: tuple[float, ...]
    rewards: tuple[tuple[float, float], ...]

    def __post_init__(self):
        if not self.weights or len(self.weights) != len(self.rewards):
            raise ValueError("weights and rewards must have the same nonzero length")
        if (not all(math.isfinite(w) and w > 0 for w in self.weights)
                or not math.isclose(sum(self.weights), 1.0, abs_tol=1e-12)):
            raise ValueError("positive context weights must sum to one")
        if not all(len(r) == 2 and all(math.isfinite(v) for v in r)
                   for r in self.rewards):
            raise ValueError("each context needs two finite rewards")

    def check(self, theta):
        if len(theta) != len(self.weights) or not all(math.isfinite(t) for t in theta):
            raise ValueError("one finite logit per context is required")


def sigmoid(z):
    if not math.isfinite(z):
        raise ValueError("logit must be finite")
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def softplus(z):
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def log_probabilities(theta):
    # 返回 log P(a=0), log P(a=1)，避免 log(sigmoid) 的下溢。
    return -softplus(theta), -softplus(-theta)


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def objective(problem, theta):
    problem.check(theta)
    return sum(w * ((1 - sigmoid(t)) * r[0] + sigmoid(t) * r[1])
               for w, r, t in zip(problem.weights, problem.rewards, theta))


def surrogate(problem, old, candidate):
    problem.check(old)
    problem.check(candidate)
    total = 0.0
    for w, rewards, t0, t1 in zip(problem.weights, problem.rewards, old, candidate):
        p0, p1 = sigmoid(t0), sigmoid(t1)
        v0 = (1 - p0) * rewards[0] + p0 * rewards[1]
        advantages = (rewards[0] - v0, rewards[1] - v0)
        # sum_a pi_old(a) * [pi_new(a)/pi_old(a)] * A_old(a)
        # 精确动作求和后约去旧概率；不需要显式计算可能溢出的比率。
        total += w * ((1 - p1) * advantages[0] + p1 * advantages[1])
    return total


def mean_kl(problem, old, candidate):
    problem.check(old)
    problem.check(candidate)
    total = 0.0
    for w, t0, t1 in zip(problem.weights, old, candidate):
        p0 = sigmoid(t0)
        log0, log1 = log_probabilities(t0), log_probabilities(t1)
        total += w * ((1 - p0) * (log0[0] - log1[0])
                      + p0 * (log0[1] - log1[1]))
    return max(0.0, total)  # KL 在相等策略处的微小负舍入误差。


def gradient(problem, old):
    problem.check(old)
    return [w * sigmoid(t) * (1 - sigmoid(t)) * (r[1] - r[0])
            for w, r, t in zip(problem.weights, problem.rewards, old)]


def fisher_vector_product(problem, old, vector, damping=0.0):
    problem.check(old)
    if len(vector) != len(old) or not all(math.isfinite(v) for v in vector):
        raise ValueError("invalid vector")
    if not math.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and nonnegative")
    # 独立上下文 logit 的 F 为对角阵；不显式构造矩阵。
    return [(w * sigmoid(t) * (1 - sigmoid(t)) + damping) * v
            for w, t, v in zip(problem.weights, old, vector)]


def conjugate_gradient(matvec, b, max_iters=10, tolerance=1e-12):
    """近似解 Ax=b；tolerance 是相对残差范数阈值。非正曲率时失败。"""
    if max_iters < 1 or not 0 < tolerance < 1:
        raise ValueError("invalid CG settings")
    x, r, p = [0.0] * len(b), list(b), list(b)
    rr = dot(r, r)
    if rr == 0:
        return x
    threshold = rr * tolerance**2
    for _ in range(max_iters):
        ap = matvec(p)
        curvature = dot(p, ap)
        if not math.isfinite(curvature) or curvature <= 0:
            raise ArithmeticError("nonpositive or nonfinite CG curvature")
        alpha = rr / curvature
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * ai for ri, ai in zip(r, ap)]
        rr_new = dot(r, r)
        if not math.isfinite(rr_new) or not all(math.isfinite(v) for v in x):
            raise ArithmeticError("nonfinite CG result")
        if rr_new <= threshold:
            return x
        p = [ri + rr_new / rr * pi for ri, pi in zip(r, p)]
        rr = rr_new
    return x


def trpo_step(problem, old, delta=0.01, damping=0.0, cg_iters=10,
              backtrack=0.5, max_backtracks=10, accept_ratio=0.1):
    problem.check(old)
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    if not math.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and nonnegative")
    if not 0 < backtrack < 1 or max_backtracks < 1 or cg_iters < 1:
        raise ValueError("invalid iteration or backtracking settings")
    if not math.isfinite(accept_ratio) or not 0 <= accept_ratio < 1:
        raise ValueError("accept_ratio must be in [0,1)")

    old = list(old)  # 所有候选从同一份旧参数出发，绝不原地修改调用者输入。
    info = {"accepted": False, "fraction": 0.0, "kl": 0.0,
            "gain": 0.0, "expected_gain": 0.0, "attempts": []}
    g = gradient(problem, old)
    if not all(math.isfinite(v) for v in g):
        info["reason"] = "nonfinite_gradient"
        return old, info
    if dot(g, g) == 0:
        info["reason"] = "zero_gradient"
        return old, info

    def matvec(v):
        return fisher_vector_product(problem, old, v, damping)

    try:
        direction = conjugate_gradient(matvec, g, max_iters=cg_iters)
        q = dot(direction, matvec(direction))
        if not math.isfinite(q) or q <= 0:
            raise ArithmeticError("invalid scaling curvature")
        full_step = [math.sqrt(2 * delta / q) * x for x in direction]
        expected = dot(g, full_step)
        if not math.isfinite(expected) or expected <= 0:
            raise ArithmeticError("invalid predicted improvement")
    except ArithmeticError:
        info["reason"] = "invalid_curvature_or_solve"
        return old, info

    base = surrogate(problem, old, old)
    for j in range(max_backtracks):
        fraction = backtrack**j
        candidate = [t + fraction * s for t, s in zip(old, full_step)]
        if not all(math.isfinite(v) for v in candidate):
            info["attempts"].append({"fraction": fraction, "finite": False})
            continue
        kl = mean_kl(problem, old, candidate)
        gain = surrogate(problem, old, candidate) - base
        predicted = fraction * expected
        finite = all(math.isfinite(v) for v in (kl, gain, predicted))
        accepted = finite and kl <= delta and gain > 0 and gain >= accept_ratio * predicted
        info["attempts"].append({"fraction": fraction, "kl": kl, "gain": gain,
                                 "expected_gain": predicted, "accepted": accepted})
        if accepted:
            info.update(accepted=True, fraction=fraction, kl=kl, gain=gain,
                        expected_gain=predicted, reason="accepted")
            return candidate, info
    info["reason"] = "line_search_failed"
    return old, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("single", "contextual"), default="single")
    parser.add_argument("--initial-p", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--cg-iters", type=int, default=10)
    parser.add_argument("--max-backtracks", type=int, default=10)
    args = parser.parse_args()
    if not 0 < args.initial_p < 1 or args.iterations < 1:
        parser.error("initial-p must be in (0,1); iterations must be positive")

    problem = (Problem((1.0,), ((0.0, 2.0),)) if args.problem == "single"
               else Problem((0.5, 0.5), ((0.0, 2.0), (2.0, 0.0))))
    theta = [math.log(args.initial_p / (1 - args.initial_p))] * len(problem.weights)
    history = []
    for iteration in range(1, args.iterations + 1):
        old = theta[:]
        try:
            theta, info = trpo_step(problem, old, delta=args.delta, damping=args.damping,
                                    cg_iters=args.cg_iters, max_backtracks=args.max_backtracks)
        except ValueError as exc:
            parser.error(str(exc))
        history.append({"iteration": iteration, "old_objective": objective(problem, old),
                        "objective": objective(problem, theta), "theta": theta[:],
                        "action1_probabilities": [sigmoid(t) for t in theta], **info})
    print(json.dumps({"problem": asdict(problem), "config": vars(args), "history": history},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
