"""标准 DQN 教学实现：宝箱环境（默认）和可选 CartPole。

从仓库根目录运行：python DQN/code/train.py --env treasure
公式、安装方式和输出说明见同目录 README.md。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool  # 只屏蔽真正终止；单纯时间截断仍保留自举。


class TreasureEnv:
    """两状态、两动作的课堂环境，接口与本实验所需 Gymnasium 接口一致。"""

    state_dim = 2
    action_dim = 2

    def __init__(self):
        self.state = 0
        self.ended = True

    def observation(self):
        return np.eye(2, dtype=np.float32)[self.state].copy()

    def reset(self, *, seed=None):
        # 本环境的转移完全确定，seed 为兼容 Gymnasium 的接口而保留。
        self.state = 0
        self.ended = False
        return self.observation(), {}

    def step(self, action):
        if self.ended:
            raise RuntimeError("回合已结束，请先 reset()。")
        if action not in (0, 1):
            raise ValueError("动作必须为 0 或 1。")
        if self.state == 0:
            reward = 3.0 if action == 0 else -1.0
            terminated = action == 0
            if action == 1:
                self.state = 1
        else:
            reward = 8.0 if action == 0 else -4.0
            terminated = True
        self.ended = terminated
        # 终止观测保留有效形状；其价值由 terminated 掩码屏蔽。
        return self.observation(), reward, terminated, False, {}

    def close(self):
        pass


def make_env(name):
    if name == "treasure":
        return TreasureEnv()
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError(
            "CartPole 需要 gymnasium。请运行 python -m pip install -r "
            "DQN/code/requirements.txt"
        ) from exc
    return gym.make("CartPole-v1")  # 无图形渲染，无需 pygame。


def env_dimensions(env, name):
    if name == "treasure":
        return env.state_dim, env.action_dim
    # Gymnasium 的 n 可能是 NumPy 整数；检查点元数据统一使用 Python int，
    # 以兼容 torch.load(weights_only=True) 的受限反序列化。
    return int(env.observation_space.shape[0]), int(env.action_space.n)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),  # Q 可以为负，不使用 softmax。
        )

    def forward(self, states):
        return self.layers(states)


class ReplayBuffer:
    def __init__(self, capacity, seed):
        self.data = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.data)

    def add(self, state, action, reward, next_state, terminated):
        # 复制观测，防止环境重用数组时覆盖历史经验。
        self.data.append(
            Transition(
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                bool(terminated),
            )
        )

    def sample(self, batch_size):
        samples = self.rng.sample(list(self.data), batch_size)
        return (
            torch.from_numpy(np.stack([t.state for t in samples])),
            torch.tensor([t.action for t in samples], dtype=torch.long),
            torch.tensor([t.reward for t in samples], dtype=torch.float32),
            torch.from_numpy(np.stack([t.next_state for t in samples])),
            torch.tensor([t.terminated for t in samples], dtype=torch.float32),
        )


@dataclass
class Config:
    env: str = "treasure"
    episodes: int = 600
    gamma: float = 0.9
    learning_rate: float = 0.001
    batch_size: int = 32
    capacity: int = 2000
    warmup: int = 32
    target_every: int = 50  # 单位：梯度更新次数。
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 400  # 单位：环境交互步数。
    eval_episodes: int = 20
    seed: int = 0


def epsilon_at(step, config):
    fraction = min(step / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def select_action(online, state, epsilon, action_dim, rng):
    if rng.random() < epsilon:
        return rng.randrange(action_dim)
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        return int(online(state_tensor).argmax(dim=1).item())


@torch.no_grad()
def compute_targets(target_net, rewards, next_states, terminated, gamma):
    """标准 DQN：由目标网络同时选出并评价下一状态的最大动作价值。"""
    next_q = target_net(next_states).max(dim=1).values
    return rewards + gamma * (1.0 - terminated) * next_q


def optimize(online, target_net, optimizer, replay, config):
    states, actions, rewards, next_states, terminated = replay.sample(
        config.batch_size
    )
    targets = compute_targets(
        target_net, rewards, next_states, terminated, config.gamma
    )
    # online(states): [B,A]；gather: [B,1]；squeeze 后与目标同为 [B]。
    current_q = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = 0.5 * (targets - current_q).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def evaluate(online, config, episodes):
    # 独立环境、局部随机数发生器，不消耗训练环境或探索 RNG 的状态。
    env = make_env(config.env)
    _, action_dim = env_dimensions(env, config.env)
    rng = random.Random(config.seed + 20_000)
    returns, discounted_returns = [], []
    first_trace = []
    was_training = online.training
    online.eval()
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=config.seed + 10_000 + episode)
            total, discounted, discount = 0.0, 0.0, 1.0
            while True:
                action = select_action(online, state, 0.0, action_dim, rng)
                next_state, reward, terminated, truncated, _ = env.step(action)
                if episode == 0:
                    first_trace.append({
                        "state": np.asarray(state).tolist(),
                        "action": action,
                        "reward": float(reward),
                    })
                total += float(reward)
                discounted += discount * float(reward)
                discount *= config.gamma
                state = next_state
                if terminated or truncated:
                    break
            returns.append(total)
            discounted_returns.append(discounted)
    finally:
        env.close()
        online.train(was_training)
    result = {
        "episodes": episodes,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),  # 总体标准差 ddof=0。
        "discounted_return_mean": float(np.mean(discounted_returns)),
        "discounted_return_std": float(np.std(discounted_returns)),
        "first_episode_trace": first_trace,
    }
    if config.env == "treasure":
        with torch.no_grad():
            result["q_values"] = online(torch.eye(2)).tolist()
        result["optimal_q_values"] = [
            [3.0, -1.0 + config.gamma * 8.0], [8.0, -4.0]
        ]
    return result


def print_evaluation(result):
    """控制台仅显示短轨迹；完整轨迹保存在 evaluation.json 中。"""
    summary = dict(result)
    trace = result["first_episode_trace"]
    if len(trace) > 10:
        summary["first_episode_trace"] = f"{len(trace)} steps (omitted from console)"
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def train(config, output):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # 小网络采用 CPU 单线程，减少线程调度开销并便于重复实验。
    torch.set_num_threads(1)
    env = make_env(config.env)
    state_dim, action_dim = env_dimensions(env, config.env)
    online = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(online.state_dict())
    target_net.requires_grad_(False)
    target_net.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.capacity, config.seed + 1)
    action_rng = random.Random(config.seed + 2)
    total_steps, updates = 0, 0
    output.mkdir(parents=True, exist_ok=True)
    recent_returns = deque(maxlen=50)
    try:
        with (output / "training.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "steps", "updates", "epsilon", "return",
                "discounted_return", "mean_loss",
            ])
            for episode in range(config.episodes):
                state, _ = env.reset(seed=config.seed + episode)
                total, discounted, discount = 0.0, 0.0, 1.0
                losses = []
                while True:
                    epsilon = epsilon_at(total_steps, config)
                    action = select_action(
                        online, state, epsilon, action_dim, action_rng
                    )
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    replay.add(state, action, reward, next_state, terminated)
                    total_steps += 1
                    if len(replay) >= max(config.warmup, config.batch_size):
                        losses.append(optimize(
                            online, target_net, optimizer, replay, config
                        ))
                        updates += 1
                        if updates % config.target_every == 0:
                            target_net.load_state_dict(online.state_dict())
                    total += float(reward)
                    discounted += discount * float(reward)
                    discount *= config.gamma
                    state = next_state
                    if terminated or truncated:
                        break
                recent_returns.append(total)
                writer.writerow([
                    episode + 1, total_steps, updates, epsilon, total, discounted,
                    float(np.mean(losses)) if losses else "",
                ])
                if (episode + 1) % 50 == 0 or episode + 1 == config.episodes:
                    print(
                        f"episode={episode + 1} steps={total_steps} "
                        f"updates={updates} epsilon={epsilon:.3f} "
                        f"recent_return={np.mean(recent_returns):.2f}",
                        flush=True,
                    )
    finally:
        env.close()

    result = evaluate(online, config, config.eval_episodes)
    result.update({"training_steps": total_steps, "gradient_updates": updates})
    checkpoint = {
        "config": asdict(config),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "model": online.state_dict(),
    }
    # 保存的是用于推理的模型，不包含恢复训练所需的优化器与回放池。
    torch.save(checkpoint, output / "model.pt")
    (output / "evaluation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    print_evaluation(result)
    print(f"Saved artifacts to {output.resolve()}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["treasure", "cartpole"], default="treasure")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluate", type=Path, metavar="MODEL_PT")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed 必须非负。")
    if args.eval_episodes is not None and args.eval_episodes <= 0:
        parser.error("--eval-episodes 必须大于 0。")

    if args.evaluate is not None:
        if args.episodes is not None or args.gamma is not None:
            parser.error("评估时环境和 gamma 由模型恢复，请勿传入 --episodes/--gamma。")
        torch.set_num_threads(1)
        checkpoint = torch.load(args.evaluate, map_location="cpu", weights_only=True)
        config = Config(**checkpoint["config"])
        model = QNetwork(checkpoint["state_dim"], checkpoint["action_dim"])
        model.load_state_dict(checkpoint["model"])
        result = evaluate(model, config, args.eval_episodes or config.eval_episodes)
        print_evaluation(result)
        return

    config = Config(env=args.env, seed=args.seed)
    if args.env == "cartpole":
        config.episodes = 400
        config.gamma = 0.99
        config.batch_size = 64
        config.capacity = 50_000
        config.warmup = 1000
        config.target_every = 250
        config.epsilon_decay_steps = 10_000
    if args.episodes is not None:
        config.episodes = args.episodes
    if args.gamma is not None:
        config.gamma = args.gamma
    if args.eval_episodes is not None:
        config.eval_episodes = args.eval_episodes
    if config.episodes <= 0:
        parser.error("--episodes 必须大于 0。")
    if not 0 <= config.gamma <= 1:
        parser.error("--gamma 必须在 [0,1] 范围内。")
    output = args.output or Path("runs") / f"{config.env}_seed{config.seed}"
    train(config, output)


if __name__ == "__main__":
    main()
