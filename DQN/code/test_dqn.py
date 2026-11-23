"""针对 DQN 易错机制的确定性测试；不以随机训练分数作为单元测试。"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from train import (
    Config, ReplayBuffer, TreasureEnv, compute_targets, env_dimensions, optimize,
)


class DQNMechanismTests(unittest.TestCase):
    def test_environment_dimensions_are_builtin_ints_for_checkpoint(self):
        env = SimpleNamespace(
            observation_space=SimpleNamespace(shape=(np.int64(4),)),
            action_space=SimpleNamespace(n=np.int64(2)),
        )
        state_dim, action_dim = env_dimensions(env, "cartpole")
        self.assertIs(type(state_dim), int)
        self.assertIs(type(action_dim), int)
        self.assertEqual((state_dim, action_dim), (4, 2))

    def test_terminal_mask_and_nonterminal_bootstrap(self):
        target_net = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            target_net.weight.copy_(torch.tensor([[4.0, 5.0], [-1.0, 2.0]]))
        targets = compute_targets(
            target_net,
            torch.tensor([8.0, -1.0]),
            torch.eye(2),
            torch.tensor([1.0, 0.0]),
            0.9,
        )
        torch.testing.assert_close(targets, torch.tensor([8.0, 3.5]))
        self.assertFalse(targets.requires_grad)

    def test_external_truncation_retains_bootstrap(self):
        # 真正终止和单纯外部截断使用相同观测，只有真正终止屏蔽未来价值。
        target_net = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            target_net.weight.fill_(6.0)
        terminated = torch.tensor([1.0, 0.0])
        targets = compute_targets(
            target_net, torch.ones(2), torch.ones(2, 1), terminated, 0.9
        )
        torch.testing.assert_close(targets, torch.tensor([1.0, 6.4]))

    def test_only_recorded_actions_are_supervised_and_target_is_frozen(self):
        online = nn.Linear(2, 2, bias=False)
        target_net = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            online.weight.copy_(torch.tensor([[2.0, 7.0], [9.0, -2.0]]))
        target_before = {k: v.clone() for k, v in target_net.state_dict().items()}
        replay = ReplayBuffer(10, seed=0)
        replay.add([1, 0], 0, 3, [0, 0], True)
        replay.add([0, 1], 1, -4, [0, 0], True)
        config = Config(batch_size=2)
        optimizer = torch.optim.SGD(online.parameters(), lr=0.5)
        loss = optimize(online, target_net, optimizer, replay, config)
        # 目标 [3,-4]，当前选中预测 [2,-2]：loss=(1+4)/4=1.25。
        self.assertAlmostEqual(loss, 1.25)
        # 批量求均值，所以每个独立参数的更新系数为 lr / B = 0.25。
        torch.testing.assert_close(
            online.weight, torch.tensor([[2.25, 7.0], [9.0, -2.5]])
        )
        for name, value in target_net.state_dict().items():
            torch.testing.assert_close(value, target_before[name])
        self.assertTrue(all(p.grad is None for p in target_net.parameters()))

    def test_replay_copies_observations_and_limits_capacity(self):
        replay = ReplayBuffer(2, seed=0)
        state = np.array([1.0, 0.0], dtype=np.float32)
        replay.add(state, 1, -1, state, False)
        state[:] = 99.0
        np.testing.assert_array_equal(replay.data[0].state, [1.0, 0.0])
        np.testing.assert_array_equal(replay.data[0].next_state, [1.0, 0.0])
        replay.add([0, 1], 0, 8, [0, 1], True)
        replay.add([1, 0], 0, 3, [1, 0], True)
        self.assertEqual(len(replay), 2)
        states, actions, rewards, next_states, terminated = replay.sample(2)
        self.assertEqual(states.shape, (2, 2))
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(actions.dtype, torch.long)
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(next_states.shape, (2, 2))
        self.assertEqual(terminated.shape, (2,))

    def test_treasure_rewards_and_episode_boundary(self):
        env = TreasureEnv()
        state, _ = env.reset()
        np.testing.assert_array_equal(state, [1.0, 0.0])
        _, reward, terminated, truncated, _ = env.step(1)
        self.assertEqual((reward, terminated, truncated), (-1.0, False, False))
        _, reward, terminated, truncated, _ = env.step(0)
        self.assertEqual((reward, terminated, truncated), (8.0, True, False))
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset()
        self.assertEqual(env.step(0)[1:4], (3.0, True, False))
        env.reset()
        env.step(1)
        self.assertEqual(env.step(1)[1:4], (-4.0, True, False))


if __name__ == "__main__":
    unittest.main()
