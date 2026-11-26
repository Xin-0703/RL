"""核对 Actor-Critic 公式、冻结时序、条件期望与实验边界。"""

import copy
import math
import random
import unittest

from actor_critic import (
    Step, analytic_values, apply_updates, batch_directions, evaluate,
    initial_parameters, mc_returns, parse_args, sample_episode, softmax,
    td_target, train,
)


def transitions():
    p = (0.5, 0.5)
    coin = Step(0, 0, 3, 0, True, False, p)
    cave = Step(0, 1, -1, 1, False, False, p)
    chest = Step(1, 0, 8, 0, True, False, p)
    trap = Step(1, 1, -4, 0, True, False, p)
    return coin, cave, chest, trap


class ActorCriticTests(unittest.TestCase):
    def test_softmax_and_initial_parameters(self):
        self.assertEqual(softmax([1000, 1000]), [0.5, 0.5])
        self.assertEqual(initial_parameters("treasure"), ([[0, 0], [0, 0]], [0, 0]))
        self.assertAlmostEqual(sum(softmax([0.1, -0.2])), 1)
        with self.assertRaises(ValueError):
            softmax([math.inf, 0])

    def test_td_terminal_and_external_truncation(self):
        terminal = Step(0, 0, 1, 1, True, False, (0.5, 0.5))
        truncated = Step(0, 0, 1, 1, False, True, (0.5, 0.5))
        self.assertEqual(td_target(terminal, [2, 6], 0.9), 1)
        self.assertAlmostEqual(td_target(truncated, [2, 6], 0.9), 6.4)
        # 终止标记下不访问不存在的下一状态。
        self.assertEqual(td_target(terminal, [2], 0.9), 1)

    def test_classroom_batch_and_frozen_parameters(self):
        coin, cave, chest, _ = transitions()
        values = [2, 5]
        before = values[:]
        ag, cg, aloss, closs = batch_directions([[coin], [cave, chest]], values, 0.9)
        self.assertEqual(values, before)
        for row, expected in zip(ag, [[-0.125, 0.125], [0.675, -0.675]]):
            for value, target in zip(row, expected):
                self.assertAlmostEqual(value, target)
        self.assertEqual(cg, [1.25, 1.5])
        self.assertAlmostEqual(closs, 3.0625)
        self.assertAlmostEqual(aloss, 2.6 * math.log(2))
        logits, new_values = apply_updates([[0, 0], [0, 0]], values, ag, cg, 0.1, 0.2)
        self.assertEqual(new_values, [2.25, 5.3])
        self.assertAlmostEqual(logits[1][0], 0.0675)

    def test_single_step_actor_sign_and_critic_semigradient(self):
        _, cave, _, _ = transitions()
        ag, cg, _, _ = batch_directions([[cave]], [2, 5], 0.9)
        z, values = apply_updates([[0, 0], [0, 0]], [2, 5], ag, cg, 0.1, 0.2)
        self.assertAlmostEqual(softmax(z[0])[1], 1 / (1 + math.exp(-0.15)))
        self.assertEqual(values, [2.3, 5])
        self.assertEqual(cg[1], 0)  # 不沿下一状态 V 求导。

    def test_mc_returns_and_no_fake_terminal(self):
        _, cave, chest, _ = transitions()
        self.assertEqual(mc_returns([cave, chest], 0.9), [6.2, 8])
        self.assertEqual(mc_returns([cave, chest], 0), [-1, 8])
        with self.assertRaises(ValueError):
            mc_returns([cave], 0.9)
        with self.assertRaises(ValueError):
            mc_returns([], 0.9)

    def test_exact_critic_td_expectation_matches_policy_gradient(self):
        coin, cave, chest, trap = transitions()
        episodes = [[coin], [cave, chest], [cave, trap]]
        probabilities = [0.5, 0.25, 0.25]
        values = [1.9, 2]
        expected_td = [[0.0, 0.0], [0.0, 0.0]]
        expected_mc = copy.deepcopy(expected_td)
        for episode, probability in zip(episodes, probabilities):
            for mode, accumulator in (("td0", expected_td), ("mc", expected_mc)):
                ag, cg, _, _ = batch_directions([episode], values, 0.9, mode)
                for s in range(2):
                    for a in range(2):
                        accumulator[s][a] += probability * ag[s][a]
        target = [[0.55, -0.55], [1.35, -1.35]]
        for estimate in (expected_td, expected_mc):
            for s in range(2):
                for a in range(2):
                    self.assertAlmostEqual(estimate[s][a], target[s][a])
                    h = 1e-6
                    plus, minus = [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]
                    plus[s][a], minus[s][a] = h, -h
                    numeric = (analytic_values("treasure", plus, 0.9)[0] -
                               analytic_values("treasure", minus, 0.9)[0]) / (2 * h)
                    self.assertAlmostEqual(estimate[s][a], numeric, places=8)

    def test_wrong_critic_bias_and_mc_baseline_cancellation(self):
        coin, cave, chest, trap = transitions()
        td, mc = [0.0, 0.0], [0.0, 0.0]
        for episode, p in zip([[coin], [cave, chest], [cave, trap]], [0.5, 0.25, 0.25]):
            for estimator, accumulator in (("td0", td), ("mc", mc)):
                ag, _, _, _ = batch_directions([episode], [1.9, 5], 0.9, estimator)
                for a in range(2):
                    accumulator[a] += p * ag[0][a]
        self.assertAlmostEqual(td[0], -0.125)
        self.assertAlmostEqual(td[1], 0.125)
        self.assertAlmostEqual(mc[0], 0.55)
        self.assertAlmostEqual(mc[1], -0.55)

    def test_gamma_zero_suppresses_later_actor_but_not_critic(self):
        _, cave, chest, _ = transitions()
        ag, cg, _, _ = batch_directions([[cave, chest]], [2, 5], 0)
        self.assertEqual(ag[1], [0, 0])
        self.assertEqual(cg[1], 3)

    def test_repeating_batch_preserves_episode_average(self):
        coin, cave, chest, _ = transitions()
        batch = [[coin], [cave, chest]]
        first = batch_directions(batch, [2, 5], 0.9)
        repeated = batch_directions(batch * 2, [2, 5], 0.9)
        self.assertEqual(first[:2], repeated[:2])
        for loss, repeated_loss in zip(first[2:], repeated[2:]):
            self.assertAlmostEqual(loss, repeated_loss)

    def test_sampling_and_evaluation_isolation(self):
        logits = [[-0.1, 0.2], [0.3, 0.0]]
        frozen = copy.deepcopy(logits)
        global_state = random.getstate()
        a = evaluate("treasure", logits, 0.9, 100, 42)
        b = evaluate("treasure", logits, 0.9, 100, 42)
        self.assertEqual(a, b)
        self.assertEqual(logits, frozen)
        self.assertEqual(global_state, random.getstate())
        episode = sample_episode("treasure", logits, random.Random(1))
        self.assertTrue(episode[-1].terminated)
        self.assertLessEqual(len(episode), 2)

    def test_last_partial_batch_and_reproducibility(self):
        args = parse_args(["--episodes", "21", "--batch-size", "20", "--eval-episodes", "20"])
        first, curve = train(args)
        second, _ = train(args)
        self.assertEqual(first, second)
        self.assertEqual(first["episodes"], 21)
        self.assertEqual(first["updates"], 2)
        self.assertEqual(curve[-1]["episodes"], 21)

    def test_argument_validation(self):
        import contextlib
        import io
        for argv in (["--gamma", "nan"], ["--gamma", "1.1"],
                     ["--actor-lr", "0"], ["--critic-lr", "inf"],
                     ["--episodes", "0"], ["--batch-size", "0"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)


if __name__ == "__main__":
    unittest.main()
