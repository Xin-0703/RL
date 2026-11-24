"""用枚举期望、有限差分与确定轨迹检查策略梯度。"""

import contextlib
import copy
import io
import math
import random
import unittest

from reinforce import (
    Episode, HistoricalBaseline, IncompleteEpisodeError, Step, analytic_gradient,
    analytic_objective, batch_gradient, evaluate, gradient_check, initial_logits,
    parse_args, returns, sample_episode, softmax, train, update_logits,
)


class ReinforceTests(unittest.TestCase):
    def test_stable_softmax_shift_invariance(self):
        self.assertEqual(softmax([1000, 1000]), [0.5, 0.5])
        for a, b in zip(softmax([1, 2]), softmax([1001, 1002])):
            self.assertAlmostEqual(a, b)
        with self.assertRaises(ValueError):
            softmax([float("nan"), 0])

    def test_log_probability_score_finite_difference(self):
        z, delta = [0.2, -0.3], 1e-6
        p = softmax(z)
        for action in (0, 1):
            for parameter in (0, 1):
                plus, minus = z[:], z[:]
                plus[parameter] += delta
                minus[parameter] -= delta
                numeric = (math.log(softmax(plus)[action]) - math.log(softmax(minus)[action])) / (2 * delta)
                self.assertAlmostEqual(numeric, float(action == parameter) - p[parameter])

    def test_returns_and_explicit_time_discount(self):
        episode = Episode((Step(0, 1, -1, (0.5, 0.5)), Step(1, 0, 8, (0.5, 0.5))))
        self.assertEqual(returns(episode, 0.9), [6.2, 8])
        gradient = batch_gradient([episode], 0.9, [0, 0])
        self.assertEqual(gradient, [[-3.1, 3.1], [3.6, -3.6]])
        self.assertEqual(batch_gradient([episode], 0, [0, 0])[1], [0, 0])

    def test_batch_divides_by_episodes_not_steps(self):
        coin = Episode((Step(0, 0, 3, (0.5, 0.5)),))
        chest = Episode((Step(0, 1, -1, (0.5, 0.5)), Step(1, 0, 8, (0.5, 0.5))))
        result = batch_gradient([coin, chest], 0.9, [0, 0])
        self.assertAlmostEqual(result[0][0], -0.8)
        self.assertEqual(result[1], [1.8, -1.8])

    def test_gradient_ascent_increases_chosen_probability(self):
        logits = [[0, 0]]
        updated = update_logits(logits, [[-1, 1]], 0.1)
        self.assertGreater(softmax(updated[0])[1], 0.5)
        self.assertEqual(logits, [[0, 0]])

    def test_enumerated_treasure_gradient_matches_analytic_with_baseline(self):
        logits, gamma = [[0.2, -0.3], [-0.1, 0.4]], 0.7
        p, q = [softmax(row) for row in logits]
        paths = [
            (p[0], Episode((Step(0, 0, 3, tuple(p)),))),
            (p[1] * q[0], Episode((Step(0, 1, -1, tuple(p)), Step(1, 0, 8, tuple(q))))),
            (p[1] * q[1], Episode((Step(0, 1, -1, tuple(p)), Step(1, 1, -4, tuple(q))))),
        ]
        exact = analytic_gradient("treasure", logits, gamma)
        for baseline in ([0, 0], [2.3, -5]):
            expected = [[0.0, 0.0], [0.0, 0.0]]
            for probability, episode in paths:
                g = batch_gradient([episode], gamma, baseline)
                for s in range(2):
                    for a in range(2):
                        expected[s][a] += probability * g[s][a]
            for s in range(2):
                for a in range(2):
                    self.assertAlmostEqual(expected[s][a], exact[s][a])

    def test_enumerated_stochastic_bandit_gradient(self):
        logits = [[0.2, -0.3]]
        p = softmax(logits[0])
        expected = [0, 0]
        for action, success in enumerate((0.2, 0.8)):
            for reward, chance in ((1, success), (0, 1 - success)):
                g = batch_gradient([Episode((Step(0, action, reward, tuple(p)),))], 0.9, [0.5])
                for a in (0, 1):
                    expected[a] += p[action] * chance * g[0][a]
        for actual, exact in zip(expected, analytic_gradient("bandit", logits, 0.9)[0]):
            self.assertAlmostEqual(actual, exact)

    def test_objective_gradient_finite_difference_both_envs(self):
        for env in ("bandit", "treasure"):
            self.assertLess(gradient_check(env, 0.9)["max_abs_error"], 1e-8)
        self.assertAlmostEqual(analytic_objective("treasure", [[0, 0], [0, 0]], 0.9), 1.9)

    def test_history_baseline_is_frozen_before_current_batch(self):
        baseline = HistoricalBaseline(1)
        frozen = baseline.values[:]
        batch = [Episode((Step(0, 1, 1, (0.5, 0.5)),))]
        g = batch_gradient(batch, 0.9, frozen)
        self.assertEqual(g, [[-0.5, 0.5]])
        baseline.fit_after_batch(batch, 0.9)
        self.assertEqual(baseline.values, [1])
        self.assertEqual(baseline.counts, [1])
        self.assertEqual(frozen, [0])
        # 若错误用本次回报当基线，本样本梯度就变成 0。
        self.assertEqual(batch_gradient(batch, 0.9, baseline.values), [[0, 0]])

    def test_incomplete_episode_and_batch_are_rejected(self):
        with self.assertRaises(IncompleteEpisodeError):
            sample_episode("treasure", [[-1000, 0], [0, 0]], random.Random(0), max_steps=1)
        with self.assertRaises(IncompleteEpisodeError):
            batch_gradient([Episode((Step(0, 1, -1, (0.5, 0.5)),), False)], 0.9, [0, 0])

    def test_training_uses_last_partial_batch_and_counts_every_visit(self):
        args = parse_args(["--env", "bandit", "--episodes", "5", "--batch-size", "2", "--eval-episodes", "10"])
        result, curve = train(args)
        self.assertEqual(result["updates"], 3)
        self.assertEqual(result["episodes"], 5)
        self.assertEqual(result["baseline_counts"], [5])
        self.assertEqual(curve[-1]["episodes"], 5)

    def test_evaluation_does_not_mutate_policy_and_is_reproducible(self):
        logits = initial_logits("treasure")
        before = copy.deepcopy(logits)
        first = evaluate("treasure", logits, 0.9, 100, 123)
        self.assertEqual(first, evaluate("treasure", logits, 0.9, 100, 123))
        self.assertFalse(first["greedy"])
        self.assertEqual(logits, before)

    def test_invalid_cli(self):
        for argv in (["--lr", "nan"], ["--lr", "0"], ["--gamma", "2"],
                     ["--episodes", "0"], ["--batch-size", "0"], ["--max-steps", "0"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv)


if __name__ == "__main__":
    unittest.main()
