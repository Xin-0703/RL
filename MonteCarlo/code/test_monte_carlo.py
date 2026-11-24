"""用确定性手算结果检查算法机制，不用随机训练分数充当正确性证明。"""

import contextlib
import copy
import io
import random
import unittest

from monte_carlo import (
    Episode, ImportanceMean, IncompleteEpisodeError, MeanTable, RandomWalk,
    Step, Treasure, discounted_returns, epsilon_greedy_probabilities,
    evaluate_treasure, parse_args, prediction_metrics, sample_episode,
    target_treasure_policy, trajectory_weight, update_episode,
)


def episode(items, probability=0.5, terminated=True):
    return Episode(tuple(Step(s, a, r, probability) for s, a, r in items), terminated)


class MonteCarloTests(unittest.TestCase):
    def test_discount_and_terminal_return(self):
        self.assertEqual(discounted_returns([2, 3, 5], 0.9), [8.75, 7.5, 5])
        self.assertEqual(discounted_returns([2, 3, 5], 0), [2, 3, 5])
        self.assertEqual(discounted_returns([2, 3, 5], 1), [10, 8, 5])

    def test_first_every_and_seen_reset(self):
        episodes = [episode([(0, 0, 1), (1, 0, 2), (0, 0, 3)]),
                    episode([(0, 0, 0), (1, 0, 4)])]
        first, every = MeanTable(), MeanTable()
        for item in episodes:
            update_episode(first, item, 1, "first")
            update_episode(every, item, 1, "every")
        self.assertEqual(first.values, {0: 5.0, 1: 4.5})
        self.assertEqual(first.counts, {0: 2, 1: 2})
        self.assertAlmostEqual(every.value(0), 13 / 3)
        self.assertEqual(every.counts, {0: 3, 1: 2})

    def test_first_visit_q_uses_state_action_pair(self):
        table = MeanTable()
        update_episode(table, episode([(0, 0, 1), (0, 1, 2), (0, 0, 3)]),
                       1, action_value=True)
        self.assertEqual(table.values, {(0, 0): 6.0, (0, 1): 5.0})
        self.assertEqual(table.counts, {(0, 0): 1, (0, 1): 1})

    def test_control_classroom_trajectory(self):
        table = MeanTable()
        for items in ([(0, 1, -1), (1, 1, -4)], [(0, 0, 3)],
                      [(0, 1, -1), (1, 0, 8)]):
            update_episode(table, episode(items), 0.9, action_value=True)
        self.assertAlmostEqual(table.value((0, 1)), 0.8)
        self.assertEqual(table.counts[(0, 1)], 2)
        self.assertEqual(table.value((1, 0)), 8)

    def test_epsilon_greedy_and_ties(self):
        for actual, expected in zip(epsilon_greedy_probabilities([5, 5, 1], 0.3), [0.45, 0.45, 0.1]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(epsilon_greedy_probabilities([0, 0], 0), [0.5, 0.5])
        self.assertEqual(epsilon_greedy_probabilities([5, 1], 1), [0.5, 0.5])
        self.assertEqual(epsilon_greedy_probabilities([5, 1], 0), [1, 0])

    def test_environment_rewards_and_natural_termination(self):
        env = Treasure()
        self.assertEqual(env.reset(), 0)
        self.assertEqual(env.step(1), (1, -1, False))
        self.assertEqual(env.step(0), (2, 8, True))
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset()
        self.assertEqual(env.step(0), (2, 3, True))
        env.reset()
        env.step(1)
        self.assertEqual(env.step(1), (2, -4, True))
        for action, final in ((0, (0, 0.0, True)), (1, (6, 1.0, True))):
            env = RandomWalk()
            env.reset()
            for _ in range(2):
                self.assertFalse(env.step(action)[2])
            self.assertEqual(env.step(action), final)

    def test_incomplete_episode_never_updates_table(self):
        table = MeanTable()
        with self.assertRaises(IncompleteEpisodeError):
            item = sample_episode(RandomWalk(), lambda _: (0.0, 1.0), random.Random(0), max_steps=1)
            update_episode(table, item, 1)
        with self.assertRaises(IncompleteEpisodeError):
            update_episode(table, episode([(3, 1, 0)], terminated=False), 1)
        self.assertEqual(table.values, {})
        self.assertEqual(table.counts, {})
        # 恰好在上限步数自然终止，应当接纳。
        item = sample_episode(RandomWalk(), lambda _: (0.0, 1.0), random.Random(0), max_steps=3)
        self.assertTrue(item.terminated)
        update_episode(table, item, 1)
        self.assertEqual(table.value(3), 1)

    def test_importance_zero_weight_and_incremental_means(self):
        estimator = ImportanceMean()
        estimator.update(3, 0)
        self.assertEqual(estimator.count, 1)
        self.assertEqual(estimator.ordinary, 0)
        self.assertIsNone(estimator.weighted)
        estimator.update(5, 2)
        self.assertEqual(estimator.weighted, 5)
        estimator.update(7, 4)
        self.assertEqual(estimator.weight_sum, 6)
        self.assertAlmostEqual(estimator.ordinary, 38 / 3)
        self.assertAlmostEqual(estimator.weighted, 19 / 3)

    def test_v_q_and_empty_product_weights(self):
        item = episode([(0, 1, -1), (1, 0, 8)])
        self.assertEqual(trajectory_weight(item, target_treasure_policy), 4)
        self.assertEqual(trajectory_weight(item, target_treasure_policy, start=1), 2)
        self.assertEqual(trajectory_weight(item, target_treasure_policy, start=2), 1)
        self.assertEqual(trajectory_weight(episode([(0, 0, 3)]), target_treasure_policy), 0)
        with self.assertRaises(ValueError):
            trajectory_weight(episode([(0, 1, -1)], probability=0), target_treasure_policy)

    def test_importance_identity_by_enumerating_all_paths(self):
        # 均匀行为策略下，三种完整路径概率分别为 1/2、1/4、1/4。
        paths = [(0.5, [(0, 0, 3)]),
                 (0.25, [(0, 1, -1), (1, 0, 8)]),
                 (0.25, [(0, 1, -1), (1, 1, -4)])]
        expected_weighted_return = 0.0
        expected_weight = 0.0
        for probability, items in paths:
            weight = trajectory_weight(episode(items), target_treasure_policy)
            total = discounted_returns([r for _, _, r in items], 0.9)[0]
            expected_weight += probability * weight
            expected_weighted_return += probability * weight * total
        self.assertEqual(expected_weight, 1)
        self.assertAlmostEqual(expected_weighted_return, 6.2)

    def test_evaluation_is_greedy_and_does_not_train(self):
        table = MeanTable(values={(0, 0): 3, (0, 1): 5.66, (1, 0): 8, (1, 1): -4})
        before = copy.deepcopy(table)
        result = evaluate_treasure(table, 0.9, episodes=10, seed=10000)
        self.assertEqual(result["return_mean"], 7)
        self.assertAlmostEqual(result["discounted_return_mean"], 6.2)
        self.assertEqual([step["action"] for step in result["first_episode_trace"]], [1, 0])
        self.assertEqual(table, before)

    def test_unvisited_states_are_not_reported_as_learned_zero(self):
        metrics = prediction_metrics(MeanTable())
        self.assertEqual(metrics["values"], [None] * 5)
        self.assertIsNone(metrics["rmse"])

    def test_argument_validation_preserves_analytic_comparison(self):
        invalid = [
            ["--episodes", "0"], ["--gamma", "nan"], ["--gamma", "0.9"],
            ["--epsilon", "0.1"], ["--task", "control", "--visit", "every"],
            ["--task", "control", "--epsilon", "2"], ["--log-every", "0"],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv)


if __name__ == "__main__":
    unittest.main()
