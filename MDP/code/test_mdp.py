"""以解析解、手算轮次与错误边界验证规划算法。"""

import contextlib
import copy
import io
import unittest

from mdp import (
    Outcome, action_values, bellman_backup, deterministic_policy, finite_horizon,
    greedy_actions, initial_policy, iterative_evaluation, linear_evaluation,
    make_model, parse_args, policy_iteration, solve_linear, uniform_policy,
    validate_model, validate_policy, value_iteration,
)


class MDPTests(unittest.TestCase):
    def test_uniform_treasure_evaluation_two_methods(self):
        model = make_model("treasure")
        policy = uniform_policy(model)
        for result in (iterative_evaluation(model, policy, 0.9),
                       linear_evaluation(model, policy, 0.9)):
            self.assertAlmostEqual(result["values"]["start"], 1.9)
            self.assertAlmostEqual(result["values"]["cave"], 2)
            self.assertEqual(result["values"]["terminal"], 0)

    def test_matrix_excludes_terminal_but_includes_arrival_reward(self):
        model = make_model("treasure")
        result = linear_evaluation(model, uniform_policy(model), 0.9)
        self.assertEqual(result["state_order"], ["start", "cave"])
        self.assertEqual(result["matrix_I_minus_gamma_P"], [[1, -0.45], [0, 1]])
        self.assertEqual(result["expected_rewards"], [1, 2])

    def test_synchronous_backups_use_old_values(self):
        model = make_model("treasure")
        old = dict.fromkeys(model, 0.0)
        first = bellman_backup(model, old, 0.9)
        second = bellman_backup(model, first, 0.9)
        self.assertEqual(old, {"start": 0, "cave": 0, "terminal": 0})
        self.assertEqual(first, {"start": 3, "cave": 8, "terminal": 0})
        self.assertAlmostEqual(second["start"], 6.2)

    def test_both_control_algorithms_match_treasure_solution(self):
        model = make_model("treasure")
        for gamma, action, value in ((0.9, "cave", 6.2), (0.3, "coin", 3)):
            for solver in (value_iteration, policy_iteration):
                result = solver(model, gamma)
                self.assertEqual(result["actions"], {"start": action, "cave": "chest"})
                self.assertAlmostEqual(result["values"]["start"], value)
                self.assertLessEqual(result["optimal_residual"], 1e-10)

    def test_stochastic_outcomes_are_averaged_not_maximized(self):
        model = make_model("stochastic")
        result = value_iteration(model, 0.9)
        self.assertAlmostEqual(result["values"]["cave"], 5)
        self.assertAlmostEqual(result["values"]["start"], 3.5)
        self.assertEqual(result["actions"]["start"], "cave")

    def test_multiple_rewards_for_same_next_state(self):
        model = {"s": {"a": [Outcome(0.25, "end", 8), Outcome(0.75, "end", 0)]}, "end": {}}
        self.assertEqual(linear_evaluation(model, uniform_policy(model), 0.9)["values"]["s"], 2)

    def test_continuing_solution_and_residual_bound(self):
        model = make_model("career")
        vi = value_iteration(model, 0.9, tolerance=1e-9)
        pi = policy_iteration(model, 0.9)
        self.assertEqual(vi["actions"], {"basic": "invest", "trained": "work"})
        for state, exact in (("basic", 18), ("trained", 20)):
            self.assertAlmostEqual(pi["values"][state], exact)
            self.assertLessEqual(abs(vi["values"][state] - exact), vi["value_error_bound"] + 1e-12)
        baseline = linear_evaluation(model, initial_policy(model), 0.9)
        self.assertAlmostEqual(baseline["values"]["basic"], 10)
        self.assertAlmostEqual(baseline["values"]["trained"], 20)

    def test_finite_horizon_policy_depends_on_remaining_time(self):
        result = finite_horizon(make_model("career"), 0.9, 3)
        layers = result["layers"]
        self.assertEqual(layers[0]["values"], {"basic": 0, "trained": 0})
        self.assertEqual([layer["actions"]["basic"] for layer in layers[1:]], ["rest", "rest", "invest"])
        for layer, value in zip(layers[1:], [1, 1.9, 3.42]):
            self.assertAlmostEqual(layer["values"]["basic"], value)
        undiscounted = finite_horizon(make_model("career"), 1, 3)
        self.assertEqual(undiscounted["layers"][3]["values"]["basic"], 4)
        self.assertEqual(len(finite_horizon(make_model("career"), 1, 0)["layers"]), 1)

    def test_policy_improvement_preserves_or_increases_value(self):
        model = make_model("treasure")
        old = linear_evaluation(model, uniform_policy(model), 0.9)["values"]
        actions = greedy_actions(model, old, 0.9)
        self.assertEqual(actions, {"start": "coin", "cave": "chest"})
        new = linear_evaluation(model, deterministic_policy(model, actions), 0.9)["values"]
        for state in model:
            self.assertGreaterEqual(new[state], old[state])
        self.assertEqual(greedy_actions(model, new, 0.9)["start"], "cave")

    def test_terminal_continuation_is_zero_and_tie_is_deterministic(self):
        model = make_model("treasure")
        values = {"start": 0, "cave": 8, "terminal": 999}
        self.assertEqual(action_values(model, values, 0.9)["cave"]["chest"], 8)
        values["terminal"] = 0
        self.assertEqual(greedy_actions(model, values, 0.5)["start"], "coin")
        result = policy_iteration(model, 0.5)
        self.assertEqual(result["actions"]["start"], "coin")

    def test_linear_solver_pivoting_and_singular_guard(self):
        self.assertEqual(solve_linear([[0, 1], [1, 1]], [2, 3]), [1, 2])
        with self.assertRaises(ValueError):
            solve_linear([[1, 1], [2, 2]], [2, 4])

    def test_invalid_models_and_policies(self):
        for outcome in (Outcome(-1, "end", 1), Outcome(0.5, "end", 1),
                        Outcome(1, "missing", 1), Outcome(1, "end", float("nan"))):
            with self.subTest(outcome=outcome), self.assertRaises(ValueError):
                validate_model({"s": {"a": [outcome]}, "end": {}})
        model = make_model("treasure")
        for policy in ({}, {"start": {"coin": 1}, "cave": {"chest": 1}},
                       {"start": {"coin": 1.1, "cave": -0.1}, "cave": {"chest": 1, "trap": 0}}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                validate_policy(model, policy)

    def test_iteration_limit_is_not_reported_as_convergence(self):
        model = make_model("career")
        with self.assertRaises(RuntimeError):
            value_iteration(model, 0.9, max_iterations=1)
        with self.assertRaises(RuntimeError):
            iterative_evaluation(model, uniform_policy(model), 0.9, max_iterations=1)
        with self.assertRaises(RuntimeError):
            policy_iteration(model, 0.9, max_iterations=1)

    def test_planning_does_not_mutate_model_or_policy(self):
        model = make_model("treasure")
        policy = uniform_policy(model)
        before = copy.deepcopy((model, policy))
        iterative_evaluation(model, policy, 0.9)
        linear_evaluation(model, policy, 0.9)
        value_iteration(model, 0.9)
        policy_iteration(model, 0.9)
        finite_horizon(model, 0.9, 3)
        self.assertEqual((model, policy), before)

    def test_cli_rejects_invalid_domains(self):
        for argv in (["--gamma", "1"], ["--gamma", "nan"], ["--gamma", "-1"],
                     ["--horizon", "-1"], ["--tolerance", "0"], ["--max-iterations", "0"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv)
        self.assertEqual(parse_args(["--algorithm", "finite", "--gamma", "1"]).gamma, 1)


if __name__ == "__main__":
    unittest.main()
