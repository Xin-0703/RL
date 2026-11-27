"""验证推导、曲率、KL 回溯和失败恢复；无需第三方测试库。"""

import math
import unittest

from trpo_demo import (Problem, conjugate_gradient, fisher_vector_product, gradient,
                       mean_kl, objective, sigmoid, surrogate, trpo_step)


class TRPOTests(unittest.TestCase):
    def setUp(self):
        self.single = Problem((1.0,), ((0.0, 2.0),))

    def test_example_one(self):
        old, candidate = [0.0], [math.log(0.6 / 0.4)]
        self.assertAlmostEqual(surrogate(self.single, old, candidate), 0.2)
        self.assertAlmostEqual(mean_kl(self.single, old, candidate), 0.0204109972601276)
        self.assertAlmostEqual(objective(self.single, candidate), 1.2)

    def test_example_two(self):
        old = [0.0]
        new, info = trpo_step(self.single, old)
        self.assertEqual(old, [0.0])
        self.assertTrue(info["accepted"])
        self.assertEqual(info["fraction"], 1.0)
        self.assertAlmostEqual(new[0], math.sqrt(0.08))
        self.assertAlmostEqual(sigmoid(new[0]), 0.570243014550411)
        self.assertAlmostEqual(info["kl"], 0.009966843372030678)

    def test_gradient_and_fisher_finite_differences(self):
        problem = Problem((0.3, 0.7), ((-1.0, 3.0), (2.0, -2.0)))
        old, v, h = [-0.7, 1.1], [0.4, -0.9], 1e-4
        plus = [t + h * d for t, d in zip(old, v)]
        minus = [t - h * d for t, d in zip(old, v)]
        g = gradient(problem, old)
        derivative = (surrogate(problem, old, plus) - surrogate(problem, old, minus)) / (2 * h)
        self.assertAlmostEqual(derivative, sum(a * b for a, b in zip(g, v)), places=8)
        fv = fisher_vector_product(problem, old, v)
        second = (mean_kl(problem, old, plus) + mean_kl(problem, old, minus)) / h**2
        self.assertAlmostEqual(second, sum(a * b for a, b in zip(v, fv)), places=6)
        for candidate in (plus, minus, [1.4, -0.3]):
            self.assertAlmostEqual(surrogate(problem, old, candidate),
                                   objective(problem, candidate) - objective(problem, old))

    def test_cg_nondiagonal_system(self):
        # 独立于示例的对角 Fisher，检查 CG 能解耦合正定系统。
        def matvec(v):
            return [4 * v[0] + v[1], v[0] + 3 * v[1]]
        x = conjugate_gradient(matvec, [1.0, 2.0])
        self.assertAlmostEqual(x[0], 1 / 11)
        self.assertAlmostEqual(x[1], 7 / 11)
        with self.assertRaises(ArithmeticError):
            conjugate_gradient(lambda v: [-v[0]], [1.0])

    def test_contextual_example(self):
        problem = Problem((0.5, 0.5), ((0.0, 2.0), (2.0, 0.0)))
        new, info = trpo_step(problem, [0.0, 0.0])
        self.assertAlmostEqual(new[0], math.sqrt(0.08))
        self.assertAlmostEqual(new[1], -math.sqrt(0.08))
        self.assertAlmostEqual(objective(problem, new), 1.140486029100822)
        self.assertLessEqual(info["kl"], 0.01)

    def test_backtrack_and_restore(self):
        old = [math.log(0.01 / 0.99)]
        new, info = trpo_step(self.single, old)
        self.assertTrue(info["accepted"])
        self.assertGreater(info["attempts"][0]["kl"], 0.01)
        self.assertEqual(info["fraction"], 0.5)
        self.assertLessEqual(info["kl"], 0.01)
        self.assertAlmostEqual(sigmoid(new[0]), 0.02014495434875215)
        restored, failed = trpo_step(self.single, old, max_backtracks=1)
        self.assertEqual(restored, old)
        self.assertEqual(failed["reason"], "line_search_failed")
        self.assertFalse(failed["accepted"])

    def test_zero_gradient_and_invalid_settings(self):
        flat = Problem((1.0,), ((2.0, 2.0),))
        theta, info = trpo_step(flat, [0.0])
        self.assertEqual(theta, [0.0])
        self.assertEqual(info["reason"], "zero_gradient")
        for kwargs in ({"delta": 0}, {"delta": math.nan}, {"damping": -1},
                       {"cg_iters": 0}, {"accept_ratio": 1}, {"backtrack": 1}):
            with self.assertRaises(ValueError):
                trpo_step(self.single, [0.0], **kwargs)

    def test_kl_direction_and_extreme_logits(self):
        self.assertEqual(mean_kl(self.single, [1000.0], [1000.0]), 0.0)
        self.assertAlmostEqual(mean_kl(self.single, [1000.0], [-1000.0]), 1000.0)
        old, candidate = [math.log(0.2 / 0.8)], [math.log(0.9 / 0.1)]
        expected = 0.8 * math.log(0.8 / 0.1) + 0.2 * math.log(0.2 / 0.9)
        self.assertAlmostEqual(mean_kl(self.single, old, candidate), expected)
        self.assertNotAlmostEqual(mean_kl(self.single, candidate, old), expected)

    def test_multiple_updates_with_damping(self):
        problem = Problem((0.3, 0.7), ((0.0, 2.0), (2.0, 0.0)))
        for damping in (0.0, 0.1):
            theta = [-1.2, 0.7]
            for _ in range(15):
                old = theta[:]
                theta, info = trpo_step(problem, old, damping=damping)
                self.assertLessEqual(mean_kl(problem, old, theta), 0.01)
                self.assertGreaterEqual(objective(problem, theta), objective(problem, old))
                if info["accepted"]:
                    self.assertGreater(info["gain"], 0)
                    self.assertGreaterEqual(info["gain"], 0.1 * info["expected_gain"])


if __name__ == "__main__":
    unittest.main()
