"""Regressions for finite PHR scalar arithmetic and real minimization."""

from fractions import Fraction
import unittest

import numpy as np

from simsopt_alm import (
    ALMPhysics, ALMSettings, augmented_inequality_objective, cached_alm_evaluator,
    minimize_alm,
)


def warm_physics(point):
    value = float(point[0])
    return ALMPhysics(
        -1e6 * value + 0.5 * value * value,
        np.array([-1e6 + value]), np.array([value]), (np.ones(1),),
    )


class PenaltyArithmeticTests(unittest.TestCase):
    def test_active_inactive_and_nonuniform_rows_match_exact_binary_inputs(self):
        rows = np.array([1e-4, -1e-4, -4.0, -2.0, 0.0, 2.0 ** -40])
        multipliers = np.array([1e6, 1e6, 2.0, 2.0, 1e308, 2.0 ** 19])
        penalties = np.array([1.0, 3.0, 2.0, 1.0, 1.0, 0.5])
        expected = []
        for row, multiplier, penalty in zip(rows, multipliers, penalties):
            g, mu, rho = map(Fraction, (row, multiplier, penalty))
            expected.append(float(((max(0, mu + rho * g)) ** 2 - mu ** 2) / (2 * rho)))
        with np.errstate(over="ignore", invalid="ignore"):
            result = augmented_inequality_objective(
                0.0, np.zeros(1), rows, [np.zeros(1)] * len(rows),
                multipliers, penalties,
            )
        np.testing.assert_allclose(result["augmented_term_by_constraint"], expected, rtol=4e-16, atol=0)

    def test_extreme_finite_terms_do_not_overflow_or_underflow_prematurely(self):
        cases = [
            (2.0, 0.0, np.nextafter(0.0, 1.0)),
            (2.7, 0.0, np.nextafter(0.0, 1.0)),
            (3.0, 0.0, np.nextafter(0.0, 1.0)),
            (1e-200, 0.0, 1e300),
            (1e200, 0.0, 1e-300),
            (-1e100, 1e200, 1e100),
            (-1.0, 1e-200, 1e-200),
            (-1e308, 1e-10, 1e-318),
            (0.6, 1.6e308, 1.6e308),
        ]
        for row, multiplier, penalty in cases:
            with self.subTest(row=row, multiplier=multiplier, penalty=penalty):
                g, mu, rho = map(Fraction, (row, multiplier, penalty))
                expected = float((max(0, mu + rho * g) ** 2 - mu ** 2) / (2 * rho))
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    result = augmented_inequality_objective(
                        0.0, np.zeros(1), [row], [np.zeros(1)], [multiplier], penalty,
                    )
                self.assertTrue(np.isfinite(result["total"]))
                np.testing.assert_allclose(result["total"], expected, rtol=5e-16, atol=0)

    def test_inactive_zero_multiplier_retains_positive_zero(self):
        result = augmented_inequality_objective(
            0.0, np.zeros(1), [-1.0], [np.zeros(1)], [0.0], 1.0,
        )
        term = result["augmented_term_by_constraint"][0]
        self.assertEqual(term, 0.0)
        self.assertFalse(np.signbit(term))

    def test_warm_start_total_agrees_with_independent_exact_oracle(self):
        point = np.array([1e-4])
        evaluator = cached_alm_evaluator(warm_physics)
        result = evaluator(point, np.array([1e6]), 1.0)
        physics = warm_physics(point)
        g = Fraction(float(point[0]))
        expected = float(Fraction(physics.base_value) + Fraction(1e6) * g + g * g / 2)
        self.assertAlmostEqual(result["total"], expected, delta=3e-14)

    def test_real_public_solver_minimizes_warm_start_objective(self):
        evaluator = cached_alm_evaluator(warm_physics)
        result = minimize_alm(
            np.array([1e-4]), ["row"], evaluator, ALMSettings(), {"maxiter": 1000},
            initial_multipliers=np.array([1e6]), base_bounds=((-1.0, 1.0),),
        )
        self.assertTrue(result.success, result.termination_reason)
        np.testing.assert_allclose(result.x, [0.0], rtol=0, atol=1e-7)
        self.assertLessEqual(result.stationarity_norm, 1e-6)
