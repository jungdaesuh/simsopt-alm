"""Feasible accepted inner iterates survive an infeasible subproblem endpoint."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from simsopt_alm import ALMSettings, augmented_inequality_objective, minimize_alm
from simsopt_alm.checkpoint import resume_boundary, transition_snapshot


class AcceptedInnerIncumbentTests(unittest.TestCase):
    def test_feasible_inner_accept_survives_infeasible_endpoint(self):
        accepted = []
        boundaries = []

        def evaluate(x, multipliers, penalty):
            return augmented_inequality_objective(
                (x[0] - 1.0) ** 2, [2.0 * (x[0] - 1.0)],
                [x[0]], [[1.0]], multipliers, penalty,
            )

        result = minimize_alm(
            [-2.0], ["upper"], evaluate,
            ALMSettings(max_outer_iterations=1, max_subproblem_continuations=0),
            {"maxiter": 100}, inner_callback=lambda x: accepted.append(x.copy()),
            on_outer_boundary=boundaries.append, constraint_blocks=["hardware"],
        )
        self.assertEqual(result.evaluation["constraint_names"], ("upper",))
        self.assertEqual(result.evaluation["constraint_blocks"], ("hardware",))
        self.assertTrue(any(np.array_equal(x, [-1.0]) for x in accepted))
        np.testing.assert_allclose(result.x, [-1.0])
        self.assertEqual(result.objective, 4.0)
        self.assertTrue(result.restored_best_feasible)
        self.assertGreater(result.last_iterate.x[0], 0.0)
        np.testing.assert_array_equal(boundaries[-1].state.best_feasible.x, result.x)
        self.assertEqual(boundaries[-1].state.best_feasible.evaluation["base_value"], 4.0)
        self.assertIsNone(result.inner_result, "an intermediate accept has no endpoint optimizer result")

    def test_only_usable_accepted_points_can_be_retained(self):
        for mode in ("unaccepted", "rejected", "nonfinite", "hard_infeasible"):
            with self.subTest(mode=mode):
                def evaluate(x, multipliers, penalty):
                    value = float(x[0])
                    evaluation = augmented_inequality_objective(
                        (value - 1.0) ** 2, [2.0 * (value - 1.0)],
                        [value], [[1.0]], multipliers, penalty,
                    )
                    if value == -1.0:
                        if mode == "rejected":
                            evaluation["search_step_success"] = False
                        elif mode == "nonfinite":
                            evaluation["nonfinite_evaluation"] = True
                        elif mode == "hard_infeasible":
                            evaluation.update(
                                hard_signed_constraint_values=np.array([1.0]),
                                hard_violation_values=np.array([1.0]),
                                surrogate_signed_constraint_values=np.array([-1.0]),
                                hard_dual_update_values=np.array([1.0]),
                            )
                    return evaluation

                def inner(fun, x, callback, **kwargs):
                    fun(np.array([-1.0]))
                    if mode != "unaccepted":
                        callback(np.array([-1.0]))
                    fun(np.array([0.5]))
                    callback(np.array([0.5]))
                    return SimpleNamespace(x=np.array([0.5]), nit=2, success=True)

                with patch("simsopt_alm._src.inner.attempts.minimize", inner):
                    result = minimize_alm(
                        [-2.0], ["upper"], evaluate,
                        ALMSettings(max_outer_iterations=1, max_subproblem_continuations=0),
                        {"maxiter": 100},
                    )
                np.testing.assert_array_equal(result.x, [-2.0])

    def test_retention_ranks_base_objective_and_reuses_inner_evaluation(self):
        calls = []

        def evaluate(x, multipliers, penalty):
            calls.append(float(x[0]))
            evaluation = augmented_inequality_objective(
                (x[0] - 1.0) ** 2, [2.0 * (x[0] - 1.0)],
                [x[0]], [[1.0]], multipliers, penalty,
            )
            # A lower augmented total alone cannot replace the base-objective
            # incumbent. The second accepted feasible point has worse physics.
            if x[0] == -1.5:
                evaluation["total"] = -100.0
            return evaluation

        def inner(fun, x, callback, **kwargs):
            for point in (-1.0, -1.5, 0.5):
                fun(np.array([point]))
                callback(np.array([point]))
            return SimpleNamespace(x=np.array([0.5]), nit=3, success=True)

        with patch("simsopt_alm._src.inner.attempts.minimize", inner):
            result = minimize_alm(
                [-2.0], ["upper"], evaluate,
                ALMSettings(max_outer_iterations=1, max_subproblem_continuations=0),
                {"maxiter": 100},
            )
        np.testing.assert_array_equal(result.x, [-1.0])
        self.assertEqual(calls.count(-1.0), 1, "retention must reuse the accepted evaluation")
        self.assertEqual(calls.count(-1.5), 1)

    def test_stateful_inner_snapshot_restores_the_matching_accepted_point(self):
        for inner_snapshots in (False, True):
            with self.subTest(inner_snapshots=inner_snapshots):
                accepted = [np.array([-2.0])]
                iterate = [np.array([-2.0])]
                promoted = []
                snapshots = []
                boundaries = []

                def evaluate(x, multipliers, penalty):
                    return augmented_inequality_objective(
                        (x[0] - 1.0) ** 2, [2.0 * (x[0] - 1.0)],
                        [x[0]], [[1.0]], multipliers, penalty,
                    )

                def accept_outer(x):
                    promoted.append(x.copy())
                    accepted[0] = x.copy()

                def snapshot_inner(x):
                    np.testing.assert_array_equal(x, iterate[0])
                    snapshots.append(x.copy())
                    return iterate[0].copy()

                def restore(state):
                    accepted[0] = state.copy()
                    iterate[0] = state.copy()

                kwargs = {}
                if inner_snapshots:
                    kwargs["snapshot_inner_state_fn"] = snapshot_inner
                result = minimize_alm(
                    [-2.0], ["upper"], evaluate,
                    ALMSettings(max_outer_iterations=1, max_subproblem_continuations=0),
                    {"maxiter": 100},
                    inner_callback=lambda x: iterate.__setitem__(0, x.copy()),
                    accepted_callback=accept_outer,
                    snapshot_accepted_state_fn=lambda: accepted[0].copy(),
                    restore_incumbent_state_fn=restore,
                    on_outer_boundary=boundaries.append,
                    **kwargs,
                )
                expected = [-1.0] if inner_snapshots else [-2.0]
                np.testing.assert_array_equal(result.x, expected)
                np.testing.assert_array_equal(accepted[0], result.x)
                np.testing.assert_array_equal(boundaries[-1].accepted_state, result.x)
                np.testing.assert_array_equal(
                    boundaries[-1].state.best_feasible.incumbent_state, result.x
                )
                self.assertEqual(len(promoted), 1, "outer callbacks remain endpoint-only")
                self.assertEqual(len(snapshots), int(inner_snapshots))
                # Public state owns its arrays independently of later caller changes.
                accepted[0][:] = 123.0
                np.testing.assert_array_equal(result.x, expected)
                np.testing.assert_array_equal(boundaries[-1].accepted_state, expected)

    def test_inner_snapshot_requires_a_restore_before_evaluation(self):
        def evaluate(*args):
            self.fail("invalid callback pairing must fail before evaluation")

        with self.assertRaisesRegex(ValueError, "snapshot_inner_state_fn requires"):
            minimize_alm(
                [-2.0], ["upper"], evaluate, ALMSettings(), {"maxiter": 100},
                snapshot_inner_state_fn=lambda x: x.copy(),
            )

    def test_retained_point_that_becomes_endpoint_owns_optimizer_result(self):
        endpoint_results = []

        def evaluate(x, multipliers, penalty):
            return augmented_inequality_objective(
                (x[0] - 1.0) ** 2, [2.0 * (x[0] - 1.0)],
                [x[0]], [[1.0]], multipliers, penalty,
            )

        def inner(fun, x, callback, **kwargs):
            point = np.array([-1.0 if not endpoint_results else 0.5])
            fun(point)
            callback(point)
            endpoint = SimpleNamespace(x=point.copy(), nit=1, success=True)
            endpoint_results.append(endpoint)
            return endpoint

        with patch("simsopt_alm._src.inner.attempts.minimize", inner):
            result = minimize_alm(
                [-2.0], ["upper"], evaluate,
                ALMSettings(max_outer_iterations=2, max_subproblem_continuations=0),
                {"maxiter": 100},
            )
        self.assertEqual(len(endpoint_results), 2)
        np.testing.assert_array_equal(result.x, [-1.0])
        self.assertIs(result.inner_result, endpoint_results[0])

    def test_inner_snapshot_can_veto_a_candidate(self):
        live = [np.array([-2.0])]
        snapshots = []

        def evaluate(x, multipliers, penalty):
            return augmented_inequality_objective(
                (x[0] - 1.0) ** 2, [2.0 * (x[0] - 1.0)],
                [x[0]], [[1.0]], multipliers, penalty,
            )

        def veto(x):
            snapshots.append(x.copy())
            return None

        result = minimize_alm(
            [-2.0], ["upper"], evaluate,
            ALMSettings(max_outer_iterations=1, max_subproblem_continuations=0),
            {"maxiter": 100}, snapshot_inner_state_fn=veto,
            snapshot_accepted_state_fn=lambda: live[0].copy(),
            accepted_callback=lambda x: live.__setitem__(0, x.copy()),
            restore_incumbent_state_fn=lambda x: live.__setitem__(0, x.copy()),
        )
        self.assertEqual(len(snapshots), 1)
        np.testing.assert_array_equal(snapshots[0], [-1.0])
        np.testing.assert_array_equal(result.x, [-2.0])
        np.testing.assert_array_equal(live[0], result.x)

    def test_checkpoint_resume_restores_the_inner_incumbent_state(self):
        live = [np.array([-2.0])]
        iterate = [np.array([-2.0])]
        boundaries = []

        def evaluate(x, multipliers, penalty):
            return augmented_inequality_objective(
                (x[0] - 1.0) ** 2, [2.0 * (x[0] - 1.0)],
                [x[0]], [[1.0]], multipliers, penalty,
            )

        def inner(fun, x, callback, **kwargs):
            for point in (-1.0, 0.5):
                fun(np.array([point]))
                callback(np.array([point]))
            return SimpleNamespace(x=np.array([0.5]), nit=2, success=True)

        def restore(state):
            live[0] = np.array(state, dtype=float)
            iterate[0] = np.array(state, dtype=float)

        state_callbacks = dict(
            inner_callback=lambda x: iterate.__setitem__(0, x.copy()),
            accepted_callback=lambda x: live.__setitem__(0, x.copy()),
            snapshot_accepted_state_fn=lambda: live[0].copy(),
            snapshot_inner_state_fn=lambda x: iterate[0].copy(),
            restore_incumbent_state_fn=restore,
        )
        settings = ALMSettings(max_outer_iterations=2, max_subproblem_continuations=0)
        with patch("simsopt_alm._src.inner.attempts.minimize", inner):
            minimize_alm(
                [-2.0], ["upper"], evaluate, settings, {"maxiter": 100},
                on_outer_boundary=boundaries.append, **state_callbacks,
            )
        saved = transition_snapshot(boundaries[0], {"maxiter": 100})
        restored = resume_boundary(saved)
        np.testing.assert_array_equal(restored.state.best_feasible.x, [-1.0])
        result = minimize_alm(
            restored.state.x, ["upper"], evaluate, settings, {"maxiter": 0},
            resume_from=restored, **state_callbacks,
        )
        np.testing.assert_array_equal(result.x, [-1.0])
        np.testing.assert_array_equal(live[0], result.x)
        self.assertEqual(result.objective, 4.0)


if __name__ == "__main__":
    unittest.main()
