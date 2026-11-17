"""Public counterexamples for the 0.2.0 behavior fixes (CHANGELOG.md).

Each test names one observable contract a caller relies on:

- A rejected tied snap onto a bound re-evaluates the kept point, so the
  live evaluator graph is at the accepted x when ``accepted_callback`` runs.
- Nested diagnostic arrays, mappings, lists and tuples in an
  evaluation are owned snapshots (opaque objects keep their identity).
- Stationarity uses the unit projection onto the original
  bounds; the inner ``gtol`` is capped at ``tol / sqrt(free coordinates)``.
- Multipliers above ``multiplier_max`` raise
  ``ValueError`` before any evaluation, callback or restore.
- An overflowing penalty request stops with ``penalty_overflow`` and a
  finite penalty, also through the checkpoint plugin.
- A restore after the inner budget ran out reports
  ``inner_budget_restored_best_feasible``.
- Skipped inner work reports no attempt and no optimizer success.
- An L-BFGS-B stop that only the trust box's projection satisfied
  is not reported as inner success.
"""

import dataclasses
import unittest
from collections.abc import Mapping

import numpy as np

from simsopt_alm import ALMSettings, augmented_inequality_objective, minimize_alm
from simsopt_alm.checkpoint import alm_checkpointing, resume_boundary
from simsopt_alm.history import ALMHistoryRecorder


def quadratic(x, multipliers, penalty):
    return augmented_inequality_objective(
        base_value=float(0.5 * (x @ x)),
        base_grad=x.copy(),
        constraint_values=np.array([]),
        constraint_grads=[],
        multipliers=multipliers,
        penalty=penalty,
    )


class ALMBehaviorFixTests(unittest.TestCase):
    def test_rejected_tied_snap_restores_the_graph_before_accepted_callback(self):
        initial = np.array([np.nextafter(1.0, 0.0)])
        live = initial.copy()
        accepted = []

        def evaluate(x, multipliers, penalty):
            live[:] = x
            evaluation = augmented_inequality_objective(
                base_value=1e20 - float(x[0]),
                base_grad=np.array([-1.0]),
                constraint_values=np.array([]),
                constraint_grads=[],
                multipliers=multipliers,
                penalty=penalty,
            )
            # Keep this regression on the inner path after the near-bound
            # projection repair; the application owns this distinct norm.
            evaluation["stationarity_norm"] = 2.0
            return evaluation

        def capture(x):
            np.testing.assert_array_equal(
                live, x, err_msg="accepted x and the live evaluator graph differ"
            )
            accepted.append(x.copy())

        minimize_alm(
            initial, [], evaluate,
            ALMSettings(max_outer_iterations=1, max_subproblem_continuations=0),
            {"maxiter": 1}, base_bounds=[(0.0, 1.0)], accepted_callback=capture,
        )
        self.assertTrue(accepted, "the tied-snap callback path was not exercised")
        np.testing.assert_array_equal(accepted[0], initial)

    def test_incumbent_owns_reused_nested_diagnostic_arrays_and_lists(self):
        nested_array = np.array([1.0])
        diagnostics = {"rows": [{"point": nested_array}], "samples": [1.0]}
        opaque = object()

        def evaluate(x, multipliers, penalty):
            value = float(x[0])
            nested_array[:] = value
            diagnostics["samples"][0] = value
            evaluation = augmented_inequality_objective(
                base_value=(value - 2.0) ** 2,
                base_grad=np.array([2.0 * (value - 2.0)]),
                constraint_values=np.array([value - 1.0]),
                constraint_grads=[np.array([1.0])],
                multipliers=multipliers, penalty=penalty,
            )
            evaluation["diagnostics"] = diagnostics
            evaluation["opaque"] = opaque
            return evaluation

        result = minimize_alm(
            np.array([1.0]), ["x<=1"], evaluate,
            ALMSettings(max_outer_iterations=10), {"maxiter": 1},
        )
        self.assertTrue(result.restored_best_feasible)
        np.testing.assert_array_equal(result.x, [1.0])
        self.assertEqual(result.objective, 1.0)
        kept = result.evaluation["diagnostics"]
        self.assertIsInstance(kept, Mapping)
        np.testing.assert_array_equal(kept["rows"][0]["point"], result.x)
        self.assertEqual(tuple(kept["samples"]), (1.0,))
        self.assertIs(result.evaluation["opaque"], opaque)
        nested_array[:] = 42.0
        diagnostics["samples"][0] = 42.0
        np.testing.assert_array_equal(kept["rows"][0]["point"], result.x)
        self.assertEqual(tuple(kept["samples"]), (1.0,))

    def test_256d_legacy_l2_target_requires_progress_to_the_known_minimizer(self):
        initial = np.full(256, 9e-8)
        self.assertGreater(np.linalg.norm(initial), 1e-6)
        self.assertLess(np.max(np.abs(initial)), 1e-7)
        result = minimize_alm(
            initial, [], quadratic,
            ALMSettings(penalty_init=1e6, stationarity_tol=1e-6), {"maxiter": 30},
        )
        self.assertTrue(result.success, result.message)
        self.assertGreater(result.nit, 0, "the initial L2 residual needs optimizer work")
        self.assertLessEqual(np.linalg.norm(result.x), 1e-6)
        self.assertLess(np.linalg.norm(result.x), np.linalg.norm(initial))

    def test_near_bound_stationarity_uses_the_unit_projection_distance(self):
        def evaluate(x, multipliers, penalty):
            return augmented_inequality_objective(
                base_value=-float(x[0]), base_grad=np.array([-1.0]),
                constraint_values=np.array([0.0]), constraint_grads=[np.array([0.0])],
                multipliers=multipliers, penalty=penalty,
            )

        initial = np.array([1.0 - 5e-7])
        result = minimize_alm(
            initial, ["active"], evaluate, ALMSettings(stationarity_tol=1e-6),
            {"maxiter": 0}, base_bounds=[(0.0, 1.0)],
        )
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.stationarity_norm, 1.0 - initial[0])
        np.testing.assert_array_equal(result.x, initial)

    def test_fixed_coordinates_do_not_count_in_the_inner_norm_bridge(self):
        initial = np.full(256, 0.5)
        initial[-1] = 9e-8
        bounds = [(0.5, 0.5)] * 255 + [(-1.0, 1.0)]
        result = minimize_alm(
            initial, [], quadratic,
            ALMSettings(penalty_init=1e6, stationarity_tol=1e-6), {"maxiter": 1},
            base_bounds=bounds,
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.nit, 0)
        self.assertAlmostEqual(result.stationarity_norm, 9e-8)

    def test_artificial_box_stationarity_does_not_replace_original_bounds(self):
        recorder = ALMHistoryRecorder(None)
        result = minimize_alm(
            np.array([1.0]), [], quadratic,
            ALMSettings(trust_radius_init=1e-3, max_inner_attempts=1,
                        max_outer_iterations=1, max_subproblem_continuations=0),
            {"maxiter": 1}, on_outer_step=recorder.record,
        )
        self.assertFalse(result.success)
        self.assertGreater(result.stationarity_norm, 0.9)
        self.assertEqual(recorder.history()[0]["inner_lbfgsb_projected_gradient_norm"], 0.0)

    def test_genuine_kkt_initial_multiplier_above_cap_is_rejected_before_callbacks(self):
        calls = []

        def evaluate(x, multipliers, penalty):
            calls.append("evaluate")
            return augmented_inequality_objective(
                base_value=-2.0 * float(x[0]), base_grad=np.array([-2.0]),
                constraint_values=x.copy(), constraint_grads=[np.array([1.0])],
                multipliers=multipliers, penalty=penalty,
            )

        with self.assertRaisesRegex(ValueError, "initial.*multiplier.*multiplier_max"):
            minimize_alm(
                np.array([0.0]), ["x<=0"], evaluate, ALMSettings(multiplier_max=1.0),
                {"maxiter": 2}, initial_multipliers=np.array([2.0]),
                outer_state_callback=lambda *_args: calls.append("outer"),
                accepted_callback=lambda _x: calls.append("accepted"),
            )
        self.assertEqual(calls, [], "invalid user state reached the evaluator or callbacks")

    def test_reduced_resume_cap_is_rejected_before_restoring_caller_state(self):
        boundaries = []
        calls = []

        def evaluate(x, multipliers, penalty):
            calls.append("evaluate")
            return augmented_inequality_objective(
                base_value=float(x[0]), base_grad=np.array([1.0]),
                constraint_values=np.array([0.0]), constraint_grads=[np.array([0.0])],
                multipliers=multipliers, penalty=penalty,
            )

        minimize_alm(
            np.array([0.0]), ["active"], evaluate,
            ALMSettings(multiplier_max=3.0, max_outer_iterations=3,
                        max_subproblem_continuations=0),
            {"maxiter": 0}, initial_multipliers=np.array([2.0]),
            on_outer_boundary=boundaries.append,
            snapshot_accepted_state_fn=lambda: "accepted geometry",
            restore_incumbent_state_fn=lambda _state: calls.append("restore"),
        )
        boundary = next(b for b in boundaries if b.termination_reason is None)
        calls.clear()
        with self.assertRaisesRegex(ValueError, "resume.*multiplier.*multiplier_max"):
            minimize_alm(
                boundary.state.x, ["active"], evaluate,
                ALMSettings(multiplier_max=1.0, max_outer_iterations=3),
                {"maxiter": 0}, resume_from=boundary,
                snapshot_accepted_state_fn=lambda: "accepted geometry",
                restore_incumbent_state_fn=lambda _state: calls.append("restore"),
                outer_state_callback=lambda *_args: calls.append("outer"),
            )
        self.assertEqual(calls, [], "changed-cap resume restored or evaluated invalid state")

    def test_overflow_stop_is_finite_through_public_checkpoint_observer(self):
        settings = ALMSettings(penalty_init=1e200, penalty_scale=1e200,
                               penalty_max=None, max_subproblem_continuations=0)
        snapshots = []
        options = {"maxiter": 0}
        plugin = alm_checkpointing(options, completed_outer_callback=snapshots.append)
        plain = minimize_alm(np.array([1.0]), [], quadratic, settings, options)
        self.assertEqual(plain.termination_reason, "penalty_overflow")
        observed = minimize_alm(
            np.array([1.0]), [], quadratic, settings, options,
            on_outer_boundary=plugin.on_outer_boundary,
        )
        self.assertEqual(observed.termination_reason, plain.termination_reason)
        self.assertFalse(observed.success)
        self.assertTrue(snapshots)
        stopped = snapshots[-1]
        self.assertEqual(stopped.termination_reason, observed.termination_reason)
        self.assertEqual(stopped.completed_action, "penalty_overflow")
        self.assertFalse(stopped.resume_eligible)
        self.assertFalse(stopped.penalty_cap_reached)
        self.assertIsNone(stopped.penalty_cap_requested)
        self.assertEqual(stopped.penalty, 1e200)
        self.assertTrue(np.isfinite(stopped.penalty))
        restored = resume_boundary(stopped)
        self.assertEqual(restored.termination_reason, "penalty_overflow")

    def test_resume_rejects_over_cap_incumbent_multipliers_before_restoring(self):
        # The live multipliers are within the new cap; only the best-feasible
        # incumbent's exceed it.
        boundaries = []
        calls = []

        def evaluate(x, multipliers, penalty):
            return augmented_inequality_objective(
                base_value=float(x[0]), base_grad=np.array([1.0]),
                constraint_values=np.array([0.0]), constraint_grads=[np.array([0.0])],
                multipliers=multipliers, penalty=penalty,
            )

        minimize_alm(
            np.array([0.0]), ["active"], evaluate,
            ALMSettings(multiplier_max=3.0, max_outer_iterations=3,
                        max_subproblem_continuations=0),
            {"maxiter": 0}, initial_multipliers=np.array([2.0]),
            on_outer_boundary=boundaries.append,
            snapshot_accepted_state_fn=lambda: "accepted geometry",
            restore_incumbent_state_fn=lambda _state: calls.append("restore"),
        )
        boundary = next(
            b for b in boundaries
            if b.termination_reason is None and b.state.best_feasible is not None
        )
        best = dataclasses.replace(boundary.state.best_feasible, multipliers=np.array([2.0]))
        state = dataclasses.replace(
            boundary.state, multipliers=np.array([0.5]), best_feasible=best
        )
        boundary = dataclasses.replace(boundary, state=state)
        calls.clear()
        with self.assertRaisesRegex(ValueError, "resume best-feasible ALM multipliers.*multiplier_max"):
            minimize_alm(
                boundary.state.x, ["active"], evaluate,
                ALMSettings(multiplier_max=1.0, max_outer_iterations=3),
                {"maxiter": 0}, resume_from=boundary,
                snapshot_accepted_state_fn=lambda: "accepted geometry",
                restore_incumbent_state_fn=lambda _state: calls.append("restore"),
            )
        self.assertEqual(calls, [], "an over-cap incumbent was restored before the check")

    def test_overflow_under_a_finite_cap_keeps_the_penalty(self):
        # 1e200 * 1e200 overflows below no finite cap: penalty_overflow, not
        # penalty_cap_reached at penalty_max.
        settings = ALMSettings(penalty_init=1e200, penalty_scale=1e200, penalty_max=1e300,
                               max_subproblem_continuations=0)
        result = minimize_alm(np.array([1.0]), [], quadratic, settings, {"maxiter": 0})
        self.assertEqual(result.termination_reason, "penalty_overflow")
        self.assertEqual(result.penalty, 1e200)

    def test_overflow_after_an_infeasible_step_restores_the_feasible_start(self):
        # min -x on [0, 1]; the hard row x <= 0 holds only at the start, the
        # smooth row x <= 100 never binds. The inner solve moves to the
        # hard-infeasible x = 1, and the penalty raise that follows overflows.
        def evaluate(x, multipliers, penalty):
            evaluation = augmented_inequality_objective(
                base_value=-float(x[0]), base_grad=np.array([-1.0]),
                constraint_values=np.array([x[0] - 100.0]), constraint_grads=[np.array([1.0])],
                multipliers=multipliers, penalty=penalty,
            )
            hard = np.array([x[0]])
            evaluation.update(
                hard_signed_constraint_values=hard,
                hard_violation_values=np.maximum(hard, 0.0),
                surrogate_signed_constraint_values=np.array([x[0] - 100.0]),
                hard_dual_update_values=hard,
                feasibility_values=np.maximum(hard, 0.0),
                dual_update_values=hard,
            )
            return evaluation

        result = minimize_alm(
            np.array([0.0]), ["x<=0"], evaluate,
            ALMSettings(penalty_init=1e200, penalty_scale=1e200, penalty_max=None,
                        max_subproblem_continuations=0),
            {"maxiter": 20}, base_bounds=[(0.0, 1.0)],
        )
        self.assertEqual(result.termination_reason, "penalty_overflow_restored_best_feasible")
        self.assertTrue(result.restored_best_feasible)
        np.testing.assert_array_equal(result.x, [0.0])
        np.testing.assert_array_equal(result.last_iterate.x, [1.0])
        self.assertEqual(result.penalty, 1e200)

    def test_restored_result_identifies_the_spent_inner_budget(self):
        def evaluate(x, multipliers, penalty):
            value = float(x[0])
            return augmented_inequality_objective(
                base_value=(value - 2.0) ** 2,
                base_grad=np.array([2.0 * (value - 2.0)]),
                constraint_values=np.array([value - 1.0]),
                constraint_grads=[np.array([1.0])],
                multipliers=multipliers, penalty=penalty,
            )

        result = minimize_alm(np.array([1.0]), ["x<=1"], evaluate,
                              ALMSettings(max_outer_iterations=10), {"maxiter": 1})
        self.assertTrue(result.restored_best_feasible)
        self.assertEqual(result.nit, 1)
        self.assertLess(result.outer_iterations, 10)
        self.assertEqual(result.termination_reason, "inner_budget_restored_best_feasible")

    def test_zero_budget_history_reports_skipped_work_without_optimizer_success(self):
        recorder = ALMHistoryRecorder(None)
        events = []

        def collect(event):
            events.append(event)
            recorder.record(event)

        result = minimize_alm(np.array([1.0]), [], quadratic,
                              ALMSettings(max_outer_iterations=1), {"maxiter": 0},
                              on_outer_step=collect)
        self.assertFalse(result.success)
        self.assertEqual(result.nit, 0)
        self.assertTrue(events)
        for event in events:
            self.assertEqual(event.inner.iterations, 0)
            self.assertEqual(event.inner.attempts, 0)
            self.assertFalse(event.inner.optimizer_success)
            self.assertEqual(event.inner.inner_profile, "inner_skipped")
        for entry in recorder.history():
            self.assertFalse(entry["inner_success"])
            self.assertEqual(entry["inner_profile"], "inner_skipped")

    def test_numerical_preinner_success_reports_inner_skipped(self):
        recorder = ALMHistoryRecorder(None)

        def evaluate(x, multipliers, penalty):
            return augmented_inequality_objective(
                base_value=float(x @ x), base_grad=2.0 * x,
                constraint_values=x.copy(), constraint_grads=[np.array([1.0])],
                multipliers=multipliers, penalty=penalty,
            )

        result = minimize_alm(np.array([0.0]), ["x<=0"], evaluate,
                              ALMSettings(), {"maxiter": 0},
                              on_outer_step=recorder.record)
        self.assertTrue(result.success)
        self.assertEqual(result.nit, 0)
        history = recorder.history()
        self.assertEqual(len(history), 1)
        self.assertFalse(history[0]["inner_success"])
        self.assertEqual(history[0]["inner_profile"], "inner_skipped")


def far_target(x, multipliers, penalty):
    """min ||x - 10||^2 s.t. x0 + x1 <= 100 (inactive): from x = 0 a box of
    radius 0.25 binds, and its minimizer (0.25, 0.25) has raw gradient -19.5."""
    offset = x - 10.0
    return augmented_inequality_objective(
        float(offset @ offset), 2.0 * offset, [x[0] + x[1] - 100.0], [np.ones(2)],
        multipliers, penalty,
    )


class ALMTrustBoxLimitedStopTests(unittest.TestCase):
    """An L-BFGS-B PGTOL stop at a binding trust box is trust-box-limited, not
    inner success: the projected gradient drops the components that push
    through the box's faces, while the raw gradient there stays large.
    Reporting only."""

    def _first_step(self, trust_radius):
        recorder = ALMHistoryRecorder(None)
        events = []

        def collect(event):
            events.append(event)
            recorder.record(event)

        minimize_alm(
            np.zeros(2), ["sum_cap"], far_target,
            ALMSettings(trust_radius_init=trust_radius, max_outer_iterations=1),
            {"maxiter": 200, "gtol": 1.0e-5}, on_outer_step=collect,
        )
        return events[0], recorder.history()[0]

    def test_binding_box_stop_is_trust_box_limited(self):
        event, entry = self._first_step(0.25)
        inner = event.inner
        np.testing.assert_array_equal(inner.x, [0.25, 0.25])
        # Older SciPy (1.10) spells the message with underscores.
        self.assertIn("PROJECTED GRADIENT", inner.optimizer_message.upper().replace("_", " "))
        self.assertGreater(entry["raw_stationarity_norm"], 1.0)
        # L-BFGS-B's own flag is kept; only the reported success excludes the box.
        self.assertTrue(inner.optimizer_success)
        self.assertEqual(inner.inner_stop, "trust_box_limited")
        self.assertFalse(inner.succeeded)
        self.assertFalse(entry["inner_success"])
        self.assertIs(entry.get("inner_trust_box_limited"), True)

    def test_non_binding_box_stop_is_raw_stationarity(self):
        event, entry = self._first_step(1.0e5)
        self.assertEqual(event.inner.inner_stop, "raw_stationarity")
        self.assertTrue(entry["inner_success"])
        self.assertNotIn("inner_trust_box_limited", entry)


if __name__ == "__main__":
    unittest.main()
