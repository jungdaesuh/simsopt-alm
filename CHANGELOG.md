# Changelog

## 0.2.0

- Layout: the implementation moved into `simsopt_alm._src` (one subpackage
  per component: `settings`, `problem`, `policy`, `inner`, `loop`,
  `checkpoint`, plus the shared `state` and `alm_math` modules), behind
  public facades. The move left solver behavior unchanged (every golden
  trajectory replayed byte-identically); only the behavior fixes below
  change what a run reports.
- Public imports: everything in `simsopt_alm.__all__` imports from
  `simsopt_alm` as before, and `import simsopt_alm` still binds
  `simsopt_alm.policy`. `simsopt_alm.policy` exports what a policy author
  needs (the protocol, `DefaultContinuationPolicy` and its instance
  `DEFAULT_CONTINUATION_POLICY`, the views, the decisions and
  `ALMInnerPlan`); `simsopt_alm.checkpoint` exports checkpoint/resume
  (`alm_checkpointing`, the snapshot types, `transition_snapshot`,
  `resume_boundary`, `ALMOuterBoundary`) and `ALMProcessBudgetExhausted`;
  `simsopt_alm.history` exports `ALMHistoryRecorder`.
- Added to `simsopt_alm.__all__`: `ALMProcessBudgetExhausted` (0.1.0:
  `simsopt_alm.control`), which an `accepted_callback` raises to stop a run;
  `simsopt_alm.checkpoint` exports it too.
- Deep imports moved, without aliases: `simsopt_alm.core`, `.control`,
  `.inner`, `.continuation`, `.events`, `.boundary`, `.evaluation`,
  `.hybrid`, `.problem` and `.taylor` no longer exist. Import public names
  from `simsopt_alm` or a facade (the views and decisions from
  `simsopt_alm.policy`); `simsopt_alm._src` paths may change between
  releases.
- Pickles: a pickle names each class by its module path. A 0.1.0 pickle of a
  class from a removed module (`ALMSettings`, `ALMResult`, `ALMLastIterate`,
  `ALMOuterBoundary`, `ALMLoopState`, `ALMFeasibleIncumbent`, the
  measurement types, `ALMPhysics`) fails to load in 0.2.0
  (`ModuleNotFoundError`). A 0.1.0 `ALMTransitionSnapshot` names
  `simsopt_alm.checkpoint`, which 0.2.0 still exports, so it still loads.
- Stable pickle paths: every class and function that `simsopt_alm` or a
  facade exports has that public module as its `__module__` (numpy's
  `set_module` pattern; the defining module is `_module_source`), so 0.2.0
  pickles name `simsopt_alm`, `simsopt_alm.policy`, `simsopt_alm.checkpoint`
  or `simsopt_alm.history`, never `simsopt_alm._src`, and a later move inside
  `_src` does not break them. `inspect.getsource` of such a class raises
  `OSError`, as for numpy's classes.

### Behavior fixes

These repair defects that a review of the solver core found. Iterates,
multipliers, penalties and step
decisions of the golden trajectories are unchanged; the goldens changed only
in the reported fields named here.

- Stationarity with `base_bounds`: `result.stationarity_norm`
  and the convergence test use the L2 norm of the gradient's unit projection
  onto the box, `x - P(x - grad)`: a component pointing out of the box counts
  up to its distance to the bound (before: in full unless x equaled the bound
  exactly), and fixed coordinates count 0. A point just inside a bound can
  now converge where it stalled (`plateau_stall`). The inner L-BFGS-B `gtol`
  is capped at the scheduled stationarity tolerance / sqrt(n), n the dofs the
  bounds do not fix, so a run with many dofs whose L2 norm exceeds the target
  gets inner iterations instead of stopping at once. The history's
  `inner_lbfgsb_projected_gradient_norm` uses the same projection on the inner
  solve's box. Callers: nothing to change; a custom `stationarity_norm` is
  still used as given.
- Multiplier cap: `initial_multipliers` above
  `settings.multiplier_max` raise `ValueError` ("initial ALM multipliers must
  be <= settings.multiplier_max") before any evaluation or callback, and a
  `resume_from` whose multipliers or best-feasible multipliers exceed it raise
  `ValueError` ("resume ALM multipliers ...") before
  `restore_incumbent_state_fn` runs. Before, such multipliers ran and could
  report `converged`. Callers: never pass multipliers above the cap; lower
  them, or raise `multiplier_max`, when warm-starting from a run with a
  larger cap.
- Penalty overflow: a penalty raise whose request overflows to inf
  (with `penalty_max=None`, a cap near the float limit, or a huge
  `penalty_scale`) keeps the finite penalty and ends the run with the new
  reason `penalty_overflow` (`penalty_overflow_restored_best_feasible` after a
  restore), also as the step's `action` and the checkpoint's
  `completed_action`, with `penalty_cap_reached` False and
  `penalty_cap_requested` None. Before, it reported `penalty_cap_reached`
  with an infinite requested penalty, which made the checkpoint plugin raise
  `ValueError`. Callers: handle the two new reasons like
  `penalty_cap_reached`.
- Restore after a spent inner budget: when `inner_options["maxiter"]`
  ends the run and the best feasible iterate is restored, the reason is the
  new `inner_budget_restored_best_feasible` (before:
  `max_outer_restored_best_feasible`, which now means only that the outer
  limit ran out). Callers: match the new reason where you matched the old
  one for a budget stop.
- Skipped inner solve: a step that starts on a spent budget reports
  `inner.attempts` 0, `inner.inner_profile` `"inner_skipped"`,
  `inner.optimizer_success` False and the message `inner_skipped: inner
  maxiter exhausted` (before: 1 attempt, no profile, success True); the
  history's `inner_success` is False and `inner_profile` `"inner_skipped"`,
  also for the pre-inner `converged` shortcut (before True and None).
  `result.success` of a pre-inner convergence stays True. Callers: read
  `inner_success` as "an optimizer ran and succeeded".
- Trust-box-limited inner stop: `ALMInnerSolveOutcome` gains `inner_stop`
  (`"raw_stationarity"`, `"trust_box_limited"` or None; default None) and the
  property `succeeded`. An L-BFGS-B stop on its projected gradient that only
  the attempt's trust box satisfied is `trust_box_limited`: the history's
  `inner_success` is False and the entry carries `inner_trust_box_limited:
  true`. `optimizer_success` and every step decision are unchanged
  (reporting only). Callers: use `inner.succeeded` or the history's
  `inner_success` for "the subproblem is solved".
- Rejected snap onto a bound: when a snap onto a bound is rejected,
  the kept point is evaluated again before `accepted_callback`, so a stateful
  evaluator's live state is at the accepted x (before: at the snapped point).
  Callers: nothing; one extra evaluation in that case.
- Owned diagnostics: where an evaluation enters the solver, every
  array, mapping, list and tuple under its other keys, nested ones too, is
  copied (a shared subtree stays shared, other objects keep their identity),
  so a restored incumbent's diagnostics are those of its own x even when the
  evaluator refills one buffer. A container cycle in a trial evaluation is
  rejected there (`ValueError` "ALM inner trial evaluation is cyclic",
  before "ALM accepted inner evaluation is cyclic"). Callers: nothing.
