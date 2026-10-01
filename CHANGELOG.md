# Changelog

## 0.2.0

- Layout: the implementation moved into `simsopt_alm._src` (one subpackage
  per component: `settings`, `problem`, `policy`, `inner`, `loop`,
  `checkpoint`, plus the shared `state` and `alm_math` modules), behind
  public facades. Solver behavior is unchanged: every golden trajectory
  replays byte-identically.
- Public imports: everything in `simsopt_alm.__all__` imports from
  `simsopt_alm` as before, and `import simsopt_alm` still binds
  `simsopt_alm.policy`. `simsopt_alm.policy` exports what a policy author
  needs (the protocol, `DefaultContinuationPolicy` and its instance
  `DEFAULT_CONTINUATION_POLICY`, the views, the decisions and
  `ALMInnerPlan`); `simsopt_alm.checkpoint` exports checkpoint/resume
  (`alm_checkpointing`, the snapshot types, `transition_snapshot`,
  `resume_boundary`, `ALMOuterBoundary`) and `ALMProcessBudgetExhausted`;
  `simsopt_alm.history` exports `ALMHistoryRecorder`.
- Deep imports moved, without aliases: `simsopt_alm.core`, `.control`,
  `.inner`, `.continuation`, `.events`, `.boundary`, `.evaluation`,
  `.hybrid`, `.problem` and `.taylor` no longer exist. Import public names
  from `simsopt_alm` or a facade (`ALMProcessBudgetExhausted` from
  `simsopt_alm.checkpoint`; the views and decisions from
  `simsopt_alm.policy`); `simsopt_alm._src` paths may change between
  releases.
- Pickles: a pickle names each class by its module path. A 0.1.0 pickle of a
  class from a removed module (`ALMSettings`, `ALMResult`, `ALMLastIterate`,
  `ALMOuterBoundary`, `ALMLoopState`, `ALMFeasibleIncumbent`, the
  measurement types, `ALMPhysics`) fails to load in 0.2.0
  (`ModuleNotFoundError`). A 0.1.0 `ALMTransitionSnapshot` names
  `simsopt_alm.checkpoint`, which 0.2.0 still exports, so it still loads; a
  0.2.0 checkpoint names `simsopt_alm._src.checkpoint.snapshot`, which 0.1.0
  cannot load.
