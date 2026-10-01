"""The runtime types the solver's components share: the routing state of an
evaluation's constraint signals, the measurement of an iterate and of an inner
solve, and what the loop carries and publishes (the best-feasible incumbent
and the geometry identity of an accepted state, the dual update, the loop
state, the outer-step event). Observers get them as read-only copies
(:mod:`.state_views`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Mapping, Optional, Protocol, Sequence, Tuple, TypeVar, runtime_checkable

import numpy as np

from .public_api import set_module

@set_module("simsopt_alm")
@dataclass(frozen=True)
class ALMConstraintSignalState:
    explicit_hybrid_signals: bool
    hard_signed_constraint_values: np.ndarray
    hard_violation_values: np.ndarray
    surrogate_signed_constraint_values: np.ndarray
    preferred_dual_update_values: np.ndarray

@set_module("simsopt_alm")
@dataclass(frozen=True)
class ALMConstraintRoutingState:
    signal_state: ALMConstraintSignalState
    hard_activity_mask: np.ndarray
    surrogate_activity_mask: np.ndarray
    signal_mismatch_active: bool
    hard_positive_shift: np.ndarray
    surrogate_positive_shift: np.ndarray
    hard_positive_shift_zero: bool
    surrogate_positive_shift_zero: bool
    hard_max_violation: float
    surrogate_max_value: float

@set_module("simsopt_alm")
@dataclass(frozen=True)
class ALMIterateMeasurement:
    """One evaluation of the problem and the solver quantities derived from it.

    ``evaluation`` is the evaluation at (x, ``multipliers``, ``penalty``). Inside
    the loop it is the solver's snapshot of the evaluator's dict, whose arrays
    no later evaluation changes; in an outer-step event all of it is an owned
    read-only copy, in a policy view a borrowed read-only view. The tolerances are the ones this iterate was judged with.
    """

    evaluation: Mapping[str, object]
    multipliers: np.ndarray
    penalty: float
    solver_constraint_values: np.ndarray
    feasibility_values: np.ndarray
    max_feasibility_violation: float
    routing_state: ALMConstraintRoutingState
    stationarity_norm: float
    kkt_stationarity_norm: Optional[float]
    signal_mismatch_active: bool
    update_feasibility_tol: float
    update_stationarity_tol: float
    effective_feasibility_tol: float


@set_module("simsopt_alm")
@dataclass(frozen=True)
class ALMInnerSolveOutcome:
    """The inner solve of one continuation step and how the loop judged it.

    ``x``/``measured`` are the accepted iterate (the start iterate when every
    trial was rejected); ``bounds`` and ``inner_options`` are the last
    attempt's box (None: unbounded) and L-BFGS-B options. The
    ``sufficient_decrease_*`` pair is the ALGENCAN measure and its reference.
    """

    x: np.ndarray
    measured: ALMIterateMeasurement
    iterations: int
    attempts: int
    optimizer_success: bool
    optimizer_message: str
    bounds: Optional[Sequence[Tuple[float, float]]]
    inner_options: Optional[Mapping[str, object]]
    inner_profile: Optional[str]
    infeasible_stall: bool
    infeasible_stall_reason: Optional[str]
    inner_false_success: bool
    nonfinite_candidate_evaluation: bool
    nonfinite_candidate_fields: Optional[Sequence[str]]
    meaningful_progress: bool
    feasibility_delta: float
    feasibility_delta_tolerance: float
    sufficient_decrease_measure: float
    sufficient_decrease_reference: float

AcceptedStateT = TypeVar("AcceptedStateT")

@runtime_checkable
class ALMGeometryIdentityCarrier(Protocol):
    """Accepted-state protocol for checkpointable geometry identities.

    The ALM core deliberately does not know how a regime serializes geometry.
    A checkpoint-capable regime exposes the resulting content identity through
    this small protocol so that the identity travels with the core incumbent
    instead of being recomputed by a later adapter.
    """

    @property
    def geometry_identity(self) -> str: ...

def _accepted_state_geometry_identity(
    accepted_state: Optional[AcceptedStateT],
) -> Optional[str]:
    """Read an identity only from an explicitly checkpointable carrier."""
    if accepted_state is None:
        return None
    if not isinstance(accepted_state, ALMGeometryIdentityCarrier):
        return None
    identity = accepted_state.geometry_identity
    if not isinstance(identity, str) or not identity:
        raise ValueError("checkpointable accepted-state geometry_identity must be non-empty")
    return identity

def _validate_geometry_identity_pair(
    *,
    accepted_state: Optional[AcceptedStateT],
    geometry_identity: Optional[str],
    context: str,
) -> None:
    """Keep a checkpointable accepted carrier and its identity inseparable."""
    if geometry_identity is not None and accepted_state is None:
        raise ValueError(f"{context} geometry_identity requires accepted_state")
    carrier_identity = _accepted_state_geometry_identity(accepted_state)
    if carrier_identity is not None:
        if geometry_identity is None:
            raise ValueError(
                f"{context} checkpointable accepted_state requires geometry_identity"
            )
        if geometry_identity != carrier_identity:
            raise ValueError(f"{context} geometry_identity does not match accepted_state")

@set_module("simsopt_alm")
@dataclass(frozen=True)
class ALMFeasibleIncumbent(Generic[AcceptedStateT]):
    x: np.ndarray
    evaluation: Mapping[str, object]
    multipliers: np.ndarray
    penalty: float
    inner_result: object
    incumbent_state: Optional[AcceptedStateT] = None
    geometry_identity: Optional[str] = None

    def __post_init__(self) -> None:
        carrier_identity = _accepted_state_geometry_identity(self.incumbent_state)
        if self.geometry_identity is None:
            object.__setattr__(self, "geometry_identity", carrier_identity)
        elif not isinstance(self.geometry_identity, str) or not self.geometry_identity:
            raise ValueError("ALM best-feasible geometry_identity must be non-empty")
        elif carrier_identity is not None and self.geometry_identity != carrier_identity:
            raise ValueError(
                "ALM best-feasible geometry_identity does not match incumbent_state"
            )
        _validate_geometry_identity_pair(
            accepted_state=self.incumbent_state,
            geometry_identity=self.geometry_identity,
            context="ALM best-feasible incumbent",
        )

@set_module("simsopt_alm")
@dataclass(frozen=True)
class ALMDualUpdate:
    """Projected multipliers after a dual step and the tightened tolerances."""

    multipliers: np.ndarray
    update_feasibility_tol: float
    update_stationarity_tol: float
    multiplier_cap_binding: bool
    multiplier_cap_binding_indices: Sequence[int]

@set_module("simsopt_alm")
@dataclass(frozen=True)
class ALMLoopState(Generic[AcceptedStateT]):
    """What the outer loop carries after a decision: the content of a checkpoint.

    ``best_feasible`` omits the inner optimizer result; its ``incumbent_state``
    is the caller's own object, and None in a skipped-inner step, whose state
    the caller snapshots only after ``on_outer_step``. ``sufficient_decrease_*``
    is the latest accepted measure with its multipliers and penalty.
    """

    x: np.ndarray
    multipliers: np.ndarray
    penalty: float
    update_feasibility_tol: float
    update_stationarity_tol: float
    trust_radius: Optional[float]
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]]
    total_inner_iterations: int
    last_cap_binding_active: bool
    cap_binding_detected: bool
    cap_binding_indices: Tuple[int, ...]
    penalty_cap_reached: bool
    penalty_cap_requested: Optional[float]
    sufficient_decrease_measure: Optional[float]
    sufficient_decrease_multipliers: Optional[np.ndarray]
    sufficient_decrease_penalty: Optional[float]

@set_module("simsopt_alm")
@dataclass(frozen=True)
class ALMOuterStepEvent(Generic[AcceptedStateT]):
    """One decision of the outer loop, passed to ``minimize_alm(on_outer_step=...)``.

    Emitted once per continuation step when its decision is final, before a
    terminal result is built. The package docstring lists the fields. An event
    is a read-only snapshot: arrays are not writable and mappings are
    ``MappingProxyType`` views, so it cannot be pickled or deep-copied; an
    observer that stores events converts the fields it needs. Objects the
    evaluator put in its dict that are not arrays, mappings or sequences
    (and the caller's accepted states) are passed by reference.
    """

    outer_iteration: int
    continuation_iteration: int
    constraint_names: Tuple[str, ...]
    action: str
    outer_termination: Optional[str]
    subproblem_limit_reason: Optional[str]
    signal_mismatch_repair: bool
    feasible_stall_count: int
    start_x: np.ndarray
    start: ALMIterateMeasurement
    inner: Optional[ALMInnerSolveOutcome]
    dual_update: Optional[ALMDualUpdate]
    penalty_update: Optional[ALMIterateMeasurement]
    dual_update_penalty_reason: Optional[str]
    after: ALMLoopState[AcceptedStateT]

    @property
    def measured(self) -> ALMIterateMeasurement:
        """The iterate the decision was made on (``start`` when no inner solve ran)."""
        return self.start if self.inner is None else self.inner.measured
