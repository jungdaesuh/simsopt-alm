"""What a continuation policy decides: the frozen decision of one ALM
continuation step, which the loop (:mod:`..loop.outer`) executes as given.
Every decision carries the ``feasible_stall_count`` the loop keeps after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Union

from ..public_api import set_module


@set_module("simsopt_alm.policy")
@dataclass(frozen=True)
class ALMConverge:
    """Return success: publish ``action`` and report ``message``."""

    action: str
    termination_reason: str
    message: str
    feasible_stall_count: int


@set_module("simsopt_alm.policy")
@dataclass(frozen=True)
class ALMStop:
    """Return failure after publishing ``action``; the best-feasible restore
    applies. ``marks_max_outer`` publishes ``max_outer`` on the final outer."""

    action: str
    termination_reason: str
    message_prefix: str
    feasible_stall_count: int
    subproblem_limit_reason: Optional[str] = None
    marks_max_outer: bool = False


# The decisions below do not return. ``max_outer_termination`` is the run's
# termination reason when such a step ends the final outer iteration.


@set_module("simsopt_alm.policy")
@dataclass(frozen=True)
class ALMDualUpdateStep:
    """Update the multipliers and end the outer iteration; with a
    ``penalty_reason`` raise the penalty in the same step."""

    penalty_reason: Optional[str]
    feasible_stall_count: int
    max_outer_termination: ClassVar[str] = "max_outer_after_dual_update"


@set_module("simsopt_alm.policy")
@dataclass(frozen=True)
class ALMRaisePenalty:
    """Raise the penalty and end the outer iteration (or return at the cap)."""

    action: str
    max_outer_termination: str
    feasible_stall_count: int
    subproblem_limit_reason: Optional[str] = None
    signal_mismatch_repair: bool = False


@set_module("simsopt_alm.policy")
@dataclass(frozen=True)
class ALMHold:
    """Keep the penalty and multipliers and end the outer iteration."""

    feasible_stall_count: int
    max_outer_termination: ClassVar[str] = "max_outer_after_sufficient_decrease_hold"


@set_module("simsopt_alm.policy")
@dataclass(frozen=True)
class ALMContinue:
    """Solve the subproblem again with this trust radius (None: no box, else
    positive) and update stationarity tolerance."""

    trust_radius: Optional[float]
    update_stationarity_tol: float
    feasible_stall_count: int
    signal_mismatch_repair: bool = False
    max_outer_termination: ClassVar[str] = "max_outer"


ALMStepDecision = Union[
    ALMConverge, ALMStop, ALMDualUpdateStep, ALMRaisePenalty, ALMHold, ALMContinue
]
