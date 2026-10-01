"""What a continuation policy reads: the views of one ALM continuation step.

The loop (:mod:`..loop.outer`) measures the outer iterate and the inner solve
(``ALMIterateMeasurement``, ``ALMInnerSolveOutcome``), shows them to an
``ALMContinuationPolicy`` (:mod:`.default`) through a view, and executes the
frozen decision (:mod:`.decisions`) it gets back. A view is borrowed for one
synchronous call and read-only all the way down: its arrays are non-writable
views of the loop's, its mappings read-only proxies, its lists tuples. It is
not a snapshot: the loop's later writes show through, so a policy copies what
it keeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from ..public_api import set_module
from ..settings.settings import ALMSettings
from ..state import ALMInnerSolveOutcome, ALMIterateMeasurement


@set_module("simsopt_alm.policy")
@dataclass(frozen=True)
class ALMInnerPlanView:
    """One inner L-BFGS-B attempt about to run.

    ``inner_options`` are this process's options (``maxiter`` is what remains
    of its inner budget); ``attempt_radius`` is the trust radius boxing the
    attempt (None: unboxed); ``start_feasible`` is measured once per inner
    solve, at the start iterate.
    """

    settings: ALMSettings
    inner_options: Mapping[str, object]
    update_stationarity_tol: float
    attempt_radius: Optional[float]
    continuation_iteration: int
    start_feasible: bool


@set_module("simsopt_alm.policy")
@dataclass(frozen=True)
class ALMStalledTrialView:
    """A trial the loop classified as an infeasible stall (no move, no gain).

    ``attempt_radius`` is the attempt's trust radius as given (None: no trust
    region); ``inner_false_success`` says L-BFGS-B reported success anyway.
    """

    settings: ALMSettings
    attempt_index: int
    attempt_radius: Optional[float]
    inner_false_success: bool


@set_module("simsopt_alm.policy")
@dataclass(frozen=True)
class ALMPreInnerView:
    """What the convergence check before the inner solve may read.

    ``start`` is the outer iterate measured before the inner solve;
    ``trust_radius`` is the loop's radius and ``feasible_stall_count`` the
    count the step started with.
    """

    settings: ALMSettings
    outer_iteration: int
    continuation_iteration: int
    start: ALMIterateMeasurement
    last_cap_binding_active: bool
    feasible_stall_count: int
    trust_radius: Optional[float]


@set_module("simsopt_alm.policy")
@dataclass(frozen=True)
class ALMPostInnerView:
    """What the decision after the inner solve may read: the
    :class:`ALMPreInnerView` fields plus the judged ``inner`` solve, with
    ``trust_radius`` after the inner solve's own trust rule.
    """

    settings: ALMSettings
    outer_iteration: int
    continuation_iteration: int
    start: ALMIterateMeasurement
    inner: ALMInnerSolveOutcome
    last_cap_binding_active: bool
    feasible_stall_count: int
    trust_radius: Optional[float]
