"""The continuation-policy API: what an ``ALMContinuationPolicy`` reads and decides.

``minimize_alm(continuation_policy=...)`` takes any object with the methods
of :class:`ALMContinuationPolicy`; :class:`DefaultContinuationPolicy` is the
default (its instance ``DEFAULT_CONTINUATION_POLICY``). A policy reads one view per call (:class:`ALMInnerPlanView`,
:class:`ALMStalledTrialView`, :class:`ALMPreInnerView`,
:class:`ALMPostInnerView`) and returns an :class:`ALMInnerPlan`, a retry
radius, or a decision (:class:`ALMConverge`, :class:`ALMStop`,
:class:`ALMDualUpdateStep`, :class:`ALMRaisePenalty`, :class:`ALMHold`,
:class:`ALMContinue`; their union is ``ALMStepDecision``). The measurements a
view holds are :class:`simsopt_alm.ALMIterateMeasurement` and
:class:`simsopt_alm.ALMInnerSolveOutcome`.
"""

from ._src.policy.decisions import (
    ALMContinue,
    ALMConverge,
    ALMDualUpdateStep,
    ALMHold,
    ALMRaisePenalty,
    ALMStepDecision,
    ALMStop,
)
from ._src.policy.default import DEFAULT_CONTINUATION_POLICY, ALMContinuationPolicy, DefaultContinuationPolicy
from ._src.policy.plan import ALMInnerPlan
from ._src.policy.views import ALMInnerPlanView, ALMPostInnerView, ALMPreInnerView, ALMStalledTrialView

__all__ = [
    "DEFAULT_CONTINUATION_POLICY",
    "ALMContinuationPolicy",
    "ALMContinue",
    "ALMConverge",
    "ALMDualUpdateStep",
    "ALMHold",
    "ALMInnerPlan",
    "ALMInnerPlanView",
    "ALMPostInnerView",
    "ALMPreInnerView",
    "ALMRaisePenalty",
    "ALMStalledTrialView",
    "ALMStepDecision",
    "ALMStop",
    "DefaultContinuationPolicy",
]
