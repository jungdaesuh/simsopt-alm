"""Opt-in checkpoint/resume for :func:`simsopt_alm.minimize_alm`.

``alm_checkpointing(inner_options, resume_state, completed_outer_callback)``
returns the ``inner_options``, ``resume_from`` and ``on_outer_boundary``
arguments of one checkpointed run (:class:`ALMCheckpointing`). The loop's
:class:`ALMOuterBoundary` becomes an immutable :class:`ALMTransitionSnapshot`
(:func:`transition_snapshot`), and a snapshot the boundary to resume from
(:func:`resume_boundary`). An ``accepted_callback`` that raises
:class:`ALMProcessBudgetExhausted` ends the run at a budget the caller
enforces (``process_budget_exhausted``); a later run resumes from the last
checkpoint. A snapshot pickles its classes by module path, so a checkpoint
pickled by one release of the package may not load in another. Importing
:mod:`simsopt_alm` does not load this module.
"""

from ._src.checkpoint.boundary import ALMOuterBoundary
from ._src.checkpoint.codec import ALMTransitionEvaluation, ALMTransitionValue
from ._src.checkpoint.snapshot import (
    ALMCheckpointing,
    ALMFeasibleIncumbentSnapshot,
    ALMTransitionSnapshot,
    alm_checkpointing,
    resume_boundary,
    transition_snapshot,
)
from ._src.loop.outer import ALMProcessBudgetExhausted

__all__ = [
    "ALMCheckpointing",
    "ALMFeasibleIncumbentSnapshot",
    "ALMOuterBoundary",
    "ALMProcessBudgetExhausted",
    "ALMTransitionEvaluation",
    "ALMTransitionSnapshot",
    "ALMTransitionValue",
    "alm_checkpointing",
    "resume_boundary",
    "transition_snapshot",
]
