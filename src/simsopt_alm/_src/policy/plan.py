"""What a continuation policy plans for one inner L-BFGS-B attempt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ALMInnerPlan:
    """L-BFGS-B ``options`` for one attempt (read-only; a call budget caps
    their ``maxiter`` at the view's) and the ``profile`` the step publishes."""

    profile: str
    options: Mapping[str, object]
