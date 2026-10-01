"""``ALMSettings`` and the finite-value parsers its validation shares with the
loop's runtime checks (inner options, checkpoints, the history recorder)."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Optional

import numpy as np

def _finite_alm_value(name: str, value) -> float:
    value_f = float(value)
    if not np.isfinite(value_f):
        raise ValueError(f"{name} must be finite")
    return value_f

def _finite_alm_value_or_none(name: str, value) -> Optional[float]:
    if value is None:
        return None
    return _finite_alm_value(name, value)

def _finite_alm_integer(name: str, value) -> int:
    _finite_alm_value(name, value)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)

def _finite_alm_integer_or_none(name: str, value) -> Optional[int]:
    if value is None:
        return None
    return _finite_alm_integer(name, value)

def _nonnegative_alm_integer(name: str, value) -> int:
    value_i = _finite_alm_integer(name, value)
    if value_i < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value_i

def _strict_alm_bool(name: str, value) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value

_MUST_BE_POSITIVE = (lambda value: value <= 0.0, "must be positive")
_POSITIVE_WHEN_PROVIDED = (lambda value: value <= 0.0, "must be positive when provided")
_MUST_BE_NONNEGATIVE = (lambda value: value < 0.0, "must be nonnegative")
_MUST_EXCEED_ONE = (lambda value: value <= 1.0, "must be greater than 1")
_IN_OPEN_UNIT_INTERVAL = (lambda value: not (0.0 < value < 1.0), "must be in (0, 1)")

# ``ALMSettings`` numeric fields: (name, finite parser, (out_of_range, message)).
# All fields are parsed in this order before any range check runs; a None
# (disabled optional) value skips its range check. Zero subproblem
# continuations is valid: the inner loop is ``range(n + 1)``.
_ALM_SETTINGS_BOUNDS = (
    ("max_outer_iterations", _finite_alm_integer, _MUST_BE_POSITIVE),
    ("max_subproblem_continuations", _finite_alm_integer, _MUST_BE_NONNEGATIVE),
    ("penalty_init", _finite_alm_value, _MUST_BE_POSITIVE),
    ("penalty_scale", _finite_alm_value, _MUST_EXCEED_ONE),
    ("penalty_max", _finite_alm_value_or_none, _POSITIVE_WHEN_PROVIDED),
    ("penalty_sufficient_decrease_tau", _finite_alm_value, _IN_OPEN_UNIT_INTERVAL),
    ("feasibility_tol", _finite_alm_value, _MUST_BE_POSITIVE),
    ("stationarity_tol", _finite_alm_value, _MUST_BE_POSITIVE),
    ("trust_radius_init", _finite_alm_value_or_none, _POSITIVE_WHEN_PROVIDED),
    ("trust_radius_min", _finite_alm_value, _MUST_BE_POSITIVE),
    ("trust_radius_shrink", _finite_alm_value, _IN_OPEN_UNIT_INTERVAL),
    ("trust_radius_grow", _finite_alm_value, _MUST_EXCEED_ONE),
    ("max_inner_attempts", _finite_alm_integer, _MUST_BE_POSITIVE),
    ("relaxed_feasibility_gate_cap", _finite_alm_value, _MUST_BE_POSITIVE),
    ("multiplier_max", _finite_alm_value_or_none, _POSITIVE_WHEN_PROVIDED),
    ("history_max_entries", _finite_alm_integer_or_none, _POSITIVE_WHEN_PROVIDED),
)

@dataclass(frozen=True)
class ALMSettings:
    max_outer_iterations: int = 10
    max_subproblem_continuations: int = 20
    penalty_init: float = 1.0
    penalty_scale: float = 10.0
    # Production ALM implementations commonly expose a maximum penalty
    # safeguard; keeping this bounded avoids runaway objective scaling in the
    # L-BFGS-B inner solve.
    penalty_max: Optional[float] = 1.0e8
    feasibility_tol: float = 1e-6
    stationarity_tol: float = 1e-6
    trust_radius_init: Optional[float] = None
    trust_radius_min: float = 1e-4
    trust_radius_shrink: float = 0.5
    trust_radius_grow: float = 1.5
    max_inner_attempts: int = 4
    relaxed_feasibility_gate_cap: float = 1e-2
    multiplier_max: Optional[float] = 1.0e6
    # Read only by ALMHistoryRecorder.from_settings; minimize_alm keeps no history.
    history_max_entries: Optional[int] = 512
    # Opt-in: when True, a hard-feasible/surrogate-active signal mismatch with
    # a live surrogate positive shift stays on the bounded inner-continuation
    # path instead of taking the penalty-increase arm. This does not
    # bypass the dual-update stationarity gate.
    continue_on_signal_mismatch: bool = False
    # ALGENCAN sufficient-decrease safeguard (Birgin & Martinez 2014,
    # Algorithm 1.1, default tau=0.5): the generic infeasible penalty-increase
    # arm holds the penalty fixed when the shifted infeasibility measure
    # ||max(g, -lambda/rho)||_inf shrank to <= tau x the previous accepted
    # iterate's measure. Without this the penalty ramps unconditionally every
    # outer iteration and a late-activating constraint meets an
    # already-inflated penalty (the resulting jump in step size can break a
    # stateful inner solve).
    penalty_sufficient_decrease_tau: float = 0.5

    def __post_init__(self) -> None:
        # Every construction path is validated here, so a value such as
        # `ALMSettings(trust_radius_grow=0.5)` is rejected rather than
        # silently shrinking the trust radius.
        parsed = {
            name: parse(f"ALMSettings.{name}", getattr(self, name))
            for name, parse, _bound in _ALM_SETTINGS_BOUNDS
        }
        _strict_alm_bool(
            "ALMSettings.continue_on_signal_mismatch",
            self.continue_on_signal_mismatch,
        )
        for name, _parse, (out_of_range, requirement) in _ALM_SETTINGS_BOUNDS:
            value = parsed[name]
            if value is not None and out_of_range(value):
                raise ValueError(f"ALMSettings.{name} {requirement}")
            if name == "penalty_max" and value is not None and (
                value < parsed["penalty_init"]
            ):
                raise ValueError(
                    f"ALMSettings.penalty_max ({self.penalty_max}) must be >= "
                    f"penalty_init ({self.penalty_init})"
                )
