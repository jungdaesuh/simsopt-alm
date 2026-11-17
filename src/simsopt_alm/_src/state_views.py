"""The read-only copy rule by which the ALM loop hands out its data.

:func:`~.loop.outer.minimize_alm` passes one ``ALMOuterStepEvent`` per
continuation-step decision to ``on_outer_step``; the event's ``after``, and
every outer boundary's ``state``, is an ``ALMLoopState``, whose incumbent is an
``ALMFeasibleIncumbent`` (the types are in :mod:`.state`). Observers get owned,
read-only copies (:func:`_frozen_event_value`): arrays not writable, mappings
read-only views, lists and tuples tuples; :func:`_writable_copy` undoes that
for plain values. A continuation policy gets the same shape without the
copies (:func:`_borrowed_read_only_value`): read-only views of the loop's
arrays, borrowed for one synchronous call. Copying needs the mappings, lists and tuples to be acyclic;
a shared subtree is fine. Where such data enters the package (the loop's
evaluations, ``ALMPhysics`` extras, a resume boundary),
:func:`_require_acyclic_containers` rejects a cycle, naming its path; the
checkpoint encoder does the same in its own walk.
"""

from __future__ import annotations

from dataclasses import fields, replace
from types import MappingProxyType
from typing import Callable, Dict, Mapping, Optional, Set

import numpy as np

from .state import (AcceptedStateT, ALMConstraintRoutingState, ALMConstraintSignalState, ALMDualUpdate,
                    ALMFeasibleIncumbent, ALMInnerSolveOutcome, ALMIterateMeasurement, ALMLoopState)

_EVENT_CARRIERS = (
    ALMIterateMeasurement,
    ALMConstraintRoutingState,
    ALMConstraintSignalState,
    ALMInnerSolveOutcome,
    ALMDualUpdate,
)

class _FrozenList(tuple):
    """A frozen list: a tuple that :func:`_writable_copy` turns back into a list."""

    __slots__ = ()

# The containers the read-only copy rule descends into.
_DATA_CONTAINERS = (Mapping, list, tuple)

def _cyclic_container_error(context: str, path: str, earlier_path: str) -> ValueError:
    return ValueError(
        f"{context} is cyclic: {path} is {earlier_path}; mappings, lists and "
        "tuples in ALM data must not contain themselves"
    )

def _require_acyclic_containers(value: object, *, context: str, path: str) -> None:
    """Raise ``ValueError`` if a mapping, list or tuple in ``value`` contains
    itself, directly or through others; ``path`` names ``value`` in the message.

    A container reached twice without a cycle (a shared subtree) is fine.
    Other objects are not traversed.
    """
    if isinstance(value, _DATA_CONTAINERS):
        _visit_containers(value, path, {}, set(), context)

def _visit_containers(
    container: object,
    path: str,
    open_paths: Dict[int, str],
    closed: Set[int],
    context: str,
) -> None:
    """Depth-first walk: ``open_paths`` holds the containers on the current
    path (a hit is a cycle), ``closed`` the ones already walked in full."""
    open_paths[id(container)] = path
    items = container.items() if isinstance(container, Mapping) else enumerate(container)
    for key, item in items:
        if not isinstance(item, _DATA_CONTAINERS) or id(item) in closed:
            continue
        item_path = f"{path}[{key!r}]"
        if id(item) in open_paths:
            raise _cyclic_container_error(context, item_path, open_paths[id(item)])
        _visit_containers(item, item_path, open_paths, closed, context)
    del open_paths[id(container)]
    closed.add(id(container))

def _frozen_event_value(value: object, memo: Dict[int, object]) -> object:
    """An owned, read-only copy of ``value`` for an outer-step event.

    Arrays become read-only copies, mappings read-only views of frozen
    values, lists and tuples tuples (a list a :class:`_FrozenList`), and the
    solver's carriers are rebuilt field by field; anything else (scalars,
    strings, None, and any other object the evaluator returned) passes through
    by reference. ``memo`` (by ``id``) keeps shared objects shared and copies
    each once; its keys' objects must stay alive while it is in use.
    ``value`` is acyclic (checked where it entered the package).
    """
    return _read_only_value(value, memo, _read_only_array_copy)

def _borrowed_read_only_value(value: object, memo: Dict[int, object]) -> object:
    """:func:`_frozen_event_value` without the copies: arrays become
    non-writable views, so the owner's arrays keep their flags and the
    owner's later writes show through. For one synchronous call."""
    return _read_only_value(value, memo, _read_only_array_view)

def _read_only_array_copy(array: np.ndarray) -> np.ndarray:
    owned = array.copy()
    owned.setflags(write=False)
    return owned

def _read_only_array_view(array: np.ndarray) -> np.ndarray:
    borrowed = array.view()
    borrowed.setflags(write=False)
    return borrowed

def _read_only_value(
    value: object,
    memo: Dict[int, object],
    read_only_array: Callable[[np.ndarray], np.ndarray],
) -> object:
    """The read-only rule of :func:`_frozen_event_value`, with each array
    made read-only by ``read_only_array``."""
    read_only = memo.get(id(value))
    if read_only is not None:
        return read_only
    if isinstance(value, np.ndarray):
        read_only = read_only_array(value)
    elif isinstance(value, Mapping):
        read_only = MappingProxyType(
            {
                key: _read_only_value(item, memo, read_only_array)
                for key, item in value.items()
            }
        )
    elif isinstance(value, list):
        read_only = _FrozenList(
            _read_only_value(item, memo, read_only_array) for item in value
        )
    elif isinstance(value, tuple):
        read_only = tuple(_read_only_value(item, memo, read_only_array) for item in value)
    elif isinstance(value, _EVENT_CARRIERS):
        read_only = replace(
            value,
            **{
                field.name: _read_only_value(getattr(value, field.name), memo, read_only_array)
                for field in fields(value)
            },
        )
    else:
        return value
    memo[id(value)] = read_only
    return read_only

def _writable_copy(value: object, memo: Optional[Dict[int, object]] = None) -> object:
    """An owned, writable copy of ``value``, and the inverse of
    :func:`_frozen_event_value` for plain values: writable array copies,
    dicts, lists for lists and frozen lists, tuples; others by reference.
    Container subclasses (namedtuples, ``OptimizeResult``) come back as plain
    tuples and dicts; list items are not coerced to float arrays. ``memo``
    (by ``id``) keeps a shared subtree shared; ``value`` is acyclic."""
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    owned: object
    if isinstance(value, np.ndarray):
        owned = value.copy()
    elif isinstance(value, Mapping):
        owned = {key: _writable_copy(item, memo) for key, item in value.items()}
    elif isinstance(value, (list, _FrozenList)):
        owned = [_writable_copy(item, memo) for item in value]
    elif isinstance(value, tuple):
        owned = tuple(_writable_copy(item, memo) for item in value)
    else:
        return value
    memo[id(value)] = owned
    return owned

def _frozen_event_incumbent(
    incumbent: Optional[ALMFeasibleIncumbent[AcceptedStateT]],
    memo: Dict[int, object],
) -> Optional[ALMFeasibleIncumbent[AcceptedStateT]]:
    """The incumbent's solver data, frozen; the caller's state passes through."""
    if incumbent is None:
        return None
    return ALMFeasibleIncumbent(
        x=_frozen_event_value(incumbent.x, memo),
        evaluation=_frozen_event_value(incumbent.evaluation, memo),
        multipliers=_frozen_event_value(incumbent.multipliers, memo),
        penalty=incumbent.penalty,
        inner_result=None,
        incumbent_state=incumbent.incumbent_state,
        geometry_identity=incumbent.geometry_identity,
    )

def _frozen_loop_state(
    state: ALMLoopState[AcceptedStateT],
    memo: Dict[int, object],
) -> ALMLoopState[AcceptedStateT]:
    """``state`` with owned read-only arrays and a frozen incumbent (its inner
    result dropped, the caller's accepted state passed through)."""
    return replace(
        state,
        x=_frozen_event_value(state.x, memo),
        multipliers=_frozen_event_value(state.multipliers, memo),
        best_feasible=_frozen_event_incumbent(state.best_feasible, memo),
        sufficient_decrease_multipliers=_frozen_event_value(
            state.sufficient_decrease_multipliers, memo
        ),
    )
