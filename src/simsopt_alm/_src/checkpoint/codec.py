"""The checkpoint encoding: :func:`_transition_json_value` encodes each raw value once (arrays, lists and
tuples as sequences; a raw tuple shaped like a tag included), :func:`_checked_encoded_value` rejects a raw
value in a snapshot (``TypeError``) and :func:`_restore_transition_json_value` inverts the encoding.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Set, Tuple, Union

import numpy as np

from ..loop.events import _cyclic_container_error

ALMTransitionValue = Union[
    str,
    int,
    float,
    bool,
    None,
    Tuple["ALMTransitionValue", ...],
    Tuple[Tuple[str, "ALMTransitionValue"], ...],
]

ALMTransitionEvaluation = Tuple[Tuple[str, ALMTransitionValue], ...]

_ALM_MAPPING_MARKER = "__alm_mapping__"

_ALM_SEQUENCE_MARKER = "__alm_sequence__"

def _transition_json_value(value: object, path: str = "value") -> ALMTransitionValue:
    """Convert a transition value to a recursively immutable finite value.

    Mapping values use an explicit tagged tuple instead of a mutable ``dict``;
    this keeps the frozen transition DTO immutable while allowing the checkpoint
    adapter to serialize it through its ordinary tuple-to-list conversion. A
    mapping, list or tuple that contains itself raises ``ValueError`` naming
    its path from ``path``; a shared subtree is encoded at each place.
    """
    return _encoded_transition_value(value, path, {})

def _encoded_transition_value(
    value: object, path: str, open_paths: Dict[int, str]
) -> ALMTransitionValue:
    """:func:`_transition_json_value` below ``path``; ``open_paths`` holds the
    containers being encoded around it (a hit is a cycle)."""
    if isinstance(value, np.generic) or (isinstance(value, np.ndarray) and value.ndim == 0):
        # A numpy scalar or 0-d array is its Python scalar.
        return _encoded_transition_value(value.item(), path, open_paths)
    if isinstance(value, np.ndarray):
        # ``tolist`` builds fresh lists, which cannot hold themselves.
        return (
            _ALM_SEQUENCE_MARKER,
            tuple(
                _encoded_transition_value(item, path, open_paths)
                for item in value.tolist()
            ),
        )
    if isinstance(value, (Mapping, list, tuple)):
        if id(value) in open_paths:
            raise _cyclic_container_error(
                "ALM transition value", path, open_paths[id(value)]
            )
        open_paths[id(value)] = path
        if isinstance(value, Mapping):
            entries: List[Tuple[str, ALMTransitionValue]] = []
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                if not isinstance(key, str):
                    raise TypeError(
                        "ALM transition mappings require string keys; "
                        f"received {type(key).__name__}"
                    )
                entries.append(
                    (key, _encoded_transition_value(item, f"{path}[{key!r}]", open_paths))
                )
            encoded = (_ALM_MAPPING_MARKER, tuple(entries))
        else:
            encoded = (
                _ALM_SEQUENCE_MARKER,
                tuple(
                    _encoded_transition_value(item, f"{path}[{index}]", open_paths)
                    for index, item in enumerate(value)
                ),
            )
        del open_paths[id(value)]
        return encoded
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("ALM transition values must be finite")
        return value
    raise TypeError(
        f"ALM transition evaluation contains a non-JSON value: {type(value).__name__}"
    )

def _canonical_transition_evaluation(
    evaluation: Sequence[Tuple[str, object]],
) -> ALMTransitionEvaluation:
    """``evaluation``'s key/value pairs, sorted by key, every value checked
    to be encoded (:func:`_checked_encoded_value`)."""
    entries: List[Tuple[str, ALMTransitionValue]] = []
    for item in evaluation:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("ALM transition evaluation must contain key/value pairs")
        key, value = item
        if not isinstance(key, str) or not key:
            raise ValueError("ALM transition evaluation keys must be non-empty strings")
        entries.append((key, _checked_encoded_value(value, f"evaluation[{key!r}]")))
    return tuple(sorted(entries, key=lambda pair: pair[0]))

def _checked_encoded_value(value: object, path: str) -> ALMTransitionValue:
    """``value`` when it is an encoded transition value (what
    :func:`_transition_json_value` returns: a str, None, bool, int or finite
    float, or a sequence or mapping marker with its encoded payload);
    ``TypeError`` naming ``path`` otherwise. Snapshot constructors take the
    checkpoint's content as it is stored, so they check and never encode: a
    raw value (an array, a list, a raw tuple) cannot be told from an encoded
    one by its shape, and ``transition_snapshot`` is where raw values are
    encoded."""
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"ALM transition value {path} must be finite")
        return value
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
        marker, payload = value
        if marker == _ALM_SEQUENCE_MARKER and isinstance(payload, tuple):
            for index, item in enumerate(payload):
                _checked_encoded_value(item, f"{path}[{index}]")
            return value
        if marker == _ALM_MAPPING_MARKER and isinstance(payload, tuple):
            for entry in payload:
                if not (isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str)):
                    break
                _checked_encoded_value(entry[1], f"{path}[{entry[0]!r}]")
            else:
                return value
    raise TypeError(
        f"ALM transition value {path} is not encoded ({type(value).__name__}); "
        "snapshots hold encoded values: build them with transition_snapshot"
    )

def _validate_frozen_transition_evaluation(
    evaluation: ALMTransitionEvaluation,
) -> None:
    if not isinstance(evaluation, tuple):
        raise TypeError("ALM transition evaluation must be an immutable tuple")
    seen_keys: Set[str] = set()
    for item in evaluation:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("ALM transition evaluation entries must be immutable pairs")
        key, value = item
        if not isinstance(key, str) or not key:
            raise ValueError("ALM transition evaluation keys must be non-empty strings")
        if key in seen_keys:
            raise ValueError("ALM transition evaluation keys must be unique")
        seen_keys.add(key)
        _checked_encoded_value(value, f"evaluation[{key!r}]")

_TRANSITION_OMITTED_JACOBIAN_KEYS = frozenset(
    {
        "constraint_grads",
        "raw_constraint_grads",
        "normalized_constraint_grads",
    }
)


def _snapshot_transition_evaluation(
    evaluation: Mapping[str, object],
) -> ALMTransitionEvaluation:
    """The encoded evaluation (every value encoded once, sorted by key).
    Constraint Jacobians stay out of the checkpoint."""
    return _canonical_transition_evaluation(
        tuple(
            (key, _transition_json_value(value, f"evaluation[{key!r}]"))
            for key, value in evaluation.items()
            if key not in _TRANSITION_OMITTED_JACOBIAN_KEYS
        )
    )

def _restore_transition_json_value(value: ALMTransitionValue) -> object:
    """The raw value of an encoded one, the inverse of
    :func:`_transition_json_value` (a snapshot holds encoded values only,
    :func:`_checked_encoded_value`): a sequence marker comes back as a tuple,
    a mapping marker as a dict, a scalar as itself."""
    if not isinstance(value, tuple):
        return value
    marker, payload = value
    if marker == _ALM_SEQUENCE_MARKER:
        return tuple(_restore_transition_json_value(item) for item in payload)
    return {str(key): _restore_transition_json_value(item) for key, item in payload}

def _restore_transition_evaluation(
    evaluation: ALMTransitionEvaluation,
) -> dict:
    return {
        str(key): _restore_transition_json_value(value) for key, value in evaluation
    }
