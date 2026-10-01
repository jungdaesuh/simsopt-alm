"""The public identity of the package's API (numpy's ``set_module`` pattern).

Every class and function a facade exports carries that facade as its
``__module__`` (``simsopt_alm``, ``simsopt_alm.policy``,
``simsopt_alm.checkpoint`` or ``simsopt_alm.history``), set where it is
defined. A pickle names a class or function by its module path, so pickles
name the facades, never ``simsopt_alm._src``, and a move inside ``_src``
breaks no saved pickle.
"""

from __future__ import annotations

import typing
from typing import Callable, TypeVar

PublicT = TypeVar("PublicT")


def set_module(module: str) -> Callable[[PublicT], PublicT]:
    """Decorator: set the ``__module__`` of a class or function to ``module``.

    A class first records its defining module as ``_module_source`` (numpy's
    name) and has its own string annotations evaluated there, so
    ``typing.get_type_hints`` still resolves them. ``inspect.getsource`` of
    such a class raises ``OSError`` (as for numpy's): its source is in
    ``sys.modules[cls._module_source]``. Apply it outermost, after
    ``@dataclass``.
    """

    def decorator(obj: PublicT) -> PublicT:
        if isinstance(obj, type):
            obj._module_source = obj.__module__
            own_annotations = obj.__dict__.get("__annotations__", {})
            if own_annotations:
                hints = typing.get_type_hints(obj)
                obj.__annotations__ = {name: hints[name] for name in own_annotations}
        obj.__module__ = module
        return obj

    return decorator
