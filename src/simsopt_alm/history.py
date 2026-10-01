"""Opt-in ALM history for :func:`simsopt_alm.minimize_alm`.

:class:`ALMHistoryRecorder` builds one entry per outer-step decision: pass
``recorder.record`` as ``on_outer_step``, then read ``recorder.history()``.
Importing :mod:`simsopt_alm` does not load this module.
"""

from ._src.loop.history import ALMHistoryRecorder

__all__ = ["ALMHistoryRecorder"]
