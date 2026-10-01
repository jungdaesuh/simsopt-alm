"""Public classes and functions pickle under their public module paths.

numpy's ``set_module`` pattern: every class and function a facade exports
(``simsopt_alm``, ``simsopt_alm.policy``, ``simsopt_alm.checkpoint``,
``simsopt_alm.history``) has that facade as its ``__module__`` (the root
first, for a name two facades export). A pickle names a class by its module
path, so pickles name the facades, never ``simsopt_alm._src``, unpickling
resolves through them, and a move inside ``_src`` breaks no saved pickle. The
class keeps its defining module as ``_module_source``; ``typing``,
``inspect.signature``, ``dataclasses`` and ``help`` still work.

Instances holding the solver's read-only mappings (``ALMResult``,
``ALMPhysics``, outer-step events, boundaries with an incumbent) cannot be
pickled at all, as in 0.1.0; their classes still pickle by reference.
"""

import dataclasses
import inspect
import json
from collections.abc import Mapping
import pickle
import pickletools
import pydoc
import subprocess
import sys
import tempfile
import typing
import unittest
from pathlib import Path

import numpy as np

import simsopt_alm as alm
from simsopt_alm import checkpoint as alm_checkpoint
from simsopt_alm import history as alm_history
from simsopt_alm import policy as alm_policy

# The facades in precedence order: a name two of them export belongs to the first.
FACADES = (alm, alm_policy, alm_checkpoint, alm_history)


def _public_objects():
    """``(name, object, facade name)`` for each class and function the facades export."""
    seen = set()
    for facade in FACADES:
        for name in facade.__all__:
            value = getattr(facade, name)
            if (inspect.isclass(value) or inspect.isfunction(value)) and id(value) not in seen:
                seen.add(id(value))
                yield name, value, facade.__name__


def _bits(value) -> object:
    """A comparable record of a value: arrays by dtype, shape and bytes,
    floats by hex, containers, dataclasses and plain objects item by item
    (with their class), exceptions by class and arguments."""
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (float, np.floating)):
        return ("float", float(value).hex())
    if isinstance(value, Mapping):
        return ("mapping", tuple((key, _bits(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_bits(item) for item in value))
    if dataclasses.is_dataclass(value):
        return (type(value).__qualname__, tuple(
            (field.name, _bits(getattr(value, field.name))) for field in dataclasses.fields(value)
        ))
    if isinstance(value, BaseException):
        return (type(value).__qualname__, _bits(value.args))
    if hasattr(value, "__dict__"):
        return (type(value).__qualname__, _bits(vars(value)))
    return (type(value).__name__, value)


def _pickled_simsopt_alm_modules(data: bytes) -> set:
    """The ``simsopt_alm`` module paths a pickle names."""
    return {
        argument
        for _opcode, argument, _position in pickletools.genops(data)
        if isinstance(argument, str) and argument.startswith("simsopt_alm")
    }


def _physics(x):
    return alm.ALMPhysics(
        base_value=float(x @ x),
        base_grad=2.0 * x,
        constraint_values=np.array([1.0 - x[0]]),
        constraint_grads=(np.array([-1.0, 0.0]),),
    )


def _checkpointed_run():
    """min ||x||^2 s.t. 1 - x0 <= 0 from the feasible (3, 2), checkpointed and
    recorded: its result, published boundaries, snapshots and recorder."""
    snapshots, boundaries = [], []
    plugin = alm_checkpoint.alm_checkpointing({"maxiter": 200}, completed_outer_callback=snapshots.append)
    recorder = alm_history.ALMHistoryRecorder(max_entries=4)

    def on_outer_boundary(boundary):
        boundaries.append(boundary)
        plugin.on_outer_boundary(boundary)

    result = alm.minimize_alm(
        np.array([3.0, 2.0]), ["x0_at_least_one"], alm.cached_alm_evaluator(_physics),
        alm.ALMSettings(max_outer_iterations=6), plugin.inner_options,
        on_outer_boundary=on_outer_boundary, on_outer_step=recorder.record,
    )
    return result, boundaries, snapshots, recorder


def _picklable_instances():
    """``(label, instance)`` of every public class whose instances pickle."""
    result, boundaries, snapshots, recorder = _checkpointed_run()
    snapshot = next(snapshot for snapshot in snapshots if snapshot.best_feasible is not None)
    state = dataclasses.replace(boundaries[0].state, best_feasible=None)
    routing = alm.ALMConstraintRoutingState(
        signal_state=alm.ALMConstraintSignalState(
            explicit_hybrid_signals=False,
            hard_signed_constraint_values=np.array([-1.0]),
            hard_violation_values=np.array([0.0]),
            surrogate_signed_constraint_values=np.array([-1.0]),
            preferred_dual_update_values=np.array([-1.0]),
        ),
        hard_activity_mask=np.array([False]),
        surrogate_activity_mask=np.array([False]),
        signal_mismatch_active=False,
        hard_positive_shift=np.array([0.0]),
        surrogate_positive_shift=np.array([0.0]),
        hard_positive_shift_zero=True,
        surrogate_positive_shift_zero=True,
        hard_max_violation=0.0,
        surrogate_max_value=-1.0,
    )
    evaluation = _physics(np.array([1.0, 0.0])).evaluation(np.array([2.0]), 1.0)
    measured = alm.ALMIterateMeasurement(
        evaluation=evaluation, multipliers=np.array([2.0]), penalty=1.0,
        solver_constraint_values=np.array([0.0]), feasibility_values=np.array([0.0]),
        max_feasibility_violation=0.0, routing_state=routing, stationarity_norm=0.0,
        kkt_stationarity_norm=None, signal_mismatch_active=False, update_feasibility_tol=1.0,
        update_stationarity_tol=1.0, effective_feasibility_tol=1.0,
    )
    return (
        ("ALMSettings", alm.ALMSettings(max_outer_iterations=6)),
        ("ALMLastIterate", result.last_iterate),
        ("ALMTransitionSnapshot with incumbent", snapshot),
        ("ALMFeasibleIncumbentSnapshot", snapshot.best_feasible),
        ("ALMCheckpointing", alm_checkpoint.alm_checkpointing({"maxiter": 5})),
        ("ALMLoopState", state),
        ("ALMOuterBoundary without incumbent", dataclasses.replace(boundaries[0], state=state)),
        ("ALMFeasibleIncumbent", alm.ALMFeasibleIncumbent(
            x=np.array([1.0, 0.0]), evaluation=evaluation, multipliers=np.array([2.0]),
            penalty=1.0, inner_result=None)),
        ("ALMDualUpdate", alm.ALMDualUpdate(
            multipliers=np.array([2.0]), update_feasibility_tol=0.1, update_stationarity_tol=0.1,
            multiplier_cap_binding=False, multiplier_cap_binding_indices=[])),
        ("ALMIterateMeasurement", measured),
        ("ALMConstraintRoutingState", routing),
        ("ALMHistoryRecorder", recorder),
        ("ALMProcessBudgetExhausted", alm.ALMProcessBudgetExhausted("budget")),
        ("DEFAULT_CONTINUATION_POLICY", alm_policy.DEFAULT_CONTINUATION_POLICY),
        ("ALMInnerPlan", alm_policy.ALMInnerPlan(profile="boxed", options={"maxiter": 5})),
        ("ALMConverge", alm_policy.ALMConverge("converged", "converged", "done", 0)),
        ("ALMStop", alm_policy.ALMStop("subproblem_limit", "plateau_stall", "stop", 2)),
        ("ALMDualUpdateStep", alm_policy.ALMDualUpdateStep(penalty_reason=None, feasible_stall_count=0)),
        ("ALMRaisePenalty", alm_policy.ALMRaisePenalty("penalty_increase", "max_outer", 0)),
        ("ALMHold", alm_policy.ALMHold(feasible_stall_count=0)),
        ("ALMContinue", alm_policy.ALMContinue(
            trust_radius=None, update_stationarity_tol=0.1, feasible_stall_count=0)),
    )


class AlmPublicIdentityTests(unittest.TestCase):
    def test_every_public_class_and_function_names_its_facade(self):
        objects = list(_public_objects())
        self.assertGreater(len(objects), 40)
        for name, value, facade in objects:
            with self.subTest(name=name):
                self.assertEqual(value.__module__, facade)
                data = pickle.dumps(value)
                self.assertNotIn(b"_src", data)
                self.assertEqual(_pickled_simsopt_alm_modules(data), {facade})
                self.assertIs(pickle.loads(data), value)

    def test_a_class_keeps_its_defining_module_and_its_introspection(self):
        for name, value, facade in _public_objects():
            if not inspect.isclass(value):
                continue
            with self.subTest(name=name):
                source = sys.modules[value._module_source]
                self.assertTrue(value._module_source.startswith("simsopt_alm._src."))
                self.assertIs(getattr(source, name), value)
                typing.get_type_hints(value)
                if dataclasses.is_dataclass(value):
                    self.assertEqual(
                        list(inspect.signature(value.__init__).parameters)[1:],
                        [field.name for field in dataclasses.fields(value) if field.init],
                    )
                self.assertIn(f"in module {facade}", pydoc.render_doc(value))
        self.assertIn("ALMProcessBudgetExhausted", pydoc.render_doc(alm))

    def test_a_function_keeps_its_source(self):
        for name, value, _facade in _public_objects():
            if inspect.isfunction(value):
                with self.subTest(name=name):
                    self.assertIn(f"def {name}(", inspect.getsource(value))
                    self.assertIn("/_src/", Path(inspect.getsourcefile(value)).as_posix())


class AlmPicklePathTests(unittest.TestCase):
    def test_picklable_instances_name_only_the_facades(self):
        facades = {facade.__name__ for facade in FACADES}
        for label, instance in _picklable_instances():
            with self.subTest(instance=label):
                data = pickle.dumps(instance)
                self.assertNotIn(b"_src", data)
                self.assertLessEqual(_pickled_simsopt_alm_modules(data), facades)
                self.assertEqual(_bits(pickle.loads(data)), _bits(instance))

    def test_read_only_instances_do_not_pickle(self):
        result, boundaries, _snapshots, _recorder = _checkpointed_run()
        for label, instance in (
            ("ALMResult", result),
            ("ALMPhysics", _physics(np.zeros(2))),
            ("ALMOuterBoundary with incumbent", boundaries[-1]),
        ):
            with self.subTest(instance=label):
                with self.assertRaisesRegex(TypeError, "mappingproxy"):
                    pickle.dumps(instance)


# A fresh interpreter that imports nothing but pickle: each pickle in DIR loads
# through the module path it names; prints each type's module and the
# simsopt_alm facades the loads imported.
UNPICKLE_PROBE = """\
import json, pickle, sys
from pathlib import Path

loaded = {path.name: type(pickle.loads(path.read_bytes())).__module__
          for path in sorted(Path(sys.argv[1]).glob("*.pkl"))}
facades = sorted(name for name in sys.modules
                 if name.startswith("simsopt_alm") and not name.startswith("simsopt_alm._src"))
print("PROBE " + json.dumps({"loaded": loaded, "facades": facades}))
"""


class AlmFreshInterpreterUnpickleTests(unittest.TestCase):
    def test_pickles_load_through_the_facades_in_a_fresh_interpreter(self):
        result, _boundaries, snapshots, _recorder = _checkpointed_run()
        objects = {
            "settings.pkl": alm.ALMSettings(max_outer_iterations=6),
            "last_iterate.pkl": result.last_iterate,
            "snapshot.pkl": snapshots[0],
            "policy.pkl": alm_policy.DEFAULT_CONTINUATION_POLICY,
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, instance in objects.items():
                (Path(directory) / name).write_bytes(pickle.dumps(instance))
            completed = subprocess.run(
                [sys.executable, "-c", UNPICKLE_PROBE, directory],
                capture_output=True, text=True, check=False,
            )
        lines = [line for line in completed.stdout.splitlines() if line.startswith("PROBE ")]
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(len(lines), 1, completed.stdout)
        report = json.loads(lines[0][len("PROBE "):])
        self.assertEqual(report["loaded"], {
            "last_iterate.pkl": "simsopt_alm",
            "policy.pkl": "simsopt_alm.policy",
            "settings.pkl": "simsopt_alm",
            "snapshot.pkl": "simsopt_alm.checkpoint",
        })
        self.assertEqual(report["facades"], ["simsopt_alm", "simsopt_alm.checkpoint", "simsopt_alm.policy"])


if __name__ == "__main__":
    unittest.main()
