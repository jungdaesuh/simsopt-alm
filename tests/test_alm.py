import ast
import dataclasses
import inspect
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import List

import numpy as np

import simsopt_alm as alm
from simsopt_alm._src import alm_math
from simsopt_alm._src.loop import outer as alm_outer
from simsopt_alm._src.settings import settings as alm_settings


PACKAGE_ROOT = Path(alm.__file__).resolve().parent
# The one module that imports simsopt (for its ``Derivative``); importing
# ``simsopt_alm`` does not load it.
SIMSOPT_MODULE_FILE = "signed_constraints.py"
SOLVER_MODULE_PATHS = tuple(
    path for path in sorted(PACKAGE_ROOT.rglob("*.py")) if path.name != SIMSOPT_MODULE_FILE
)
# The implementation (``simsopt_alm._src``), by path below the package; the
# public facades (``__init__``, ``policy``, ``checkpoint``, ``history``) only
# re-export it.
LIBRARY_MODULE_FILES = tuple(
    path.relative_to(PACKAGE_ROOT).as_posix()
    for path in sorted((PACKAGE_ROOT / "_src").rglob("*.py"))
    if path.name != "__init__.py"
)


def _parse_library_module(filename: str) -> ast.Module:
    return ast.parse((PACKAGE_ROOT / filename).read_text(encoding="utf-8"))


def _top_level_import_bindings(tree: ast.Module) -> List[str]:
    """Names the module's imports bind (``from __future__`` binds none)."""
    return [
        (alias.asname or alias.name).split(".")[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        for alias in node.names
    ]


# The solver's only third-party dependencies are numpy and scipy; every other
# import is the standard library (or relative, inside the package).
ALLOWED_IMPORT_ROOTS = frozenset(
    (
        "__future__",
        "dataclasses",
        "enum",
        "functools",
        "numbers",
        "pickle",
        "types",
        "typing",
        "warnings",
        "numpy",
        "scipy",
    )
)


class AlmPackageImportTests(unittest.TestCase):
    def test_package_imports_only_numpy_scipy_and_the_standard_library(self):
        for path in SOLVER_MODULE_PATHS:
            with self.subTest(module=path.name):
                roots = set()
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.Import):
                        roots.update(alias.name.split(".")[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        roots.add(node.module.split(".")[0])
                self.assertEqual(
                    sorted(roots - ALLOWED_IMPORT_ROOTS),
                    [],
                    f"{path.name} imports outside numpy, scipy and the standard library",
                )

    def test_package_imports_in_a_fresh_interpreter(self):
        # Fresh interpreter: other tests in this process may already have
        # imported a module the package forgot to import itself.
        probe = (
            "import simsopt_alm as alm\n"
            "alm.minimize_alm, alm.ALMSettings, alm.augmented_inequality_objective\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"simsopt_alm must import standalone: {completed.stdout}{completed.stderr}",
        )

    def test_importing_the_solver_loads_no_simsopt_module(self):
        probe = (
            "import json, sys\n"
            "import simsopt_alm, simsopt_alm.checkpoint, simsopt_alm.history, simsopt_alm.policy\n"
            "print(json.dumps(sorted(name for name in sys.modules\n"
            "                        if name == 'simsopt' or name.startswith('simsopt.'))))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(completed.stdout.strip(), "[]")


def _solve_unit_halfspace_problem():
    """min ||x||^2 s.t. 1 - x0 <= 0; the solution is x = (1, 0), lambda = 2."""

    def evaluate_problem(x, multipliers, penalty):
        return alm.augmented_inequality_objective(
            base_value=float(x @ x),
            base_grad=2.0 * x,
            constraint_values=np.array([1.0 - x[0]]),
            constraint_grads=[np.array([-1.0, 0.0])],
            multipliers=multipliers,
            penalty=penalty,
        )

    return alm.minimize_alm(
        np.array([3.0, 2.0]),
        ["x0_at_least_one"],
        evaluate_problem,
        alm.ALMSettings(),
        {"maxiter": 200},
    )


class AlmEvaluatorContractTests(unittest.TestCase):
    def test_minimal_evaluator_solves_and_returns_a_frozen_dataclass(self):
        # An evaluator that returns only augmented_inequality_objective(...)
        # output (no optional keys, no hybrid quartet) is a complete problem,
        # and minimize_alm answers it with a frozen dataclass.
        result = _solve_unit_halfspace_problem()

        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, [1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(result.multipliers, [2.0], rtol=1e-4)
        self.assertTrue(
            dataclasses.is_dataclass(result),
            f"minimize_alm returned {type(result).__name__}, not a dataclass",
        )
        self.assertTrue(type(result).__dataclass_params__.frozen)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.success = False


class AlmPublicApiTests(unittest.TestCase):
    def test_public_api_is_the_solver_surface(self):
        self.assertEqual(
            sorted(alm.__all__),
            [
                "ALMConstraintRoutingState",
                "ALMConstraintSignalState",
                "ALMDualUpdate",
                "ALMEvaluation",
                "ALMEvaluator",
                "ALMFeasibleIncumbent",
                "ALMInnerSolveOutcome",
                "ALMIterateMeasurement",
                "ALMLastIterate",
                "ALMLoopState",
                "ALMOuterBoundary",
                "ALMOuterStepEvent",
                "ALMPhysics",
                "ALMResult",
                "ALMSettings",
                "CachedALMEvaluator",
                "alm_problem_physics",
                "augmented_inequality_objective",
                "cached_alm_evaluator",
                "evaluate_alm_problem",
                "minimize_alm",
                "normalize_alm_constraints",
                "run_directional_taylor_test",
                "signed_lower_bound",
                "signed_upper_bound",
            ],
        )
        for name in alm.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(alm, name))

    def test_import_binds_the_policy_facade_and_no_internal_module(self):
        # 0.1.0's ``import simsopt_alm`` bound eleven submodules. ``policy`` is
        # public: it stays bound (the policy facade, its 0.1.0 names included).
        # core, control, inner, continuation, events, boundary, evaluation,
        # hybrid, problem and taylor were internal and are gone. A fresh
        # interpreter: this process may have imported the facade itself.
        probe = (
            "import json, types, simsopt_alm\n"
            "policy = simsopt_alm.policy\n"
            "policy.DefaultContinuationPolicy, policy.ALMContinuationPolicy\n"
            "assert simsopt_alm.minimize_alm.__defaults__[-1] is policy.DEFAULT_CONTINUATION_POLICY\n"
            "print(json.dumps(sorted(name for name, value in vars(simsopt_alm).items()\n"
            "                        if isinstance(value, types.ModuleType))))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=False
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.strip(), '["_src", "policy"]')


class AlmModuleOwnershipTests(unittest.TestCase):
    def test_the_math_module_owns_the_math(self):
        for module, name in (
            (alm_math, "augmented_inequality_objective"),
            (alm_math, "_constraint_routing_state"),
            (alm_math, "_updated_nonnegative_multipliers"),
            (alm_settings, "ALMSettings"),
        ):
            with self.subTest(name=name):
                # The imported package's own file: an installed (non-editable)
                # package lives outside this checkout.
                self.assertEqual(
                    Path(inspect.getsourcefile(getattr(module, name))).resolve(),
                    Path(module.__file__).resolve(),
                )

    def test_the_outer_loop_module_owns_the_loop(self):
        for name in ("minimize_alm", "_run_alm_continuation_step"):
            with self.subTest(name=name):
                self.assertEqual(
                    Path(inspect.getsourcefile(getattr(alm_outer, name))).resolve(),
                    Path(alm_outer.__file__).resolve(),
                )
        self.assertFalse(hasattr(alm_outer.minimize_alm, "__wrapped__"))
        self.assertIs(alm.minimize_alm, alm_outer.minimize_alm)


class AlmLibraryBoundaryTests(unittest.TestCase):
    def test_alm_library_modules_have_no_unused_imports(self):
        # A library module that imports names it never reads couples itself
        # to modules (and vocabulary) it does not need.
        for filename in LIBRARY_MODULE_FILES:
            with self.subTest(module=filename):
                tree = _parse_library_module(filename)
                read_names = {
                    node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
                }
                unused = sorted(
                    name
                    for name in _top_level_import_bindings(tree)
                    if name not in read_names
                )
                self.assertEqual(
                    unused,
                    [],
                    f"{filename} imports {len(unused)} names it never uses",
                )

    def test_loop_imports_the_policy_core_but_no_plugin(self):
        # The loop consults the continuation policy; the hybrid rules are
        # reached only through the default policy, and the history and
        # checkpoint plugins are attached by callers. The loop also reads the
        # modules that own the boundaries, the shared state, the read-only
        # copy rule, the evaluation contract and the inner solve.
        imported = {
            "." * node.level + node.module
            for node in _parse_library_module("_src/loop/outer.py").body
            if isinstance(node, ast.ImportFrom) and node.level > 0
        }
        self.assertEqual(
            imported,
            {"..alm_math", "..checkpoint.boundary", "..inner.attempts", "..policy.decisions",
             "..policy.default", "..policy.views", "..problem.evaluation", "..settings.settings",
             "..state", "..state_views"},
        )

    def test_only_the_facades_and_the_loop_import_the_loop(self):
        # The shared state and its read-only copy rule (state_views) sit
        # beside the components; the problem, policy, inner-solve, checkpoint
        # and settings components never depend on the loop.
        offenders = []
        for filename in LIBRARY_MODULE_FILES:
            if filename.startswith("_src/loop/"):
                continue
            for node in _parse_library_module(filename).body:
                if isinstance(node, ast.ImportFrom) and node.level > 0 and (node.module or "").split(".")[0] == "loop":
                    offenders.append(f"{filename}: from {'.' * node.level}{node.module}")
        self.assertEqual(offenders, [])

    def test_alm_library_is_coil_agnostic(self):
        # No identifier or string literal may name an application: a coil
        # family, a device configuration, or an application-only constraint.
        application_word = re.compile(
            r"stage[-_ ]?2|banana|vacuum|boozer|keepout|hardware", re.IGNORECASE
        )
        for filename in LIBRARY_MODULE_FILES:
            with self.subTest(module=filename):
                offenders = set()
                for node in ast.walk(_parse_library_module(filename)):
                    if isinstance(node, ast.Name):
                        texts = (node.id,)
                    elif isinstance(node, ast.Attribute):
                        texts = (node.attr,)
                    elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        texts = (node.name,)
                    elif isinstance(node, ast.arg):
                        texts = (node.arg,)
                    elif isinstance(node, ast.keyword):
                        texts = (node.arg or "",)
                    elif isinstance(node, ast.alias):
                        texts = (node.name, node.asname or "")
                    elif isinstance(node, ast.Constant) and isinstance(
                        node.value, str
                    ):
                        texts = (node.value,)
                    else:
                        continue
                    offenders.update(
                        text for text in texts if application_word.search(text)
                    )
                self.assertEqual(sorted(offenders), [])


class AlmLibrarySizeBudgetTests(unittest.TestCase):
    """Line-count ratchets at the measured sizes. Lower a budget in the change
    that earns it; raising one needs a reason in the change that does."""

    def _assert_line_budget(self, filename: str, budget: int) -> None:
        line_count = len(
            (PACKAGE_ROOT / filename).read_text(encoding="utf-8").splitlines()
        )
        self.assertLessEqual(
            line_count,
            budget,
            f"{filename} has {line_count} lines; budget is {budget}",
        )

    def _assert_definition_line_budget(
        self, filename: str, name: str, budget: int
    ) -> None:
        tree = _parse_library_module(filename)
        definition = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
        )
        line_count = definition.end_lineno - definition.lineno + 1
        self.assertLessEqual(
            line_count,
            budget,
            f"{filename}:{name} has {line_count} lines; budget is {budget}",
        )

    def test_module_line_budgets(self):
        # Re-pointed by the 0.2.0 layout at the measured sizes, each below the
        # 0.1.0 budget of the module its code came from (core.py 822,
        # continuation.py 225, events.py 339, control.py 1606, checkpoint.py
        # 601, boundary.py 290, evaluation.py 382, history.py 660, hybrid.py
        # 69, inner.py 768, policy.py 396, taylor.py 201); codec.py and
        # snapshot.py split checkpoint.py's 601 lines between them.
        budgets = {
            "_src/alm_math.py": 681,
            "_src/checkpoint/boundary.py": 286,
            "_src/checkpoint/codec.py": 200,
            "_src/checkpoint/snapshot.py": 401,
            "_src/inner/attempts.py": 763,
            "_src/loop/history.py": 657,
            "_src/loop/outer.py": 1591,
            "_src/policy/decisions.py": 82,
            "_src/policy/default.py": 380,
            "_src/policy/hybrid.py": 55,
            "_src/policy/plan.py": 15,
            "_src/policy/views.py": 86,
            "_src/problem/evaluation.py": 382,
            "_src/problem/taylor.py": 201,
            "_src/settings/settings.py": 132,
            "_src/state.py": 233,
            "_src/state_views.py": 195,
        }
        for filename, budget in budgets.items():
            with self.subTest(module=filename):
                self._assert_line_budget(filename, budget)

    def test_definition_line_budgets(self):
        budgets = (
            ("_src/loop/outer.py", "_run_alm_continuation_step", 238),
            ("_src/loop/outer.py", "_execute_step_decision", 174),
            ("_src/loop/outer.py", "minimize_alm", 206),
            ("_src/checkpoint/snapshot.py", "ALMTransitionSnapshot", 132),
        )
        for filename, name, budget in budgets:
            with self.subTest(definition=f"{filename}:{name}"):
                self._assert_definition_line_budget(filename, name, budget)

    def test_alm_core_package_line_budget(self):
        # The core package: math, settings, shared state, loop, continuation
        # vocabulary, default policy, hybrid rules, inner solve, boundaries
        # and problem API: every implementation module but the opt-in history
        # and checkpoint plugins (and the package __init__ docstrings).
        plugins = {"_src/loop/history.py", "_src/checkpoint/snapshot.py", "_src/checkpoint/codec.py"}
        core_files = [filename for filename in LIBRARY_MODULE_FILES if filename not in plugins]
        self.assertEqual(len(core_files), len(LIBRARY_MODULE_FILES) - len(plugins))
        line_count = sum(
            len((PACKAGE_ROOT / filename).read_text(encoding="utf-8").splitlines())
            for filename in core_files
        )
        self.assertLessEqual(
            line_count,
            5292,
            f"the core package has {line_count} lines; budget is 5292",
        )


if __name__ == "__main__":
    unittest.main()
