---
date: 2026-11-18
problem: Reduce L-BFGS-B workspace while retaining solver trajectories and checkpoint semantics.
tags: [alm, lbfgsb, memory, checkpoint]
---

# Size correction history after the iteration budget is finalized

An inner solve can add at most one correction pair per accepted iteration. SciPy reserves workspace using the requested pair count before the first step. The Stage 2 example requested room for 300 pairs even when its call budget allowed only 12 iterations.

Reducing the example's configured history globally would also limit longer solves. Applying the cap at the final SciPy dispatch lets each policy, retry and resumed call work with its own budget. Preserve the configured history in the caller's options and checkpoints. Publish the effective value in the event for the attempt that used it.

Check the complete sequence of physics trial coordinates, the returned point and its diagnostics. In the shipped Stage 2 geometry, 12- and 50-iteration traces matched bit for bit. Peak allocations traced during each solve fell by about 8 MiB. Process peak includes native geometry, libraries and other allocations, so measure it separately.

A saved event can legitimately change when its effective options change. Expand any changed event digest and compare fields before regenerating a fixture. For the inner-budget scenario the only changes were two newly reported history sizes, 5 and 4. The focused regression exercises the actual SciPy dispatch. See [the test](../../tests/test_alm_logic_regressions.py) and [the solver](../../src/simsopt_alm/_src/inner/attempts.py).
