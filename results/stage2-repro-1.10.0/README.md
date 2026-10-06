# Stage 2 reproduction on LMhdX 1.10.0

A check that the Stage 2 results ([results/stage2](../stage2/README.md), run in 2026-09 on LMhdX 1.6.0)
still come out the same on LMhdX main after its 1.9.0 module layout, the merge of the study code
([uwplasma/LMhdX#216](https://github.com/uwplasma/LMhdX/pull/216)) and the open-axis Newton work
([#218](https://github.com/uwplasma/LMhdX/pull/218), [#219](https://github.com/uwplasma/LMhdX/pull/219)).
It reruns a representative subset of six stages. The results are unchanged.

**Run:** 2026-10-06 · **LMhdX** 1.10.0 (95aa3b5, clean) · **Python** 3.10.21 · **JAX** 0.6.2 ·
**SOLVAX** 0.19.0 · CPU, WSL2. These are the same Python, JAX and SOLVAX versions as 2026-09; only
LMhdX changed.

## What ran

| Stage | What it checks |
|:--|:--|
| `verify_R_out_2` | The default Case R optimum (V_min 10 mm/s) on three meshes, with the spectral reference |
| `dlaw_100` | The design law at H = 100 (Ha\* 285) on three meshes |
| `dlaw_1000` | The design law at H = 1000 (Ha\* 6,109), the highest point |
| `tiltlaw_100` | The tilt law at H = 100, six tilts on three meshes, built on the fresh `dlaw_100` |
| `ramp_200_square` | The open-axis 3-D excess at Ha 200, the code #218 and #219 touched |
| `case_p_correction` | Case P's exact 1/R field |

The checkpoint here started as the 2026-09 one with these six entries removed, so they ran fresh,
and `verify_R_out_2` and `case_p_correction` read the same 2026-09 Pareto rows. Each stage ran
as its own process; its output is in [logs/](logs).

## Tolerances

These were fixed and committed before the run (b358fd6), in [compare.py](compare.py):

- Every fully developed quantity must agree within 1e-6 relative of the 2026-09 value. This
  covers β\*, W\*, s\*, Ha\*, the pressure drops, the reductions, κ, and Case P's flow ratios and
  centroids. The solve tolerance is 1e-9.
- The open-axis 3-D excess must agree within 1e-3 of itself, since it is a small difference of
  two pressure drops.

Every other numeric field is reported but not graded: stationarity residuals, GCI orders,
iteration counts and timings.

## Result: all pass

| Stage | Graded values | Worst graded difference | Tolerance |
|:--|--:|--:|--:|
| `verify_R_out_2` | 16 | 0 (bit-identical) | 1e-6 |
| `dlaw_100` | 32 | 0 (bit-identical) | 1e-6 |
| `dlaw_1000` | 32 | 0 (bit-identical) | 1e-6 |
| `tiltlaw_100` | 169 | 0 (bit-identical) | 1e-6 |
| `ramp_200_square` | 11 | 4.6e-9 (excess) | 1e-3 |
| `case_p_correction` | 8 | 5.1e-10 (square-duct centroid) | 1e-6 |

The four fully developed stages are identical to the last bit: every numeric field, graded or
not, apart from timings. For example, at H = 100 β\* is 0.12417, 0.12336 and
0.12300 on the three meshes against 0.12271 by the reference. At H = 1000, Ha\* is 6,109 and the
reduction 72.9%. The default optimum's W\* on the middle mesh is 4.14948e-5, as in 2026-09.

Case P's 3-D solve with the exact 1/R field differs only at round-off. The flow ratios agree to
3e-14. The flow centroids are about 3e-4 of the half-width and agree to 5e-10 of themselves,
which is about 2e-13 absolute. The uniform-field solves are bit-identical.

On the open axis, the pressure drops agree within 1.7e-12 and the excess within 4.6e-9. Charge
and mass balance stay at round-off (1e-14 and 1e-16). What changed is the work. The CG solves
took 245–257 iterations, where 2026-09 needed 368–735, and the stage took 329 s instead of
492 s. That is a solver speed-up on main, not a change in the answer. The likely cause is
[LMhdX #188](https://github.com/uwplasma/LMhdX/pull/188) (e15f1f1, "one smaller damping rate for varying fields"), which changed the preconditioner
for spatially varying fields like the ramp and came after the 1.6.0 run. Both runs stop below the same
certified residual, 1e-9.

Wall times on the other stages moved by −18% to +45%: `verify_R_out_2` took 336 s against
251 s, and `dlaw_100` 130 s against 90 s. Timings depend on the machine's load, are not graded,
and don't enter any result. They are kept in `checkpoint.json` under `stage_cost`.

## Files

| Path | What it holds |
|:--|:--|
| `checkpoint.json` | The 2026-09 checkpoint with the six re-run entries replaced by this run's, and their stage costs |
| `compare.py` | The comparison and its tolerances (run from the repo root) |
| `compare.json` | Every paired value: path, 2026-09, now, relative difference, and grade |
| `logs/` | Each stage's output |

To repeat it from the repo root:

```bash
export DUCTOPT_RESULTS=results/stage2-repro-1.10.0 PYTHONPATH=.:/path/to/LMhdX
python -u duct_optimization_poc.py --stage dlaw_100   # then each stage above, dlaw_100 before tiltlaw_100
python results/stage2-repro-1.10.0/compare.py
```
