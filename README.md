# LMhdX Duct Optimization

Optimization studies of liquid-metal blanket ducts, run on
[LMhdX](https://github.com/uwplasma/LMhdX), the laminar inductionless MHD solver in JAX.
This repository holds the study code, the raw results, the figures and tables, and a write-up
for each study, so the solver repository stays free of them. Its commits are the record of the
actual runs: each results commit states what ran, against which tolerances, and what came out.

## Studies

| Study | Question | Headline | Write-up |
|:--|:--|:--|:--|
| Stage 2 (LMhdX plan step 3b.3) | The fixed-flow, minimum-pumping-power shape of one straight, insulated PbLi duct in a tokamak field | The optimum aspect ratio follows β\*√<i>Ha</i>\* → 2.08 from <i>Ha</i>\* 14 to 6,109, saving 9–73% over an equal-area square duct; a tilted field moves it along κ = √<i>Ha</i>\*<sub>0</sub> · <i>B</i><sub>p</sub>/<i>B</i><sub>T</sub> | [results/stage2](results/stage2/README.md) |

## Layout

```text
ductopt/                   the study package: evaluator, optimizer, design law, validity checks
duct_optimization_poc.py   the checkpointed driver: one stage per process, --list / --stage / --status / --finalize
duct_opt_figures.py        figures and tables from results.json alone (NumPy and Matplotlib only)
duct_opt_probes/           exploratory probes behind the study's plan (P1-P11, cost and tilt probes)
results/<study>/           results.json, checkpoint.json, figures/, tables/, captions.md, the write-up
```

## Running

The code needs LMhdX 1.9.0 or later (the module layout of `lmhdx.grid`,
`lmhdx.fully_developed` and `lmhdx.cases`), installed, with its checkout on `PYTHONPATH`, because
the spectral reference lives in the checkout's `validation/shercliff.py` rather than in the
installed package.

```bash
git clone https://github.com/uwplasma/LMhdX.git
cd LMhdX && uv pip install -e '.[dev]' 'solvax==0.19.0'
```

Stage 2 ran on LMhdX 1.6.0 with Python 3.10.21, JAX 0.6.2 and SOLVAX 0.19.0, on a CPU. From this
repository:

```bash
export PYTHONPATH=.:/path/to/LMhdX
python -u duct_optimization_poc.py --list          # every stage and whether it is done
python -u duct_optimization_poc.py --stage <name>  # run one stage; it checkpoints when it finishes
python -u duct_optimization_poc.py --finalize      # assemble results/stage2/results.json
python duct_opt_figures.py                         # figures, tables and captions.md
```

Each stage runs as its own process and writes its result to `results/stage2/checkpoint.json` as
it finishes, so an interrupted run loses at most the stage in flight. The figure script measures
nothing; it reads only `results.json`.

The code was ported from LMhdX's pre-1.9.0 module names (`lmx`, `lmhdx.bc`, `lmhdx.design`,
`lmhdx.specs`) with the same import changes LMhdX made to its own copy (LMhdX ea93172). The
probes in `duct_opt_probes/` are kept as the record of what was measured; their imports were
ported, but they have not been run again since.

## Provenance

`results.json` records the code that produced it in `meta`. Stage 2 ran while the study code still
lived in LMhdX, so its `git_sha` (d665151) is an LMhdX commit on `stage2/port-fix`. From this
repository on, `git_sha` is this repository's commit and `lmhdx_git_sha` and `lmhdx_version` name
the solver.

The study's history up to 2026-09-29 was carried over from LMhdX's `stage2/port-fix` branch
(`examples/scratch/`) with `git subtree split`, so those commits have new hashes here. Their
messages cite the original hashes:

| LMhdX | Here | Commit |
|:--|:--|:--|
| 1472e95 | a41700f | ductopt package, the fully developed duct evaluator and optimizer |
| 8ff19b2 | 44913c9 | checkpointed driver for the Tier 0 run |
| 86e2f05 | 7062185 | figure and table script for the Tier 0 results |
| 684177d | 3cd454c | probes P1-P11 and the 5A.A cost probe |
| 0c928fb | 89aeb61 | 5A.B: open-axis ramp study, code and pre-registered tolerances |
| 526ef09 | 7eb25cd | 5A.B: open-axis validity results |
| b4affd3 | 21ce85d | 5A.C: verification and full-scope stages, with tolerances and predictions |
| 97023f2 | 286aa5d | 5A.C: fix the beta cache key; add the h/10 and Richardson readings |
| b990705 | 5445910 | 5A.C: O4 set-up before it runs |
| 12f0d26 | fbb8d94 | 5A.C: full-scope results, re-optimization on three meshes, reference, O4, tilt |
| 82b344f | 9607a20 | 5A.C: figure fixes; landscapes re-run on the wider V range |
| d665151 | 5bcf267 | tilt-aware design law: study code and tolerances, before its runs |
| 25e03c4 | 15214d8 | tilt-aware design law: results, F9, T10, tilted spectral reference |

## License

MIT, as for LMhdX, from which this code was carried over. See [LICENSE](LICENSE).
