"""Compare this reproduction's checkpoint with the 2026-09 Stage 2 record, against tolerances fixed before the run.

Every numeric leaf of the re-run entries is paired by path with the 2026-09 value. The graded ones are named
below and must agree within their relative tolerance; every other numeric leaf (stationarity residuals, GCI
orders, iteration counts, timings) is reported but not graded. Run from the repo root:

    python results/stage2-repro-1.10.0/compare.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "stage2" / "checkpoint.json"
NEW = HERE / "checkpoint.json"

# The re-run entries, by checkpoint path.
ENTRIES = {
    "verify_R_out_2": ("verify", "R_out_10"),
    "dlaw_100": ("dlaw", "100"),
    "dlaw_1000": ("dlaw", "1000"),
    "tiltlaw_100": ("tiltlaw", "100"),
    "ramp_200_square": ("ramp_excess", "Ha200_square"),
    "case_p_correction": ("case_p_correction",),
}

# Fixed before the run (stage_2_plan.txt 5D W1). The solve tolerance is 1e-9, so a fully developed quantity
# should repeat to well inside 1e-6; the 3-D excess is a small difference of two pressure drops, so 1e-3 of itself.
FULLY_DEVELOPED_RTOL = 1e-6
EXCESS_RTOL = 1e-3
GRADED = {
    # optimum shape and its objective
    "beta_star": FULLY_DEVELOPED_RTOL,
    "w_star": FULLY_DEVELOPED_RTOL,
    "value_star": FULLY_DEVELOPED_RTOL,
    "W_at_fv_beta": FULLY_DEVELOPED_RTOL,
    "u_multiplier": FULLY_DEVELOPED_RTOL,
    "s_star": FULLY_DEVELOPED_RTOL,
    "Ha_star": FULLY_DEVELOPED_RTOL,
    "dp_star": FULLY_DEVELOPED_RTOL,
    "dp_square": FULLY_DEVELOPED_RTOL,
    "dp_at_aligned_beta": FULLY_DEVELOPED_RTOL,
    "reduction": FULLY_DEVELOPED_RTOL,
    "beta_aligned": FULLY_DEVELOPED_RTOL,
    "kappa": FULLY_DEVELOPED_RTOL,
    # Case P's exact 1/R field
    "q_uniform": FULLY_DEVELOPED_RTOL,
    "q_exact": FULLY_DEVELOPED_RTOL,
    "flow_ratio": FULLY_DEVELOPED_RTOL,
    "flow_centroid_fraction": FULLY_DEVELOPED_RTOL,
    # the open-axis 3-D excess
    "excess_percent": EXCESS_RTOL,
}


def leaves(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, (*path, key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from leaves(value, (*path, i))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, float(node)


def rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0.0 else abs(a - b) / scale


def entry(ckpt: dict, path: tuple):
    for key in path:
        ckpt = ckpt[key]
    return ckpt


def main() -> int:
    old, new = json.loads(OLD.read_text()), json.loads(NEW.read_text())
    report, failed = {}, 0
    for stage, path in ENTRIES.items():
        a = dict(leaves(entry(old, path)))
        b = dict(leaves(entry(new, path)))
        graded, other = [], []
        for leaf in sorted(a.keys() & b.keys(), key=str):
            row = {
                "path": "/".join(map(str, leaf)),
                "old": a[leaf],
                "new": b[leaf],
                "rel": rel(a[leaf], b[leaf]),
            }
            name = next((k for k in reversed(leaf) if isinstance(k, str)), "")
            if name in GRADED:
                row.update(rtol=GRADED[name], ok=row["rel"] <= GRADED[name])
                graded.append(row)
            else:
                other.append(row)
        misses = [r for r in graded if not r["ok"]]
        failed += len(misses)
        worst = max(graded, key=lambda r: r["rel"] / r["rtol"])
        report[stage] = {
            "graded": len(graded),
            "misses": misses,
            "worst_graded": worst,
            "worst_other": max(other, key=lambda r: r["rel"], default=None),
            "only_old": sorted("/".join(map(str, k)) for k in a.keys() - b.keys()),
            "only_new": sorted("/".join(map(str, k)) for k in b.keys() - a.keys()),
            "rows": graded + other,
        }
        print(
            f"{stage:18s} {'PASS' if not misses else 'FAIL'}  graded {len(graded):3d}, misses {len(misses)}; "
            f"worst graded {worst['rel']:.1e} ({worst['path']}, tol {worst['rtol']:.0e})"
        )
        if report[stage]["worst_other"]:
            w = report[stage]["worst_other"]
            print(
                f"{'':18s}       worst ungraded {w['rel']:.1e} ({w['path']}: {w['old']:.6g} -> {w['new']:.6g})"
            )
        if report[stage]["only_old"] or report[stage]["only_new"]:
            print(f"{'':18s}       unpaired: old {report[stage]['only_old']} new {report[stage]['only_new']}")
    (HERE / "compare.json").write_text(json.dumps(report, indent=2))
    print(f"{'ALL PASS' if not failed else f'{failed} MISS(ES)'}; details in {HERE.name}/compare.json")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
