"""Figures of the Stage 2 reproduction, from compare.json and the two checkpoints alone, in Stage 2's house style.

R1: every graded value's relative difference from the 2026-09 record, by stage, against its tolerance. R2: the
re-run design-law optima on the 2026-09 design law. Run from the repo root (NumPy and Matplotlib only):

    python results/stage2-repro-1.10.0/make_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from duct_opt_figures import COLOR, INK, INK_MUTED, SINGLE_COL, axis_label  # noqa: E402

FIGURES = HERE / "figures"
FLOOR = 1e-16  # where an exact repeat (relative difference 0) is drawn
STAGES = {
    "verify_R_out_2": "Default optimum,\nthree meshes",
    "dlaw_100": "Design law,\nH = 100",
    "dlaw_1000": "Design law,\nH = 1000",
    "tiltlaw_100": "Tilt law,\nH = 100",
    "ramp_200_square": "Open-axis\nexcess, Ha 200",
    "case_p_correction": "Case P's\n1/R field",
}
CAPTIONS: list[str] = []


def save(fig, name: str, rows: list[dict], caption: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)
    with (FIGURES / f"{name}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    CAPTIONS.append(f"**{name}.** {caption}")


def figure_differences(compare: dict) -> None:
    fig, ax = plt.subplots(figsize=(1.9 * SINGLE_COL, 2.8))
    rows, labels = [], []
    for i, (stage, label) in enumerate(STAGES.items()):
        graded = [r for r in compare[stage]["rows"] if "rtol" in r]
        rel = np.array([r["rel"] for r in graded])
        moved = rel[rel > 0.0]
        jitter = np.linspace(-0.2, 0.2, len(moved)) if len(moved) > 1 else np.zeros(len(moved))
        ax.plot(
            i + jitter, moved, "o", color=COLOR["R_outboard"], markersize=3.5, mec="white", mew=0.4, zorder=3
        )
        exact = int(np.sum(rel == 0.0))
        if exact:  # one marker for every exact repeat of the stage, on the floor
            ax.plot(i, FLOOR, "s", color=COLOR["R_outboard"], markersize=4.5, mec="white", mew=0.4, zorder=3)
        tolerance = graded[0]["rtol"]
        ax.plot([i - 0.32, i + 0.32], [tolerance] * 2, color=INK, lw=1.2, solid_capstyle="round", zorder=2)
        counts = f"{len(rel)} graded, all exact" if exact == len(rel) else f"{len(rel)} graded, {exact} exact"
        labels.append(f"{label}\n{counts}")
        rows += [{"stage": stage, "path": r["path"], "old": r["old"], "new": r["new"], "rel": r["rel"],
                  "rtol": r["rtol"]} for r in graded]  # fmt: skip
    ax.axhline(FLOOR, color=INK_MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 10, 10.0)
    ax.set_xticks(range(len(STAGES)), labels, fontsize=6)
    ax.set_xlim(-0.6, len(STAGES) - 0.4)
    ax.set_ylabel(axis_label("Relative difference from 2026-09"))
    ax.grid(axis="x", visible=False)
    ax.plot([], [], color=INK, lw=1.2, label="Tolerance, fixed before the run")
    ax.plot([], [], "o", color=COLOR["R_outboard"], markersize=3.5, label="A graded value that moved")
    ax.plot([], [], "s", color=COLOR["R_outboard"], markersize=4.5, label="Every exact repeat (difference 0)")
    ax.legend(loc="upper left", frameon=False, ncol=3)
    save(
        fig,
        "R1_differences",
        rows,
        "Every graded value of the six re-run stages against the 2026-09 Stage 2 record, as a relative difference, with "
        "each stage's tolerance (1e-6 for fully developed quantities, 1e-3 for the open-axis excess). The square on the "
        "dashed floor stands for all of a stage's exact repeats; the counts are under each stage.",
    )


def figure_design_law(old: dict, new: dict) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    explore = [
        next(m for m in d["meshes"] if (m["cells"], m["cells_in_layer"]) == (48, 6))
        for d in old["dlaw"].values()
    ]
    law = sorted((m["Ha_star"], m["beta_star"]) for m in explore)
    ax.plot(*zip(*law), "-", color=INK_MUTED, lw=1.0, zorder=1, label="2026-09 design law, 48/6")
    rows = []
    for H in ("100", "1000"):
        for before, after in zip(old["dlaw"][H]["meshes"], new["dlaw"][H]["meshes"]):
            ax.plot(
                before["Ha_star"],
                before["beta_star"],
                "o",
                mfc="none",
                mec=INK,
                markersize=8,
                mew=1.0,
                zorder=2,
            )
            ax.plot(after["Ha_star"], after["beta_star"], "o", color=COLOR["R_outboard"], markersize=3.5, mec="white",
                    mew=0.4, zorder=3)  # fmt: skip
            rows.append({"H": H, "mesh": f"{after['cells']}/{after['cells_in_layer']}", "Ha_star_2026_09": before["Ha_star"], "Ha_star_repro": after["Ha_star"],
                         "beta_star_2026_09": before["beta_star"], "beta_star_repro": after["beta_star"]})  # fmt: skip
    ax.plot([], [], "o", mfc="none", mec=INK, markersize=8, label="2026-09, each re-run mesh")
    ax.plot([], [], "o", color=COLOR["R_outboard"], markersize=3.5, label="LMhdX 1.10.0 re-run")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(axis_label("Optimal Hartmann number", "Ha^*"))
    ax.set_ylabel(axis_label("Optimal aspect ratio", r"\beta^*"))
    ax.legend(loc="upper right", frameon=False)
    save(
        fig,
        "R2_design_law",
        rows,
        "The re-run design-law optima at H = 100 (three meshes) and 1000 (two), inside the 2026-09 values they "
        "reproduce bit for bit, on the 2026-09 design law (48/6).",
    )


def main() -> None:
    compare = json.loads((HERE / "compare.json").read_text())
    old = json.loads((HERE.parent / "stage2" / "checkpoint.json").read_text())
    new = json.loads((HERE / "checkpoint.json").read_text())
    figure_differences(compare)
    figure_design_law(old, new)
    (FIGURES / "captions.md").write_text("\n\n".join(CAPTIONS) + "\n")


if __name__ == "__main__":
    main()
