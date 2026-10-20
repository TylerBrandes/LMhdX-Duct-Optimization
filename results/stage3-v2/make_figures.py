"""Figures of Stage 3 v2, from its results.json alone, in Stage 2's house style; measures nothing itself.

V1: the open duct's peaks and delivered share against the axial mesh, with the extrapolation and its band. V2: the
heated wall's face temperature along the duct, against the developed screening's. V3: each framing's optimum and
Stage 2's design from the common feasible flow to twice it. Per figure a vector PDF, a 300-dpi PNG and a CSV twin;
captions go to figures/captions.md. Run from the repo root (NumPy and Matplotlib only):

    python results/stage3-v2/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import duct_thermal_figures as v1  # noqa: E402
from duct_opt_figures import (  # noqa: E402
    DOUBLE_COL,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    _panel_letter,
    axis_label,
)

v1.FIGURES = HERE / "figures"  # v1's save() writes here; its CAPTIONS collect this study's captions
FRAMINGS = ("duty", "flux")
LIMIT = 625.0
SHAPE = {
    "stage3": {"label": "thermal optimum", "marker": "*", "ls": "-"},
    "stage2": {"label": "Stage 2's design", "marker": "o", "ls": (0, (4, 2))},
}


def _limit(ax) -> None:
    ax.axhline(LIMIT, color=INK, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.annotate("625 K limit", (0.99, LIMIT), xycoords=("axes fraction", "data"), xytext=(0, 2),
                textcoords="offset points", ha="right", va="bottom", fontsize=6, color=INK_SECONDARY)  # fmt: skip


def figure_refinement(results: dict) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.7))
    rows = []
    for framing in FRAMINGS:
        r, style = results["refine"][framing], v1.FRAMING[framing]
        cells = sorted(int(n) for n in r["levels"])
        h = np.array([1.0 / n for n in cells])
        for name, ls, label in (("wall_face_peak_K", "-", "hottest wall face"),
                                ("hottest_cell_K", (0, (3, 2)), "hottest cell")):  # fmt: skip
            values = [r["levels"][str(n)][name] for n in cells]
            ax.plot(h, values, ls=ls, color=style["color"], marker=style["marker"], markersize=3.5,
                    label=f"{framing}: {label}")  # fmt: skip
            ext = r["extrapolated"][name]
            ax.errorbar(0.0, ext["value"], yerr=ext["band"], color=style["color"], marker=style["marker"],
                        mfc="white", mec=style["color"], markersize=4.5, capsize=2.5, lw=0.9, ls="none")  # fmt: skip
            rows += [{"panel": "a", "framing": framing, "quantity": name, "axial_cells": n, "value": v}
                     for n, v in zip(cells, values)]  # fmt: skip
            rows.append({"panel": "a", "framing": framing, "quantity": f"{name} extrapolated",
                         "axial_cells": None, "value": ext["value"]})  # fmt: skip
        undelivered = [100.0 * (1.0 - r["levels"][str(n)]["delivered"]) for n in cells]
        bx.plot(h, undelivered, color=style["color"], marker=style["marker"], markersize=3.5, label=framing)
        ext = r["extrapolated"]["delivered"]
        bx.errorbar(0.0, 100.0 * (1.0 - ext["value"]), yerr=100.0 * ext["band"], color=style["color"],
                    marker=style["marker"], mfc="white", mec=style["color"], markersize=4.5, capsize=2.5, lw=0.9,
                    ls="none")  # fmt: skip
        rows += [{"panel": "b", "framing": framing, "quantity": "undelivered_pct", "axial_cells": n, "value": v}
                 for n, v in zip(cells, undelivered)]  # fmt: skip
    _limit(ax)
    for axis in (ax, bx):
        axis.set_xticks([0.0, 1 / 48, 1 / 24, 1 / 12], ["extrap.", "48", "24", "12"])
        axis.set_xlim(-0.006, 1 / 12 + 0.006)
        axis.set_xlabel(axis_label("Axial cells over the 1 m duct (spacing to scale)"))
    ax.set_ylabel(axis_label("Peak temperature", "", "K"))
    ax.set_ylim(624.25, 625.06)
    ax.legend(frameon=False, fontsize=6, loc="lower center", ncol=2)
    bx.set_ylabel(axis_label("Heat not delivered", "", "% of the load"))
    bx.legend(frameon=False, loc="upper left")
    bx.set_ylim(bottom=0.0)
    _panel_letter(ax, "a")
    _panel_letter(bx, "b")
    v1.save(fig, "V1_refinement", rows,
            "The open duct on 12, 24 and 48 axial cells, each optimum's held flow from a 573 K inlet. (a) The hottest "
            "face of any wall (solid) and the hottest cell (dashed), with the extrapolated value and its band at the "
            "left (open markers). (b) The share of the load conducted back out of the inlet rather than delivered at "
            "the outlet. The cross-section is 48/6 throughout.")  # fmt: skip


def figure_along(results: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.2), sharex=True, height_ratios=(1.6, 1.0))
    rows = []
    for column, framing in enumerate(FRAMINGS):
        r, style = results["refine"][framing], v1.FRAMING[framing]
        top, bottom = axes[0, column], axes[1, column]
        bulk_rise = r["developed_bulk_out_K"] - 573.0
        film = r["developed_wall_peak_K"] - r["developed_bulk_out_K"]
        cells = sorted(int(n) for n in r["levels"])
        shades = np.linspace(0.35, 1.0, len(cells))
        for n, shade in zip(cells, shades):
            level = r["levels"][str(n)]
            centres, face = np.array(level["x_centres"]), np.array(level["heated_face_along_K"])
            # Upwind, a cell carries the temperature of its outlet face (exactly so for a linear rise), so each
            # cell is drawn there; at its centre it would sit gamma h / 2 above the screening.
            x = centres + 0.5 * (centres[1] - centres[0])
            developed = 573.0 + bulk_rise * x + film  # the anchored screening there (L = 1 m)
            color = matplotlib.colors.to_rgba(style["color"], shade)
            top.plot(x, face, color=color, marker=style["marker"], markersize=2.5, lw=1.0, label=f"{n} cells")
            bottom.plot(x, face - developed, color=color, marker=style["marker"], markersize=2.5, lw=1.0)
            rows += [{"framing": framing, "axial_cells": n, "x_m": xi, "heated_face_K": fi, "developed_K": di}
                     for xi, fi, di in zip(x, face, developed)]  # fmt: skip
        line = np.linspace(0.0, 1.0, 50)
        top.plot(line, 573.0 + bulk_rise * line + film, color=INK, lw=0.9, ls=(0, (4, 2)),
                 label="developed screening")  # fmt: skip
        _limit(top)
        bottom.axhline(0.0, color=INK, lw=0.8, ls=(0, (4, 2)))
        top.set_title(style["title"], fontsize=8)
        top.legend(frameon=False, fontsize=6, loc="lower right")
        top.set_ylabel(axis_label("Heated wall's face", "", "K"))
        bottom.set_ylabel(axis_label("Less the screening", "", "K"))
        bottom.set_xlabel(axis_label("Distance from the inlet to each cell's outlet face", "x", "m"))
        _panel_letter(top, "ab"[column])
        _panel_letter(bottom, "cd"[column])
    v1.save(fig, "V2_along", rows,
            "The heated wall's face temperature along the duct, its largest value across the wall in each axial cell, "
            "on 12, 24 and 48 axial cells (light to dark). Each cell is drawn at its outlet face: with upwind "
            "transport a cell carries that face's temperature, exactly so for a linear rise. Top: against the "
            "developed screening, which takes the profile as developed from the inlet (dashed). Bottom: the "
            "difference, the entrance region's margin.")  # fmt: skip


def figure_common_flow(results: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.2), sharex=True)
    rows = []
    for column, framing in enumerate(FRAMINGS):
        record, style = results["common_flow"][framing]["48/6"], v1.FRAMING[framing]
        name, symbol, unit, scale = v1.COST[framing]
        factors = np.array(record["sweep_factors"])
        top, bottom = axes[0, column], axes[1, column]
        for shape in ("stage2", "stage3"):
            sweep, look = record["shapes"][shape]["sweep"], SHAPE[shape]
            cost = np.array([s["W"] for s in sweep]) * scale
            wall = np.array([s["T_wall_peak"] for s in sweep])
            color = style["color"] if shape == "stage3" else INK_MUTED
            top.plot(factors, cost, color=color, ls=look["ls"], label=look["label"])
            top.plot(
                factors[0], cost[0], look["marker"], color=color, markersize=7 if shape == "stage3" else 4.5
            )
            bottom.plot(factors, wall, color=color, ls=look["ls"], label=look["label"])
            bottom.plot(
                factors[0], wall[0], look["marker"], color=color, markersize=7 if shape == "stage3" else 4.5
            )
            rows += [{"framing": framing, "shape": shape, "flow_over_common": f, "cost": c, "cost_unit": unit,
                      "T_wall_peak_K": t} for f, c, t in zip(factors, cost, wall)]  # fmt: skip
        ratio = record["W_ratio_stage2_over_stage3"]
        top.annotate(f"Stage 2 / optimum at the\ncommon flow: {ratio:.4f}", (0.03, 0.95), xycoords="axes fraction",
                     va="top", fontsize=6.5, color=INK_SECONDARY)  # fmt: skip
        _limit(bottom)
        top.set_title(style["title"], fontsize=8)
        top.set_ylabel(axis_label(name.replace("power per channel", "power\nper channel"), symbol, unit))
        bottom.set_ylabel(axis_label("Peak wall temperature", "", "K"))
        flow = "Flow per channel" if framing == "duty" else "Flow per metre of first wall"
        bottom.set_xlabel(axis_label(f"{flow}, over the common flow"))
        top.legend(frameon=False, loc="lower right")
        _panel_letter(top, "ab"[column])
        _panel_letter(bottom, "cd"[column])
    v1.save(fig, "V3_common_flow", rows,
            "Each framing's thermal optimum and Stage 2's design from one common feasible flow (markers) to twice it, "
            "on 48/6: the larger of the two shapes' cheapest feasible flows, per channel for the fixed duty and per "
            "metre of first wall for the fixed flux, so both shapes have the same bulk temperature rise. Top: the "
            "cost, in Stage 3's units. Bottom: the peak wall temperature.")  # fmt: skip


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    v1.FIGURES.mkdir(exist_ok=True)
    figure_refinement(results)
    figure_along(results)
    figure_common_flow(results)
    (v1.FIGURES / "captions.md").write_text("\n\n".join(v1.CAPTIONS) + "\n")


if __name__ == "__main__":
    main()
