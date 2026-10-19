"""Stage 3's figures, from results/stage3-thermal/results.json, in Stage 2's style (duct_opt_figures.py).

F1: the cost of each shape's cheapest feasible flow over (b/a, A): pumping power per channel for the fixed duty,
per metre of first wall for the fixed flux, with the infeasible shapes hatched by the limit that rules them out; the thermal optimum and Stage 2's
design are marked. F2: at the Stage 2 design's pumping power for 1.5 kW, the duty each aspect removes at the
area cap.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "stage3-thermal"
FIGURES = RESULTS / "figures"

# Stage 2's palette and ink (duct_opt_figures.py; validated 2026-09-24 with the dataviz validator).
SEQ_BLUE = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#0d366b"])
INK, INK_SECONDARY, INK_MUTED, GRID_COLOR = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#ffffff"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.dpi": 200,
    }
)
LABELS = {"duty": "Fixed duty, 1.5 kW per channel", "flux": "Fixed flux, 96.8 kW/m² on the first wall"}
COST_LABELS = {"duty": "Pumping power per channel (W)", "flux": "Pumping power per metre of first wall (W/m)"}


def _edges(values: list[float]) -> np.ndarray:
    """Cell edges for a log-spaced sample, half a step beyond each end."""
    logs = np.log(values)
    step = logs[1] - logs[0]
    return np.exp(np.concatenate([logs - step / 2, [logs[-1] + step / 2]]))


def figure_landscape(results: dict) -> None:
    landscape = results["landscape"]
    betas, areas = landscape["beta"], landscape["A"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True, constrained_layout=True)
    u2, w2 = results["stage2_design"]["u"], results["stage2_design"]["w"]
    for ax, framing in zip(axes, ("duty", "flux")):
        grid = landscape[framing]
        W = np.array([[c["W"] if c["feasible"] else np.nan for c in row] for row in grid])
        norm = LogNorm(vmin=np.nanmin(W), vmax=np.nanmax(W))  # each framing has its own unit
        mesh = ax.pcolormesh(_edges(areas) * 1e4, _edges(betas), W, cmap=SEQ_BLUE, norm=norm, shading="flat")
        colorbar = fig.colorbar(mesh, ax=ax, shrink=0.9, pad=0.02)
        colorbar.set_label(COST_LABELS[framing])
        for limit, hatch in (("temperature", "////"), ("Re/Ha", "\\\\\\\\")):
            mask = np.array([[(not c["feasible"]) and c["limit"] == limit for c in row] for row in grid])
            ax.pcolor(
                _edges(areas) * 1e4,
                _edges(betas),
                np.ma.masked_where(~mask, np.ones_like(W)),
                hatch=hatch,
                facecolor="none",
                edgecolor=INK_MUTED,
                linewidth=0.0,
            )
        optimum = results["optimum"][framing]
        star, ring = (optimum["A"] * 1e4, optimum["beta"]), (np.exp(u2) * 1e4, np.exp(w2))
        ax.plot(*ring, "o", mfc=SURFACE, mec=INK, markersize=11, mew=1.4, zorder=3)
        ax.plot(*star, "*", color=INK, markersize=8, mec=SURFACE, mew=0.6, zorder=5)
        if abs(np.log(star[0] / ring[0])) < 0.1 and abs(np.log(star[1] / ring[1])) < 0.1:
            ax.annotate("thermal optimum (★) at\nStage 2's design (○)", ring, xytext=(-10, 0),
                        textcoords="offset points", ha="right", va="center", color=INK, fontsize=7)  # fmt: skip
        else:
            ax.annotate("thermal optimum", star, xytext=(-9, 0), textcoords="offset points", ha="right",
                        va="center", color=INK, fontsize=7)  # fmt: skip
            ax.annotate("Stage 2 design", ring, xytext=(-10, 0), textcoords="offset points", ha="right",
                        va="center", color=INK, fontsize=7)  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(LABELS[framing], fontsize=8, color=INK, loc="left")
        ax.set_xlabel("Area A (cm²)")
    axes[0].set_ylabel("Aspect b/a")
    fig.legend(
        handles=[
            Patch(facecolor="none", edgecolor=INK_MUTED, hatch="////", label="Infeasible: temperature"),
            Patch(facecolor="none", edgecolor=INK_MUTED, hatch="\\\\\\\\", label="Infeasible: Re/Ha > 200"),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    fig.savefig(FIGURES / "F1_landscape.png")
    plt.close(fig)


def figure_capacity(results: dict) -> None:
    capacity = results["capacity"]
    rows = capacity["rows"]
    beta = np.array([r["beta"] for r in rows])
    duty = np.array([r["duty"] for r in rows]) / 1e3
    laminar = np.array([r["laminar"] for r in rows])
    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    ax.plot(beta, np.where(laminar, duty, np.nan), color="#2a78d6", lw=2, solid_capstyle="round")
    ax.plot(beta, np.where(~laminar, duty, np.nan), color="#2a78d6", lw=2, ls=(0, (2, 2)))
    ax.axhline(1.5, color=INK_MUTED, lw=0.8)
    ax.annotate(
        "1.5 kW duty",
        (beta[0], 1.5),
        xytext=(2, 3),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=7,
    )
    stage2 = results["stage2_design"]["beta"]
    ax.axvline(stage2, color=INK_MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.annotate("Stage 2's b/a", (stage2, 0.0), xycoords=("data", "axes fraction"), xytext=(4, 6),
                textcoords="offset points", color=INK_SECONDARY, fontsize=7)  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("Aspect b/a, at the area cap 2.4 cm²")
    ax.set_ylabel("Duty removed within 625 K (kW)")
    ax.set_title(
        f"At the Stage 2 design's pumping power, {capacity['power_W'] * 1e3:.2f} mW",
        fontsize=8,
        loc="left",
        color=INK,
    )
    ax.grid(True, axis="y")
    if (~laminar).any():
        ax.annotate("dashed: Re/Ha > 200", (0.98, 0.04), xycoords="axes fraction", ha="right",
                    color=INK_SECONDARY, fontsize=7)  # fmt: skip
    fig.savefig(FIGURES / "F2_capacity.png")
    plt.close(fig)


def main() -> None:
    results = json.loads((RESULTS / "results.json").read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_landscape(results)
    figure_capacity(results)


if __name__ == "__main__":
    main()
