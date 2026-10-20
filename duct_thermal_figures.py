"""Stage 3's figures, in Stage 2's house style (duct_opt_figures.py), from results/stage3-thermal/results.json and
the ungraded figure data of duct_thermal_figure_data.py; measures nothing itself.

Per figure, a vector PDF, a 300-dpi PNG and a CSV twin of the plotted data; the movie is an animated WebP with a
PNG of its last frame. Captions go to figures/captions.md. Run with (NumPy, Matplotlib and Pillow only):
    python duct_thermal_figures.py
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

from duct_opt_figures import (  # noqa: E402
    COLOR,
    DOUBLE_COL,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SEQ_BLUE,
    SINGLE_COL,
    _panel_letter,
    axis_label,
)

RESULTS = Path(__file__).resolve().parent / "results" / "stage3-thermal"
FIGURES = RESULTS / "figures"
SURFACE = "#ffffff"
# Temperature, a magnitude: one warm hue, light to dark (the dark end is Stage 2's diverging red).
SEQ_HEAT = LinearSegmentedColormap.from_list("seq_heat", ["#fbe3d6", "#8a1414"])
FRAMING = {
    "duty": {"title": "Fixed duty, 1.5 kW per channel", "color": COLOR["R_outboard"], "marker": "o"},
    "flux": {"title": "Fixed flux, 96.8 kW/m² on the first wall", "color": COLOR["R_inboard"], "marker": "s"},
}
COST = {
    "duty": ("Pumping power per channel", "W_c", "mW", 1e3),
    "flux": ("Pumping power per metre\nof first wall", "W_w", "W/m", 1.0),
}
CAPTIONS: list[str] = []


def save(fig, name: str, rows: list[dict], caption: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)
    if rows:
        with (FIGURES / f"{name}.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    CAPTIONS.append(f"**{name}.** {caption}")
    print(f"  wrote {name}")


def _edges(values) -> np.ndarray:
    logs = np.log(values)
    step = logs[1] - logs[0]
    return np.exp(np.concatenate([logs - step / 2, [logs[-1] + step / 2]]))


def _mark_designs(ax, star: tuple[float, float], ring: tuple[float, float]) -> None:
    """The thermal optimum (star) and Stage 2's design (ring), labelled once each, or together where they meet."""
    ax.plot(*ring, "o", mfc=SURFACE, mec=INK, markersize=10, mew=1.3, zorder=4)
    ax.plot(*star, "*", color=INK, markersize=8, mec=SURFACE, mew=0.6, zorder=5)
    style = dict(textcoords="offset points", ha="right", va="center", color=INK, fontsize=7)
    if abs(np.log(star[0] / ring[0])) < 0.1 and abs(np.log(star[1] / ring[1])) < 0.1:
        ax.annotate("thermal optimum (★) at\nStage 2's design (○)", ring, xytext=(-10, 0), **style)
    else:
        ax.annotate("thermal optimum", star, xytext=(-9, 0), **style)
        ax.annotate("Stage 2 design", ring, xytext=(-10, 0), **style)


def figure_landscape(results: dict) -> None:
    landscape = results["landscape"]
    betas, areas = landscape["beta"], landscape["A"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0), sharey=True)
    s2 = results["stage2_design"]
    rows = []
    for ax, framing, letter in zip(axes, ("duty", "flux"), "ab"):
        grid = landscape[framing]
        name, symbol, unit, scale = COST[framing]
        W = np.array([[c["W"] * scale if c["feasible"] else np.nan for c in row] for row in grid])
        mesh = ax.pcolormesh(_edges(areas) * 1e4, _edges(betas), W, cmap=SEQ_BLUE,
                             norm=LogNorm(np.nanmin(W), np.nanmax(W)), shading="flat")  # fmt: skip
        for limit, hatch in (("temperature", "////"), ("Re/Ha", "\\\\\\\\")):
            mask = np.array([[(not c["feasible"]) and c["limit"] == limit for c in row] for row in grid])
            ax.pcolor(_edges(areas) * 1e4, _edges(betas), np.ma.masked_where(~mask, np.ones_like(W)), hatch=hatch,
                      facecolor="none", edgecolor=INK_MUTED, linewidth=0.0)  # fmt: skip
        optimum = results["optimum"][framing]
        _mark_designs(ax, (optimum["A"] * 1e4, optimum["beta"]), (s2["A"] * 1e4, s2["beta"]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(False)
        ax.set_title(FRAMING[framing]["title"], loc="left")
        ax.set_xlabel(axis_label("Area", "A", "cm²"))
        fig.colorbar(mesh, ax=ax, shrink=0.9, pad=0.02).set_label(axis_label(name, symbol, unit))
        _panel_letter(ax, letter, dx=-0.16)
        for i, beta in enumerate(betas):
            for j, area in enumerate(areas):
                c = grid[i][j]
                rows.append({"framing": framing, "beta": beta, "A_m2": area, "feasible": c["feasible"],
                             "limit": c["limit"], "cost": c["W"], "pumping_power_W": c["pumping_power"]})  # fmt: skip
    axes[0].set_ylabel(axis_label("Aspect ratio", r"\beta = b/a"))
    fig.legend(handles=[Patch(facecolor="none", edgecolor=INK_MUTED, hatch="////", label="Infeasible: temperature"),
                        Patch(facecolor="none", edgecolor=INK_MUTED, hatch="\\\\\\\\", label="Infeasible: Re/Ha > 200")],
               loc="outside lower center", ncol=2, frameon=False)  # fmt: skip
    save(fig, "F1_landscape", rows,
         "The cost of each shape's cheapest feasible flow over area and aspect ratio (13 x 13 samples, 48/6 cells): "
         "pumping power per channel at a fixed duty (a), per metre of first wall at a fixed flux (b). Hatched shapes "
         "are infeasible: no flow keeps the peaks within 625 K, or the one that does breaks Re/Ha <= 200. The star is "
         "the thermal optimum, the ring Stage 2's design.")  # fmt: skip


def figure_cap(results: dict, data: dict) -> None:
    cap = data["cap"]
    beta = np.array(cap["beta"])
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))
    s2 = results["stage2_design"]
    rows = []
    for ax, framing, letter in zip(axes, ("duty", "flux"), "ab"):
        name, symbol, unit, scale = COST[framing]
        cost = np.array([r["W"] * scale if r["feasible"] else np.nan for r in cap[framing]])
        ax.plot(beta, cost, color=FRAMING[framing]["color"], lw=1.6, solid_capstyle="round")
        optimum = results["optimum"][framing]
        ax.plot(
            s2["beta"], s2[framing]["W"] * scale, "o", mfc=SURFACE, mec=INK, markersize=8, mew=1.2, zorder=4
        )
        ax.plot(
            optimum["beta"],
            optimum["W"] * scale,
            "*",
            color=INK,
            markersize=9,
            mec=SURFACE,
            mew=0.6,
            zorder=5,
        )
        ratio = s2[framing]["W_over_optimum"]
        ax.annotate(f"Stage 2's design, ×{ratio:.3f}" if ratio < 1.01 else f"Stage 2's design, ×{ratio:.2f}",
                    (s2["beta"], s2[framing]["W"] * scale), xytext=(8, 10), textcoords="offset points",
                    fontsize=7, color=INK)  # fmt: skip
        ax.annotate(f"optimum, b/a {optimum['beta']:.3f}", (optimum["beta"], optimum["W"] * scale), xytext=(6, -12),
                    textcoords="offset points", fontsize=7, color=INK)  # fmt: skip
        edges = _edges(beta)
        infeasible = ~np.isfinite(cost)
        i = 0
        while i < len(beta):  # shade each infeasible run and name its limits
            if not infeasible[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(beta) and infeasible[j + 1]:
                j += 1
            ax.axvspan(edges[i], edges[j + 1], color=INK_MUTED, alpha=0.12, lw=0)
            limits = sorted({r["limit"] for r in cap[framing][i : j + 1]})
            ax.annotate("infeasible: " + ", ".join(limits), (np.sqrt(edges[i] * edges[j + 1]), 0.5),
                        xycoords=("data", "axes fraction"), ha="center", va="center", rotation=90, fontsize=6,
                        color=INK_SECONDARY)  # fmt: skip
            i = j + 1
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(beta[0] / 1.1, beta[-1] * 1.1)
        ax.set_title(FRAMING[framing]["title"], loc="left")
        ax.set_xlabel(axis_label("Aspect ratio at the area cap", r"\beta = b/a"))
        ax.set_ylabel(axis_label(name, symbol, unit))
        _panel_letter(ax, letter, dx=-0.24)
        rows += [{"framing": framing, "beta": b, "feasible": r["feasible"], "limit": r["limit"], "cost": r["W"],
                  "pumping_power_W": r["pumping_power"], "Q_m3s": r["Q"]} for b, r in zip(beta, cap[framing])]  # fmt: skip
    save(fig, "F2_cap", rows,
         "The cost along the area cap, where both optima lie, against the aspect ratio (29 samples, 48/6 cells). "
         "At a fixed duty (a) Stage 2's design sits at the minimum; at a fixed flux (b) the minimum is squarer, and "
         "Stage 2's flatter design, whose wider wall collects more heat, costs 21 % more per metre of first wall.")  # fmt: skip


def figure_designs(results: dict, arrays) -> None:
    names = {("duty", "stage2"): "Stage 2's design", ("duty", "optimum"): "Thermal optimum",
             ("flux", "stage2"): "Stage 2's design", ("flux", "optimum"): "Thermal optimum"}  # fmt: skip
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 3.4), sharex=True, sharey=True)
    rows = []
    for r, framing in enumerate(("duty", "flux")):
        fields = {k: arrays[f"field_{framing}_{k}"] for k in ("stage2", "optimum")}
        norm = Normalize(min(f.min() for f in fields.values()), 625.0)
        for c, kind in enumerate(("stage2", "optimum")):
            ax = axes[r, c]
            y, z = arrays[f"field_{framing}_{kind}_y"] * 1e3, arrays[f"field_{framing}_{kind}_z"] * 1e3
            mesh = ax.pcolormesh(
                y, z, fields[kind].T, cmap=SEQ_HEAT, norm=norm, shading="flat", rasterized=True
            )
            ax.plot([y[0], y[-1]], [z[0], z[0]], color=INK, lw=2.2, solid_capstyle="butt")
            ax.set_aspect("equal")
            ax.set_ylim(-4.8, 4.8)  # a margin, so the heated wall's line shows on the tallest duct too
            ax.grid(False)
            design = (
                results["stage2_design"][framing]
                if kind == "stage2"
                else results["optimum"][framing]["design"]
            )
            size = f"{2e3 * design['a']:.1f} × {2e3 * design['b']:.1f} mm"
            ax.set_title(f"{names[framing, kind]}: b/a {design['beta']:.3f}, {size}", loc="left")
            rows.append({"framing": framing, "design": kind, "beta": design["beta"], "a_mm": design["a"] * 1e3,
                         "b_mm": design["b"] * 1e3, "outlet_min_K": float(fields[kind].min()),
                         "outlet_max_K": float(fields[kind].max()), "power_W": design["power"]})  # fmt: skip
        fig.colorbar(mesh, ax=axes[r, :], shrink=0.95, pad=0.02).set_label(
            axis_label("Outlet temperature", "T", "K")
        )
        axes[r, 0].set_ylabel(FRAMING[framing]["title"].split(",")[0] + "\n" + axis_label("z", "", "mm"))
    for ax in axes[1]:
        ax.set_xlabel(axis_label("y, along the field B", "", "mm"))
    save(fig, "F3_designs", rows,
         "Each design's outlet cross-section, drawn to scale, from the developed duct at its cheapest feasible flow "
         "(48/6 cells). The heated wall z = -b (black) faces the plasma; the field B runs along y. Every design's "
         "wall peak is 625 K by construction. At a fixed duty the two designs nearly coincide; at a fixed flux the "
         "optimum is narrower along the field, with about twice Stage 2's aspect ratio.")  # fmt: skip


def figure_capacity(results: dict) -> None:
    capacity = results["capacity"]
    rows = capacity["rows"]
    beta = np.array([r["beta"] for r in rows])
    duty = np.array([r["duty"] for r in rows]) / 1e3
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
    ax.plot(beta, duty, color=COLOR["R_outboard"], lw=1.6, solid_capstyle="round")
    ax.axhline(1.5, color=INK_MUTED, lw=0.8)
    ax.annotate(
        "1.5 kW duty",
        (beta[0], 1.5),
        xytext=(2, -10),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=7,
    )
    stage2 = results["stage2_design"]["beta"]
    ax.axvline(stage2, color=INK_MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.annotate("Stage 2's b/a", (stage2, 0.0), xycoords=("data", "axes fraction"), xytext=(4, 6),
                textcoords="offset points", color=INK_SECONDARY, fontsize=7)  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel(axis_label("Aspect ratio at the area cap", r"\beta = b/a"))
    ax.set_ylabel(axis_label("Duty removed within 625 K", "P", "kW"))
    ax.set_title(f"At Stage 2's pumping power, {capacity['power_W'] * 1e3:.2f} mW", loc="left")
    save(fig, "F4_capacity", rows,
         "At the Stage 2 design's pumping power for 1.5 kW (2.20 mW), the duty each aspect ratio removes within 625 K "
         "at the area cap. It peaks near Stage 2's aspect ratio; a square duct removes 0.90 kW. Every sample is "
         "laminar (Re/Ha <= 200).")  # fmt: skip


def figure_verification(results: dict) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))
    rows = []
    for framing in ("duty", "flux"):
        t = results["optimum"][framing]["taylor_test"]
        h = np.array(t["h"])
        style = FRAMING[framing]
        ax.loglog(
            h,
            t["remainder_zeroth"],
            ls=(0, (3, 2)),
            color=style["color"],
            marker=style["marker"],
            markersize=3.5,
        )
        ax.loglog(h, t["remainder_first"], color=style["color"], marker=style["marker"], markersize=3.5,
                  label=f"{framing}: slopes {t['slope_zeroth']:.2f}, {t['slope_first']:.2f}")  # fmt: skip
        rows += [{"panel": "a", "framing": framing, "h": hh, "remainder_zeroth": z, "remainder_first": f,
                  "quantity": None, "change_pct": None, "gate_pct": None}
                 for hh, z, f in zip(h, t["remainder_zeroth"], t["remainder_first"])]  # fmt: skip
    ax.set_xlabel(axis_label("Step in ln A", "h"))
    ax.set_ylabel(axis_label("Taylor remainder of the cost"))
    ax.legend(
        frameon=False, title="dashed: |W(u+h) - W|, solid: minus h dW/du", title_fontsize=6, loc="lower right"
    )
    _panel_letter(ax, "a")
    labels, values, gates = [], [], []
    for framing in ("duty", "flux"):
        v = results["verify"][framing]
        for key, gate, label in (("W_rel", 0.01, "cost"), ("beta_rel", 0.02, "b/a")):
            labels.append(f"{label},\n{framing}")
            values.append(100 * v[key])
            gates.append(100 * gate)
            rows.append({"panel": "b", "framing": framing, "h": None, "remainder_zeroth": None, "remainder_first": None,
                         "quantity": label, "change_pct": 100 * v[key], "gate_pct": 100 * gate})  # fmt: skip
    x = np.arange(len(values))
    bx.bar(x, values, width=0.5, color=[FRAMING[f]["color"] for f in ("duty", "duty", "flux", "flux")])
    for xi, gate in zip(x, gates):
        bx.plot([xi - 0.35, xi + 0.35], [gate, gate], color=INK, lw=1.2)
    for xi, value in zip(x, values):
        bx.annotate(
            f"{value:.2f} %", (xi, value), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7
        )
    bx.set_xticks(x, labels)
    bx.set_ylabel(axis_label("Change from 48/6 to 72/9", "", "%"))
    bx.set_ylim(0, 2.3)
    bx.annotate("gates (black): 1 % in cost, 2 % in b/a", (0.02, 0.95), xycoords="axes fraction", fontsize=6,
                color=INK_SECONDARY, va="top")  # fmt: skip
    bx.grid(axis="x", visible=False)
    _panel_letter(bx, "b")
    save(fig, "F5_verification", rows,
         "(a) G3's Taylor test of dW/du at each optimum: the remainder falls at first order without the gradient and "
         "at second order with it. (b) G4: each optimum re-optimized on 72/9 cells against 48/6, the change in its "
         "cost and aspect ratio, against the gates.")  # fmt: skip


def movie_open_duct(arrays) -> None:
    """M1: each optimum's open duct heating from 573 K, frame by frame: the mean temperature along the duct and
    the outlet cross-section, to scale."""
    import warnings

    warnings.filterwarnings("ignore", message="There are no gridspecs with layoutgrids")  # the fixed layout's
    framings = ("duty", "flux")
    times = {f: arrays[f"open_{f}_time"] for f in framings}
    count = min(len(t) for t in times.values())
    norm = Normalize(573.0, 625.0)
    # Fixed rectangles, so the frames do not jitter; the flat cross-sections keep their width (equal aspect taken
    # up by the data limits, not by shrinking the box).
    fig = plt.figure(figsize=(DOUBLE_COL, 3.4), dpi=150, layout="none")
    ax_x = fig.add_axes([0.08, 0.15, 0.36, 0.72])
    ax_s = [fig.add_axes([0.52, 0.60, 0.36, 0.25]), fig.add_axes([0.52, 0.17, 0.36, 0.25])]
    frames, rows = [], []
    for k in range(count):
        ax_x.clear()
        for f in framings:
            x = arrays[f"open_{f}_x"]
            ax_x.plot(x, arrays[f"open_{f}_mean_along_x"][-1], color=FRAMING[f]["color"], lw=0.8, alpha=0.35)
            ax_x.plot(x, arrays[f"open_{f}_mean_along_x"][k], color=FRAMING[f]["color"], lw=1.6,
                      label=f"{f} optimum")  # fmt: skip
        ax_x.set_ylim(570, 626)
        ax_x.set_xlim(0, 1.0)
        ax_x.set_xlabel(axis_label("Distance from the inlet", "x", "m"))
        ax_x.set_ylabel(axis_label("Mean temperature", r"\bar{T}", "K"))
        ax_x.legend(loc="upper left", frameon=False)
        residence = arrays["open_duty_residence_time"]
        ax_x.set_title(f"t = {times['duty'][k]:.0f} s, {times['duty'][k] / residence:.1f} duty residence times "
                       "(faint: steady)", loc="left")  # fmt: skip
        for ax, f in zip(ax_s, framings):
            ax.clear()
            y, z = arrays[f"open_{f}_y_faces"] * 1e3, arrays[f"open_{f}_z_faces"] * 1e3
            ax.pcolormesh(y, z, arrays[f"open_{f}_outlet"][k].T, cmap=SEQ_HEAT, norm=norm, shading="flat")
            ax.plot([y[0], y[-1]], [z[0], z[0]], color=INK, lw=2.0)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlim(-21, 21)
            ax.set_ylabel(axis_label("z", "", "mm"))
            ax.grid(False)
            ax.set_title(
                f"Outlet, {f} optimum: hottest {arrays[f'open_{f}_outlet'][k].max():.1f} K", loc="left"
            )
        ax_s[1].set_xlabel(axis_label("y, along B", "", "mm"))
        if k == 0:
            colorbar_axes = fig.add_axes([0.91, 0.17, 0.015, 0.68])
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ_HEAT), cax=colorbar_axes).set_label(
                axis_label("Temperature", "T", "K")
            )
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150)
        frames.append(Image.open(buffer).convert("RGB"))
    fig.savefig(FIGURES / "M1_open_duct_final.png", dpi=300)
    plt.close(fig)
    durations = [140] * (len(frames) - 1) + [1500]
    frames[0].save(FIGURES / "M1_open_duct.webp", save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, quality=80, method=6)  # fmt: skip
    for f in framings:
        for k, t in enumerate(times[f][:count]):
            rows.append({"framing": f, "time_s": float(t), "outlet_hottest_K": float(arrays[f"open_{f}_outlet"][k].max()),
                         "outlet_mean_K": float(arrays[f"open_{f}_mean_along_x"][k][-1])})  # fmt: skip
    with (FIGURES / "M1_open_duct.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    CAPTIONS.append("**M1_open_duct.** Each optimum's held flow down an open duct from a 573 K inlet (24 axial cells; "
                    "ungraded figure data, G5's check on 24 cells instead of 12), from the uniform inlet temperature to "
                    "steady state over six residence times: the mean temperature along the duct, and the outlet "
                    "cross-section to scale.")  # fmt: skip
    print("  wrote M1_open_duct")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for stale in ("F2_capacity.png",):  # renamed to F4_capacity
        (FIGURES / stale).unlink(missing_ok=True)
    results = json.loads((RESULTS / "results.json").read_text())
    data = json.loads((RESULTS / "figure_data.json").read_text())
    arrays = np.load(RESULTS / "figure_data.npz")
    figure_landscape(results)
    figure_cap(results, data)
    figure_designs(results, arrays)
    figure_capacity(results)
    figure_verification(results)
    movie_open_duct(arrays)
    (FIGURES / "captions.md").write_text("\n\n".join(CAPTIONS) + "\n")


if __name__ == "__main__":
    main()
