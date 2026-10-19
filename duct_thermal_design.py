"""Stage 3: the thermal design of Case P (LMhdX plan step 5.5), as resumable stages.

``python duct_thermal_design.py STAGE`` runs one stage and saves it to the checkpoint; ``all`` runs every
stage the checkpoint lacks, each in this process; ``finalize`` writes results.json. The specification and
the gates are results/stage3-thermal/README.md's, committed before the run (11ea591); they are written
into results.json and not relaxed after it.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("DUCTOPT_RESULTS", str(REPO_ROOT / "results" / "stage3-thermal"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from lmhdx.core3d import ChannelProblem, energy_budget, zero_velocity  # noqa: E402
from lmhdx.fully_developed import channel_flow_rate  # noqa: E402
from lmhdx.grid import (  # noqa: E402
    CENTER,
    DIRICHLET,
    NEUMANN,
    PERIODIC,
    BoundaryCondition,
    Field,
    Grid,
    uniform_faces,
)
from lmhdx.steady import solve_steady_state  # noqa: E402
from lmhdx.thermal import (  # noqa: E402
    ThermalProblem,
    advance_thermal,
    build_thermal_step,
    developed_heat,
    initial_thermal_state,
    load_attribution,
    thermal_boundary,
)

from ductopt import cases, checkpoint  # noqa: E402
from ductopt.optimize import fd_step_study, polish_w, taylor_test  # noqa: E402
from ductopt.physics import _build_problem, mesh_key  # noqa: E402
from ductopt.thermal import (  # noqa: E402
    _dimensionless_thermal,
    film,
    optimize_thermal,
    profile,
    thermal_box,
    thermal_objective,
)

OUT_DIR = checkpoint.RESULTS_DIR
CELLS, LAYER = 48, 6  # Stage 2's optimizer mesh
FINE = (72, 9)  # the next mesh of Stage 2's family (32/4, 48/6, 72/9)
FRAMINGS = ("duty", "flux")
STAGE2_RESULTS = REPO_ROOT / "results" / "stage2" / "results.json"
BETA_EXPANSION = 1.21e-4  # 1/K, Martelli 2019 Table 1, for the reported buoyancy ratio
GRAVITY = 9.81  # m/s^2

# The README's gates, fixed before the run (11ea591).
GATES = {
    "G1_scaling_rel": 1.0e-10,
    "G2_flow_independence_rel": 1.0e-12,
    "G3_taylor_slope_first": (1.9, 2.1),
    # Stage 2's c_stationarity_rel: |dW/dw| / W at the polished optimum on the scaled-mesh family. As in Stage 2's
    # record, the Richardson and h/10 readings are gated; h and 2h carry the h^2 truncation and are reported.
    "G3_stationarity_rel": 1.0e-6,
    "G4_W_rel": 0.01,
    "G4_beta_rel": 0.02,
    "G5_delivered_min": 0.95,
    "G5_peak_max_K": 625.0,
    "G6_re_over_ha_max": 200.0,
    "G6_dissipation_over_duty_max": 1.0e-3,
}


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _box(framing: str):
    return thermal_box(cases.case_p(), framing)


def _stage2_design() -> tuple[float, float]:
    """Stage 2's Case P optimum at V_min 10 mm/s: (u, w), from its record."""
    rows = json.loads(STAGE2_RESULTS.read_text())["case_p"]["pareto"]
    row = next(r for r in rows if r["V_min"] == cases.V_MIN_DEFAULT)
    return float(row["u"]), float(row["w"])


def _physical_flow(box, design: dict, cells: int, layer: int):
    """The design in metres: its grid, its ChannelProblem in PbLi and its velocity at ``Q``."""
    beta_k, ha_mesh_k = mesh_key(design["beta"], box.ha_mesh_for_beta(design["beta"]))
    unit = _build_problem(beta_k, ha_mesh_k, cells, layer)
    a = design["a"]
    grid = Grid(unit.grid.x_faces, a * np.asarray(unit.grid.y_faces), a * np.asarray(unit.grid.z_faces))
    flow = ChannelProblem(
        grid,
        unit.conditions,
        density=box.rho,
        viscosity=box.mu / box.rho,
        conductivity=box.sigma,
        magnetic_field=(0.0, box.B, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
    )
    velocity = solve_steady_state(flow, forcing=(1.0, 0.0, 0.0)).velocity
    scale = design["Q"] / float(channel_flow_rate(flow, velocity[0].data[0]))
    return unit, flow, tuple(c.replace_data(c.data * scale) for c in velocity), scale


def stage_reduction() -> None:
    """G1 and G2 at three aspects of each framing, and the Ha derivative against central differences."""
    rows = []
    for framing in FRAMINGS:
        box = _box(framing)
        for beta in (0.15, 0.5, 2.0):
            design = thermal_objective(box, float(np.log(1.0e-4)), float(np.log(beta)), CELLS, LAYER)
            if not design["feasible"]:
                rows.append({"framing": framing, "beta": beta, "feasible": False, "limit": design["limit"]})
                continue
            unit, flow, velocity, _ = _physical_flow(box, design, CELLS, LAYER)
            insulated = thermal_boundary("heat_flux", conductivity=box.k)
            heated = thermal_boundary("heat_flux", -design["wall_flux"], 0.0, conductivity=box.k)
            thermal = ThermalProblem(
                flow,
                (BoundaryCondition(PERIODIC), insulated, heated),
                box.cp,
                box.k,
                0.0,
                joule_heating=False,
            )
            nothing = Field(jnp.zeros(flow.grid.shape), (CENTER,) * 3, flow.grid)
            heat = developed_heat(thermal, velocity, nothing)
            wall, fluid = (float(t) for t in heat.anchored_peaks(box.t_in, box.L))
            rise = design["T_wall_peak"] - box.t_in
            g1 = max(abs(wall - design["T_wall_peak"]), abs(fluid - design["T_fluid_peak"])) / rise
            # G2: the dimensionless film at the unit-drive flow and at 7.3 times it.
            shape = solve_steady_state(
                unit, forcing=(1.0, 0.0, 0.0), field_scale=design["Ha"] / unit.magnetic_field[1]
            ).velocity
            dimensionless = _dimensionless_thermal(unit)
            nothing_unit = Field(jnp.zeros(unit.grid.shape), (CENTER,) * 3, unit.grid)
            films = []
            for factor in (1.0, 7.3):
                h = developed_heat(
                    dimensionless, tuple(c.replace_data(factor * c.data) for c in shape), nothing_unit
                )
                films.append(np.array([h.peak_wall_temperature - h.bulk_temperature,
                                       jnp.max(h.temperature.data) - h.bulk_temperature]))  # fmt: skip
            g2 = float(np.max(np.abs(films[1] - films[0]) / np.abs(films[0])))
            ha_mesh = box.ha_mesh_for_beta(beta)
            _, d_dha = film(beta, design["Ha"], ha_mesh, CELLS, LAYER)
            step = 1.0e-4 * design["Ha"]
            central = (
                film(beta, design["Ha"] + step, ha_mesh, CELLS, LAYER)[0]
                - film(beta, design["Ha"] - step, ha_mesh, CELLS, LAYER)[0]
            ) / (2.0 * step)
            rows.append(
                {
                    "framing": framing,
                    "beta": beta,
                    "A": 1.0e-4,
                    "feasible": True,
                    "Q": design["Q"],
                    "T_wall_peak_scaled": design["T_wall_peak"],
                    "T_wall_peak_dimensional": wall,
                    "T_fluid_peak_scaled": design["T_fluid_peak"],
                    "T_fluid_peak_dimensional": fluid,
                    "G1_worst_rel": g1,
                    "G2_worst_rel": g2,
                    "d_dHa": d_dha.tolist(),
                    "d_dHa_central": central.tolist(),
                    "d_dHa_rel": (np.abs(d_dha - central) / np.abs(central)).tolist(),
                }  # fmt: skip
            )
            print(f"{framing} b/a {beta}: G1 {g1:.2e}, G2 {g2:.2e}", flush=True)
    checkpoint.set_key(("reduction",), {"rows": rows})


def stage_landscape() -> None:
    """W, Q_min and the binding limit over a 13 x 13 grid of (b/a, A), both framings from one film each."""
    betas, areas = np.geomspace(0.02, 5.0, 13), np.geomspace(2.4e-6, 2.4e-4, 13)
    out = {"beta": betas.tolist(), "A": areas.tolist()}
    t0 = time.perf_counter()
    for framing in FRAMINGS:
        box = _box(framing)
        grid = []
        for beta in betas:
            row = []
            for area in areas:
                r = thermal_objective(box, float(np.log(area)), float(np.log(beta)), CELLS, LAYER)
                row.append(
                    {
                        k: r.get(k)
                        for k in (
                            "feasible",
                            "limit",
                            "W",
                            "pumping_power",
                            "Q",
                            "Re_over_Ha",
                            "film",
                            "theta_wall",
                        )
                    }
                )
            grid.append(row)
        out[framing] = grid
    out["time_s"] = time.perf_counter() - t0
    checkpoint.set_key(("landscape",), out)


def _best_start(framing: str) -> tuple[float, float]:
    landscape = checkpoint.load()["landscape"]
    best = None
    for i, beta in enumerate(landscape["beta"]):
        for j, area in enumerate(landscape["A"]):
            cell = landscape[framing][i][j]
            if cell["feasible"] and (best is None or cell["W"] < best[0]):
                best = (cell["W"], float(np.log(area)), float(np.log(beta)))
    return best[1], best[2]


def _optimum(framing: str, cells: int, layer: int, w0: float) -> dict:
    box = _box(framing)
    t0 = time.perf_counter()
    result = optimize_thermal(box, w0, cells, layer)
    design = result["design"]
    return {
        "cells": [cells, layer], "start_w": w0, "w": result["w"], "beta": result["beta"], "u": design["u"],
        "A": design["A"], "W": design["W"], "converged": result["converged"], "active_u_bound": result["bound"],
        "u_multiplier": result["multiplier"], "u_range": result["u_range"], "rounds": result["rounds"],
        "profiles_evaluated": result["profiles_evaluated"], "time_s": time.perf_counter() - t0, "design": design,
    }  # fmt: skip


def _polish(framing: str, w_star: float) -> dict:
    """G3's stationarity, as Stage 2 measured it: W*(w) on the scaled-mesh family centred at the optimum.

    One mesh, built for the optimum's aspect, has its z faces scaled to each nearby aspect (Stage 2's O12), so
    W*(w) is smooth and ``dW/dw`` is not the mesh-to-mesh noise of a rebuilt mesh.
    """
    box, centre = _box(framing), float(np.exp(w_star))

    def value(w: float) -> float:
        found = profile(box, w, CELLS, LAYER, centre)
        return np.inf if found is None else found["design"]["W"]

    return polish_w(value, w_star)


def stage_optimum(framing: str) -> None:
    """G3: the optimizer from the landscape's cheapest aspect, a Taylor test in u, the polished stationarity in w."""
    result = _optimum(framing, CELLS, LAYER, _best_start(framing)[1])
    box = _box(framing)
    result["taylor_test"] = taylor_test(
        box, result["u"], result["w"], CELLS, LAYER, evaluate=thermal_objective
    )
    result["fd_step_study"] = fd_step_study(
        box, result["u"], result["w"], CELLS, LAYER, evaluate=thermal_objective
    )
    result["polish"] = _polish(framing, result["w"])
    checkpoint.set_key(("optimum", framing), result)


def stage_verify(framing: str) -> None:
    """G4: re-optimized on 72/9 from the 48/6 optimum."""
    coarse = checkpoint.load()["optimum"][framing]
    fine = _optimum(framing, *FINE, coarse["w"])
    fine["W_rel"] = abs(fine["W"] - coarse["W"]) / abs(fine["W"])
    fine["beta_rel"] = abs(fine["beta"] - coarse["beta"]) / fine["beta"]
    checkpoint.set_key(("verify", framing), fine)


def stage_open_duct(framing: str) -> None:
    """G5: the candidate's held flow down an open duct from a 573 K inlet; the delivered share and the peak."""
    box = _box(framing)
    optimum = checkpoint.load()["optimum"][framing]
    design = optimum["design"]
    _, periodic, velocity, _ = _physical_flow(box, design, CELLS, LAYER)
    axial_profile, section = velocity[0].data[0], periodic.grid
    nx = 12
    grid = Grid(uniform_faces(nx, 0.0, box.L), section.y_faces, section.z_faces)
    dt = 0.5 * (box.L / nx) / float(jnp.max(axial_profile))  # Courant 0.5 along the duct
    wall = BoundaryCondition(NEUMANN)
    flow = ChannelProblem(
        grid,
        (BoundaryCondition(DIRICHLET, lower=np.asarray(axial_profile), upper_kind=NEUMANN), wall, wall),
        density=box.rho,
        viscosity=box.mu / box.rho,
        conductivity=box.sigma,
        magnetic_field=(0.0, box.B, 0.0),
        dt=dt,
    )
    inlet = thermal_boundary("temperature", box.t_in, upper_kind="heat_flux", conductivity=box.k)
    insulated = thermal_boundary("heat_flux", conductivity=box.k)
    problem = ThermalProblem(
        flow, (inlet, insulated, insulated), box.cp, box.k, box.t_in, joule_heating=False
    )
    rest = zero_velocity(flow)
    start = initial_thermal_state(
        problem, box.t_in, (rest[0].replace_data(jnp.broadcast_to(axial_profile, rest[0].shape)), *rest[1:])
    )
    step = build_thermal_step(problem, dtype=jnp.float64, hold_flow=True)
    # The wall flux as the source q A / V in the fluid row along z = -b: a face flux enters its cell exactly so.
    data = np.zeros(grid.shape)
    data[:, :, 0] = design["power"] / float(np.sum(grid.cell_volumes()[:, :, 0]))
    load = Field(jnp.asarray(data), (CENTER,) * 3, grid)
    nothing = Field(jnp.zeros(grid.shape), (CENTER,) * 3, grid)
    steps = int(np.ceil(6.0 * design["A"] * box.L / design["Q"] / dt))  # six residence times
    t0 = time.perf_counter()
    credit = load_attribution(step, start, nothing, load, steps)
    history = advance_thermal(step, start, load, steps)[1]
    result = {
        "nx": nx, "dt": dt, "steps": steps, "delivered": float(credit.delivered / credit.imposed),
        "imposed": float(credit.imposed), "conducted": float(credit.conducted), "stored": float(credit.stored),
        "residual": float(credit.residual), "peak_K": float(jnp.max(history.temperature_max)),
        "peak_last_K": float(history.temperature_max[-1]), "time_s": time.perf_counter() - t0,
        "developed_fluid_peak_K": design["T_fluid_peak"], "developed_wall_peak_K": design["T_wall_peak"],
    }  # fmt: skip
    checkpoint.set_key(("open_duct", framing), result)


def stage_neglected(framing: str) -> None:
    """G6: Re/Ha, the Joule and viscous dissipation against the duty, and the buoyancy ratio Ri/N (reported)."""
    box = _box(framing)
    design = checkpoint.load()["optimum"][framing]["design"]
    _, flow, velocity, _ = _physical_flow(box, design, CELLS, LAYER)
    budget = energy_budget(velocity, flow)  # over the 1 m periodic cell: W per metre of duct
    viscous, joule = float(budget.viscous) * box.L, float(budget.joule) * box.L
    u_mean = design["V"]
    ri_over_n = (
        box.rho
        * GRAVITY
        * BETA_EXPANSION
        * design["wall_flux"]
        * design["a"]
        / box.k
        / (box.sigma * u_mean * box.B**2)
    )
    checkpoint.set_key(
        ("neglected", framing),
        {
            "Re_over_Ha": design["Re_over_Ha"],
            "viscous_W": viscous,
            "joule_W": joule,
            "pumping_power": design["pumping_power"],
            "dissipation_sum_over_pumping": (viscous + joule) / design["pumping_power"],
            "viscous_over_duty": viscous / design["power"],
            "joule_over_duty": joule / design["power"],
            "Ri_over_N": ri_over_n,
        },  # fmt: skip
    )


def stage_stage2_design() -> None:
    """Stage 2's design at the same limits: its cheapest feasible flow and power, against each optimum's."""
    u, w = _stage2_design()
    out = {"u": u, "w": w, "A": float(np.exp(u)), "beta": float(np.exp(w))}
    ckpt = checkpoint.load()
    for framing in FRAMINGS:
        design = thermal_objective(_box(framing), u, w, CELLS, LAYER)
        optimum = ckpt["optimum"][framing]
        out[framing] = {
            **design,
            "W_over_optimum": design["W"] / optimum["W"],
            "optimum_W": optimum["W"],
            "pumping_over_optimum": design["pumping_power"] / optimum["design"]["pumping_power"],
        }
    checkpoint.set_key(("stage2_design",), out)


def stage_capacity() -> None:
    """At the Stage 2 design's pumping power for 1.5 kW: the largest duty each aspect removes at the area cap."""
    box = _box("duty")
    reference = checkpoint.load()["stage2_design"]["duty"]
    power0 = reference["pumping_power"]
    rows = []
    for beta in np.geomspace(0.02, 5.0, 25):
        u, w = box.u_hi, float(np.log(beta))
        a = box.a_of(u, w)
        ha = box.B * a * float(np.sqrt(box.sigma / box.mu))
        values, _ = film(float(beta), ha, box.ha_mesh_for_beta(float(beta)), CELLS, LAYER)
        q, theta = float(values[0]), float(max(values[1], values[2]))
        Q = float(np.sqrt(power0 * a**4 * q / (box.mu * box.L)))  # W = mu L Q^2 / (a^4 q)
        allowance = box.t_max - box.t_in
        duty = allowance / (1.0 / (box.rho * box.cp * Q) + theta / (2.0 * box.L * box.k))
        hydraulic = 4.0 * a * beta * a / (a + beta * a)
        re_over_ha = box.rho * (Q / np.exp(u)) * hydraulic / box.mu / ha
        rows.append({"beta": float(beta), "Q": Q, "duty": duty, "duty_over_1500": duty / box.duty,
                     "Re_over_Ha": re_over_ha, "laminar": bool(re_over_ha <= box.laminar)})  # fmt: skip
    checkpoint.set_key(("capacity",), {"power_W": power0, "A": float(np.exp(box.u_hi)), "rows": rows})


STAGES = [
    "reduction",
    "landscape",
    "optimum_duty",
    "optimum_flux",
    "verify_duty",
    "verify_flux",
    "open_duct_duty",
    "open_duct_flux",
    "neglected_duty",
    "neglected_flux",
    "stage2_design",
    "capacity",
]


def run_stage(name: str) -> None:
    for prefix, function in (("optimum_", stage_optimum), ("verify_", stage_verify),
                             ("open_duct_", stage_open_duct), ("neglected_", stage_neglected)):  # fmt: skip
        if name.startswith(prefix):
            return function(name[len(prefix) :])
    {"reduction": stage_reduction, "landscape": stage_landscape, "stage2_design": stage_stage2_design,
     "capacity": stage_capacity}[name]()  # fmt: skip


def _done(ckpt: dict, name: str) -> bool:
    for prefix in ("optimum_", "verify_", "open_duct_", "neglected_"):
        if name.startswith(prefix):
            return name[len(prefix) :] in ckpt.get(prefix.rstrip("_"), {})
    return name in ckpt


def _gates(ckpt: dict) -> dict:
    g = GATES
    out = {
        "G1": max(r["G1_worst_rel"] for r in ckpt["reduction"]["rows"] if r["feasible"]) <= g["G1_scaling_rel"],
        "G2": max(r["G2_worst_rel"] for r in ckpt["reduction"]["rows"] if r["feasible"]) <= g["G2_flow_independence_rel"],
    }  # fmt: skip
    for framing in FRAMINGS:
        o, v = ckpt["optimum"][framing], ckpt["verify"][framing]
        lo, hi = g["G3_taylor_slope_first"]
        polish = o["polish"]
        out[f"G3_{framing}"] = bool(
            lo <= o["taylor_test"]["slope_first"] <= hi
            and polish["interior"]
            and max(abs(polish["dW_dw_rel_richardson"]), abs(polish["dW_dw_rel_small"]))
            <= g["G3_stationarity_rel"]
            and (o["active_u_bound"] == "interior" or o["u_multiplier"] >= 0.0)
        )
        out[f"G4_{framing}"] = bool(v["W_rel"] <= g["G4_W_rel"] and v["beta_rel"] <= g["G4_beta_rel"])
        d = ckpt["open_duct"][framing]
        out[f"G5_{framing}"] = bool(
            d["delivered"] >= g["G5_delivered_min"] and d["peak_K"] <= g["G5_peak_max_K"]
        )
        n = ckpt["neglected"][framing]
        out[f"G6_{framing}"] = bool(
            n["Re_over_Ha"] <= g["G6_re_over_ha_max"]
            and max(n["viscous_over_duty"], n["joule_over_duty"]) <= g["G6_dissipation_over_duty_max"]
        )
    return out


def finalize() -> None:
    ckpt = checkpoint.load()
    missing = [s for s in STAGES if not _done(ckpt, s)]
    if missing:
        raise SystemExit(f"stages missing: {missing}")
    import lmhdx

    lmhdx_repo = Path(lmhdx.__file__).resolve().parents[2]

    def lmhdx_status(*paths: str) -> str:
        command = ["git", "status", "--porcelain", "--untracked-files=no", "--", *paths]
        return subprocess.run(command, cwd=lmhdx_repo, capture_output=True, text=True).stdout.strip()

    meta = {
        "study": "stage3-thermal",
        "duct_opt_sha": _git_sha(REPO_ROOT),
        # The package metadata of an editable install keeps its install-time version; the SHA is the code that ran.
        "lmhdx_sha": _git_sha(lmhdx_repo),
        "lmhdx_src_clean": not lmhdx_status("src"),  # the package that ran
        "lmhdx_tracked_tree_clean": not lmhdx_status(),  # its docs and scripts too
        "lmhdx_installed_version": importlib.metadata.version("lmhdx"),
        "note": "The stages ran on the working tree that duct_opt_sha records, except the size of the compiled-film "
        "cache (ductopt.thermal._compiled_film): 48 for reduction and landscape, 16 for the optimum stages, 8 after. "
        "The cache holds compiled programs only and changes no number.",
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "solvax": importlib.metadata.version("solvax"),
    }
    box = _box("duty")
    inputs = {**cases.inputs_record(), "thermal": {k: getattr(box, k) for k in (
        "B", "L", "duty", "rho", "cp", "mu", "sigma", "k", "t_in", "t_max", "laminar", "area_min", "area_max",
        "beta_lo", "beta_hi")}, "flux_W_m2": _box("flux").flux, "cells": [CELLS, LAYER], "fine": list(FINE)}  # fmt: skip
    results = {"meta": meta, "inputs": inputs, "gates_fixed": GATES, **ckpt, "gates": _gates(ckpt)}
    path = OUT_DIR / "results.json"
    path.write_text(json.dumps(results, indent=2, default=float) + "\n")
    print(json.dumps(results["gates"], indent=2))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "all"
    if name == "finalize":
        return finalize()
    names = [s for s in STAGES if not _done(checkpoint.load(), s)] if name == "all" else [name]
    for stage in names:
        t0 = time.perf_counter()
        print(f"== {stage}", flush=True)
        run_stage(stage)
        print(f"== {stage}: {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
