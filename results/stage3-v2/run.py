"""Stage 3 v2: the open duct refined along its axis, the heated wall's face, and a common feasible flow.

``python results/stage3-v2/run.py STAGE`` runs one stage and saves it to this folder's checkpoint; ``all`` runs
every stage the checkpoint lacks; ``finalize`` writes results.json. The specification and the gates are this
folder's README, committed before the run. v1's record (results/stage3-thermal/results.json, release
stage3-v1) is read and never written.
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

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
os.environ["DUCTOPT_RESULTS"] = str(HERE)  # before ductopt.checkpoint is imported
sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # before LMhdX is imported, as in v1

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from lmhdx.core3d import ChannelProblem, zero_velocity  # noqa: E402
from lmhdx.grid import (  # noqa: E402
    CENTER,
    DIRICHLET,
    NEUMANN,
    PERIODIC,
    BoundaryCondition,
    Field,
    Grid,
    pad,
    uniform_faces,
)
from lmhdx.thermal import (  # noqa: E402
    ThermalProblem,
    advance_thermal,
    build_thermal_step,
    developed_heat,
    initial_thermal_state,
    load_attribution,
    thermal_boundary,
)

import duct_thermal_design as v1  # noqa: E402
from ductopt import checkpoint  # noqa: E402
from ductopt.thermal import thermal_objective  # noqa: E402

V1_RESULTS = REPO_ROOT / "results" / "stage3-thermal" / "results.json"
AXIAL = (12, 24, 48)
MESHES = ((v1.CELLS, v1.LAYER), v1.FINE)
SWEEP = np.linspace(1.0, 2.0, 11)  # the common flow and up to twice it

# The README's gates and checks, fixed before the run.
GATES = {
    "C0_v1_reproduced_rel": 1.0e-12,
    "C1_formulations_rel": 1.0e-9,
    "G5a_delivered_min": 0.95,
    "G5b_wall_face_max_K": 625.0,
    "C2_common_flow_rel": 1.0e-10,
}
SAFETY = 1.25  # Celik et al. (2008), three meshes
UNCONFIRMED = 3.0  # Roache's factor where the order is not confirmed


def _v1() -> dict:
    return json.loads(V1_RESULTS.read_text())


def richardson(fine: float, medium: float, coarse: float) -> dict:
    """The README's extrapolation from three levels at ratio 2: Richardson and Celik's band where monotone."""
    e21, e32 = medium - fine, coarse - medium
    if e21 == 0.0:
        return {"monotone": True, "order": None, "value": fine, "band": 0.0}
    ratio = e32 / e21
    if ratio > 1.0:  # same sign and shrinking
        p = float(np.log(ratio) / np.log(2.0))
        correction = (fine - medium) / (2.0**p - 1.0)
        return {"monotone": True, "order": p, "value": fine + correction, "band": SAFETY * abs(correction)}
    return {"monotone": False, "order": None, "value": fine, "band": UNCONFIRMED * max(abs(e21), abs(e32))}


def _open_duct(box, design: dict, nx: int) -> dict:
    """v1's G5 open duct on ``nx`` axial cells, with the wall flux as a source and as a boundary condition."""
    _, periodic, velocity, _ = v1._physical_flow(box, design, v1.CELLS, v1.LAYER)
    axial_profile, section = velocity[0].data[0], periodic.grid
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
    heated = thermal_boundary("heat_flux", -design["wall_flux"], 0.0, conductivity=box.k)
    rest = zero_velocity(flow)
    held = (rest[0].replace_data(jnp.broadcast_to(axial_profile, rest[0].shape)), *rest[1:])
    steps = int(np.ceil(6.0 * design["A"] * box.L / design["Q"] / dt))  # six residence times
    nothing = Field(jnp.zeros(grid.shape), (CENTER,) * 3, grid)

    # The wall flux as the source q A / V in the fluid row along z = -b (v1's G5).
    problem = ThermalProblem(
        flow, (inlet, insulated, insulated), box.cp, box.k, box.t_in, joule_heating=False
    )
    start = initial_thermal_state(problem, box.t_in, held)
    step = build_thermal_step(problem, dtype=jnp.float64, hold_flow=True)
    data = np.zeros(grid.shape)
    data[:, :, 0] = design["power"] / float(np.sum(grid.cell_volumes()[:, :, 0]))
    load = Field(jnp.asarray(data), (CENTER,) * 3, grid)
    t0 = time.perf_counter()
    credit = load_attribution(step, start, nothing, load, steps)
    final, history = advance_thermal(step, start, load, steps)

    # The same flux as the heated wall's boundary condition, with no source (C1).
    as_wall = ThermalProblem(flow, (inlet, insulated, heated), box.cp, box.k, box.t_in, joule_heating=False)
    walled, _ = advance_thermal(
        build_thermal_step(as_wall, dtype=jnp.float64, hold_flow=True),
        initial_thermal_state(as_wall, box.t_in, held),
        nothing,
        steps,
    )
    seconds = time.perf_counter() - t0

    temperature = np.asarray(final.temperature.data)
    rise = float(np.max(temperature)) - box.t_in
    formulations = float(np.max(np.abs(np.asarray(walled.temperature.data) - temperature))) / rise
    # Face temperatures as developed_heat defines them: the mean of the wall cell and its ghost under the
    # wall's own condition. The heated wall is z = -b; the others are insulated.
    faces, along = [], None
    for axis, condition in ((1, insulated), (2, heated)):
        padded = np.asarray(pad(jnp.asarray(temperature), axis, condition, grid=grid))
        for pair in ((0, 1), (-1, -2)):
            slab = (slice(None),) * axis
            face = 0.5 * (padded[slab + (pair[0],)] + padded[slab + (pair[1],)])
            faces.append(float(np.max(face)))
            if axis == 2 and pair == (0, 1):
                along = np.max(face, axis=1)  # the heated face's largest value in each axial cell
    outlet_areas = grid.face_areas(0)[-1]
    flux = np.asarray(axial_profile) * outlet_areas
    mixed_cup = float(np.sum(flux * temperature[-1]) / np.sum(flux))
    flow_rate = float(np.sum(flux))
    return {
        "nx": nx, "dt": dt, "steps": steps, "time_s": seconds,
        "delivered": float(credit.delivered / credit.imposed), "imposed": float(credit.imposed),
        "conducted": float(credit.conducted), "stored": float(credit.stored), "residual": float(credit.residual),
        "hottest_cell_K": float(jnp.max(history.temperature_max)),
        "hottest_cell_last_K": float(history.temperature_max[-1]),
        "wall_face_peak_K": max(faces), "heated_face_peak_K": faces[2], "wall_faces_K": faces,
        "outlet_mixed_cup_K": mixed_cup,
        "outlet_delivered": box.rho * box.cp * flow_rate * (mixed_cup - box.t_in) / design["power"],
        "formulations_rel": formulations,
        "x_centres": (0.5 * (np.asarray(grid.x_faces)[1:] + np.asarray(grid.x_faces)[:-1])).tolist(),
        "heated_face_along_K": along.tolist(),
    }  # fmt: skip


def stage_refine(framing: str) -> None:
    """A: the optimum's open duct on 12, 24 and 48 axial cells, and the extrapolation of each quantity."""
    box, design = v1._box(framing), _v1()["optimum"][framing]["design"]
    levels = {}
    for nx in AXIAL:
        levels[str(nx)] = _open_duct(box, design, nx)
        row = levels[str(nx)]
        print(f"{framing} nx {nx}: delivered {row['delivered']:.6f}, hottest cell {row['hottest_cell_K']:.4f} K, "
              f"wall face {row['wall_face_peak_K']:.4f} K, C1 {row['formulations_rel']:.1e}, "
              f"{row['time_s']:.0f} s", flush=True)  # fmt: skip
    extrapolated = {
        name: richardson(*(levels[str(nx)][name] for nx in reversed(AXIAL)))
        for name in ("delivered", "hottest_cell_K", "wall_face_peak_K", "outlet_mixed_cup_K")
    }
    v1_record = _v1()["open_duct"][framing]
    c0 = max(
        abs(levels["12"]["delivered"] - v1_record["delivered"]) / v1_record["delivered"],
        abs(levels["12"]["hottest_cell_K"] - v1_record["peak_K"]) / v1_record["peak_K"],
    )
    v1._save(
        ("refine", framing),
        {
            "levels": levels,
            "extrapolated": extrapolated,
            "C0_rel": c0,
            "developed_wall_peak_K": design["T_wall_peak"],
            "developed_fluid_peak_K": design["T_fluid_peak"],
            "developed_bulk_out_K": design["T_bulk_out"],
        },
    )


def _at_flow(box, design: dict, Q: float) -> dict:
    """The closed form at flow ``Q``: one shape's film is independent of the flow (v1's G2)."""
    a, power = design["a"], design["power"]
    bulk = box.t_in + power / (box.rho * box.cp * Q)
    scale = design["wall_flux"] * a / box.k
    pumping = box.mu * box.L * Q**2 / (a**4 * design["q"])
    reynolds = box.rho * (Q / design["A"]) * design["D_h"] / box.mu
    return {
        "Q": Q, "Q_per_wall_m": Q / (2.0 * a), "W": pumping if box.framing == "duty" else pumping / (2.0 * a),
        "pumping_power": pumping, "dp": pumping / Q, "T_bulk_out": bulk, "film": scale * design["theta_wall"],
        "T_wall_peak": bulk + scale * design["theta_wall"], "T_fluid_peak": bulk + scale * design["theta_fluid"],
        "margin_K": box.t_max - (bulk + scale * design["theta_wall"]), "Re_over_Ha": reynolds / design["Ha"],
        "laminar": bool(reynolds / design["Ha"] <= box.laminar),
    }  # fmt: skip


def _dimensional_peaks(box, design: dict, Q: float) -> tuple[float, float]:
    """C2: a dimensional developed_heat of the shape at ``Q``, its anchored wall and fluid peaks."""
    _, flow, velocity, _ = v1._physical_flow(box, {**design, "Q": Q}, v1.CELLS, v1.LAYER)
    insulated = thermal_boundary("heat_flux", conductivity=box.k)
    heated = thermal_boundary("heat_flux", -design["wall_flux"], 0.0, conductivity=box.k)
    thermal = ThermalProblem(
        flow, (BoundaryCondition(PERIODIC), insulated, heated), box.cp, box.k, 0.0, joule_heating=False
    )
    nothing = Field(jnp.zeros(flow.grid.shape), (CENTER,) * 3, flow.grid)
    heat = developed_heat(thermal, velocity, nothing)
    return tuple(float(t) for t in heat.anchored_peaks(box.t_in, box.L))


def stage_common_flow() -> None:
    """B: each framing's optimum and Stage 2's design at one common flow, on 48/6 and 72/9."""
    record, out = _v1(), {}
    shapes = {
        "stage2": (record["stage2_design"]["u"], record["stage2_design"]["w"]),
    }
    for framing in v1.FRAMINGS:
        box = v1._box(framing)
        both = {"stage3": (record["optimum"][framing]["u"], record["optimum"][framing]["w"]), **shapes}
        out[framing] = {}
        for cells, layer in MESHES:
            designs = {name: thermal_objective(box, u, w, cells, layer) for name, (u, w) in both.items()}
            # The common flow: per channel for the duty, per metre of first wall for the flux.
            per = {
                name: d["Q"] if framing == "duty" else d["Q"] / (2.0 * d["a"]) for name, d in designs.items()
            }
            common = max(per.values())
            rows = {}
            for name, d in designs.items():
                Q = common if framing == "duty" else common * 2.0 * d["a"]
                rows[name] = {
                    "u": d["u"], "w": d["w"], "beta": d["beta"], "a": d["a"], "A": d["A"], "Ha": d["Ha"],
                    "power": d["power"], "wall_flux": d["wall_flux"], "theta_wall": d["theta_wall"],
                    "channels_per_wall_m": 1.0 / (2.0 * d["a"]), "own_Q_min": d["Q"], "own_W": d["W"],
                    "at_common": _at_flow(box, d, Q),
                    "sweep": [_at_flow(box, d, f * Q) for f in SWEEP],
                }  # fmt: skip
                if (cells, layer) == (v1.CELLS, v1.LAYER):
                    wall, fluid = _dimensional_peaks(box, d, Q)
                    closed = rows[name]["at_common"]
                    rise = closed["T_wall_peak"] - box.t_in
                    rows[name]["C2_rel"] = (
                        max(abs(wall - closed["T_wall_peak"]), abs(fluid - closed["T_fluid_peak"])) / rise
                    )
            s2, s3 = rows["stage2"]["at_common"], rows["stage3"]["at_common"]
            out[framing][f"{cells}/{layer}"] = {
                "common": common, "common_unit": "m^3/s per channel" if framing == "duty" else "m^2/s per metre of first wall",
                "shapes": rows, "W_ratio_stage2_over_stage3": s2["W"] / s3["W"],
                "film_difference_K": s2["film"] - s3["film"], "sweep_factors": SWEEP.tolist(),
            }  # fmt: skip
            print(f"{framing} {cells}/{layer}: W ratio {s2['W'] / s3['W']:.5f}, film {s2['film']:.4f} "
                  f"against {s3['film']:.4f} K", flush=True)  # fmt: skip
        fine, coarse = (out[framing][f"{c}/{n}"] for c, n in (v1.FINE, (v1.CELLS, v1.LAYER)))
        out[framing]["W_ratio_mesh_change"] = abs(
            fine["W_ratio_stage2_over_stage3"] - coarse["W_ratio_stage2_over_stage3"]
        )
        out[framing]["film_difference_mesh_change_K"] = abs(
            fine["film_difference_K"] - coarse["film_difference_K"]
        )
    v1._save(("common_flow",), out)


STAGES = ["refine_duty", "refine_flux", "common_flow"]


def run_stage(name: str) -> None:
    if name.startswith("refine_"):
        return stage_refine(name[len("refine_") :])
    {"common_flow": stage_common_flow}[name]()


def _done(ckpt: dict, name: str) -> bool:
    if name.startswith("refine_"):
        return name[len("refine_") :] in ckpt.get("refine", {})
    return name in ckpt


def _gates(ckpt: dict) -> dict:
    g, out = GATES, {}
    for framing in v1.FRAMINGS:
        r = ckpt["refine"][framing]
        delivered, face = r["extrapolated"]["delivered"], r["extrapolated"]["wall_face_peak_K"]
        out[f"C0_{framing}"] = r["C0_rel"] <= g["C0_v1_reproduced_rel"]
        out[f"C1_{framing}"] = (
            max(level["formulations_rel"] for level in r["levels"].values()) <= g["C1_formulations_rel"]
        )
        out[f"G5a_{framing}"] = bool(
            r["levels"]["48"]["delivered"] >= g["G5a_delivered_min"]
            and delivered["value"] - delivered["band"] >= g["G5a_delivered_min"]
        )
        out[f"G5b_{framing}"] = bool(
            r["levels"]["48"]["wall_face_peak_K"] <= g["G5b_wall_face_max_K"]
            and face["value"] + face["band"] <= g["G5b_wall_face_max_K"]
        )
        shapes = ckpt["common_flow"][framing][f"{v1.CELLS}/{v1.LAYER}"]["shapes"]
        out[f"C2_{framing}"] = max(s["C2_rel"] for s in shapes.values()) <= g["C2_common_flow_rel"]
    return out


def finalize() -> None:
    ckpt = checkpoint.load()
    missing = [s for s in STAGES if not _done(ckpt, s)]
    if missing:
        raise SystemExit(f"stages missing: {missing}")
    import lmhdx

    lmhdx_repo = Path(lmhdx.__file__).resolve().parents[2]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", "src"],
        cwd=lmhdx_repo, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    meta = {
        "study": "stage3-v2",
        "duct_opt_sha": v1._git_sha(REPO_ROOT),
        "lmhdx_sha": v1._git_sha(lmhdx_repo),
        "lmhdx_src_clean": not status,
        "lmhdx_installed_version": importlib.metadata.version("lmhdx"),
        "v1_record": "results/stage3-thermal/results.json (release stage3-v1)",
        "v1_duct_opt_sha": _v1()["meta"]["duct_opt_sha"],
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "solvax": importlib.metadata.version("solvax"),
    }
    inputs = {"axial_cells": list(AXIAL), "meshes": [list(m) for m in MESHES], "sweep": SWEEP.tolist(),
              "safety": SAFETY, "unconfirmed": UNCONFIRMED}  # fmt: skip
    results = {"meta": meta, "inputs": inputs, "gates_fixed": GATES, **ckpt, "gates": _gates(ckpt)}
    path = HERE / "results.json"
    path.write_text(json.dumps(v1.strict(results), indent=2, default=float, allow_nan=False) + "\n")
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
