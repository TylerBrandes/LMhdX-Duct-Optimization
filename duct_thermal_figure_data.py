"""Stage 3's figure data, measured for the figures only, not graded and not part of results.json.

Three sets, from the designs results.json records:
- ``cap``: the cost of the cheapest feasible flow along the area cap, 29 aspects, both framings (F2).
- ``fields``: the outlet cross-section's temperature of the four designs (each optimum, Stage 2's design in
  each framing), from a dimensional developed duct on the optimizer's mesh (F3).
- ``open_duct``: each optimum's held flow down an open duct from a 573 K inlet, as G5 but on 24 axial cells,
  sampled 40 times over six residence times (M1).

Writes results/stage3-thermal/figure_data.npz and figure_data.json. Usage: python duct_thermal_figure_data.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("DUCTOPT_RESULTS", str(REPO_ROOT / "results" / "stage3-thermal"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

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
    uniform_faces,
)
from lmhdx.thermal import (  # noqa: E402
    ThermalProblem,
    advance_thermal,
    build_thermal_step,
    developed_heat,
    initial_thermal_state,
    thermal_boundary,
)

import duct_thermal_design as stage3  # noqa: E402
from ductopt.thermal import thermal_objective  # noqa: E402

RESULTS = REPO_ROOT / "results" / "stage3-thermal"
CELLS, LAYER = stage3.CELLS, stage3.LAYER


def cap_curve(record: dict) -> dict:
    out = {"beta": np.geomspace(0.03, 3.0, 29).tolist()}
    for framing in stage3.FRAMINGS:
        box = stage3._box(framing)
        rows = [thermal_objective(box, box.u_hi, float(np.log(b)), CELLS, LAYER) for b in out["beta"]]
        out[framing] = [
            {k: r.get(k) for k in ("feasible", "limit", "W", "pumping_power", "Q", "power")} for r in rows
        ]
    return stage3.strict(out)


def outlet_field(framing: str, design: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The outlet's temperature over the cross-section, in K, and its y and z faces in metres."""
    box = stage3._box(framing)
    _, flow, velocity, _ = stage3._physical_flow(box, design, CELLS, LAYER)
    insulated = thermal_boundary("heat_flux", conductivity=box.k)
    heated = thermal_boundary("heat_flux", -design["wall_flux"], 0.0, conductivity=box.k)
    thermal = ThermalProblem(
        flow, (BoundaryCondition(PERIODIC), insulated, heated), box.cp, box.k, 0.0, joule_heating=False
    )
    heat = developed_heat(thermal, velocity, Field(jnp.zeros(flow.grid.shape), (CENTER,) * 3, flow.grid))
    outlet = box.t_in + float(heat.axial_gradient) * box.L - float(heat.bulk_temperature)
    temperature = np.asarray(heat.temperature.data)[0] + outlet
    return temperature, np.asarray(flow.grid.y_faces), np.asarray(flow.grid.z_faces)


def open_duct_frames(framing: str, design: dict, nx: int = 24, frames: int = 40) -> dict:
    """G5's open duct on ``nx`` axial cells: the outlet cross-section and the mean temperature along x, sampled."""
    box = stage3._box(framing)
    _, periodic, velocity, _ = stage3._physical_flow(box, design, CELLS, LAYER)
    axial_profile, section = velocity[0].data[0], periodic.grid
    grid = Grid(uniform_faces(nx, 0.0, box.L), section.y_faces, section.z_faces)
    dt = 0.5 * (box.L / nx) / float(jnp.max(axial_profile))
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
    held = (rest[0].replace_data(jnp.broadcast_to(axial_profile, rest[0].shape)), *rest[1:])
    state = initial_thermal_state(problem, box.t_in, held)
    step = build_thermal_step(problem, dtype=jnp.float64, hold_flow=True)
    data = np.zeros(grid.shape)
    data[:, :, 0] = design["power"] / float(np.sum(grid.cell_volumes()[:, :, 0]))
    load = Field(jnp.asarray(data), (CENTER,) * 3, grid)
    steps = int(np.ceil(6.0 * design["A"] * box.L / design["Q"] / dt))
    per_frame = max(1, steps // frames)
    run = jax.jit(lambda s: advance_thermal(step, s, load, per_frame)[0])
    areas = np.asarray(grid.cell_volumes())[0] / float(grid.widths[0][0])  # the cross-section's cell areas
    times, outlets, means = [], [], []
    for _ in range(frames + 1):
        temperature = np.asarray(state.temperature.data)
        times.append(float(state.time))
        outlets.append(temperature[-1])
        means.append(np.sum(temperature * areas[None], axis=(1, 2)) / np.sum(areas))
        state = run(state)
    return {
        "time": np.asarray(times), "outlet": np.asarray(outlets), "mean_along_x": np.asarray(means),
        "x": np.asarray(grid.centers[0]), "y_faces": np.asarray(grid.y_faces), "z_faces": np.asarray(grid.z_faces),
        "residence_time": design["A"] * box.L / design["Q"], "dt": dt, "steps_per_frame": per_frame,
    }  # fmt: skip


def main() -> None:
    record = json.loads((RESULTS / "results.json").read_text())
    arrays, meta = {}, {"study": "stage3-thermal figure data", "results_sha": record["meta"]["duct_opt_sha"]}
    meta["duct_opt_sha"] = stage3._git_sha(REPO_ROOT)
    meta["cap"] = cap_curve(record)
    designs = {
        "duty_optimum": ("duty", record["optimum"]["duty"]["design"]),
        "flux_optimum": ("flux", record["optimum"]["flux"]["design"]),
        "duty_stage2": ("duty", record["stage2_design"]["duty"]),
        "flux_stage2": ("flux", record["stage2_design"]["flux"]),
    }
    for name, (framing, design) in designs.items():
        temperature, y, z = outlet_field(framing, design)
        arrays |= {f"field_{name}": temperature, f"field_{name}_y": y, f"field_{name}_z": z}
        print(f"{name}: outlet peak {temperature.max():.3f} K", flush=True)
    for framing in stage3.FRAMINGS:
        frames = open_duct_frames(framing, record["optimum"][framing]["design"])
        arrays |= {f"open_{framing}_{k}": np.asarray(v) for k, v in frames.items()}
        print(
            f"open duct {framing}: {len(frames['time'])} frames, last outlet peak {frames['outlet'][-1].max():.3f} K",
            flush=True,
        )
    np.savez_compressed(RESULTS / "figure_data.npz", **arrays)
    (RESULTS / "figure_data.json").write_text(
        json.dumps(stage3.strict(meta), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
