"""Stage 3 gates G1 and G2: the dimensionless film scales exactly, and it does not depend on the flow rate.

G1: a dimensional developed duct of one physical Case P design (metres, PbLi, its own field, the wall
flux in W/m^2) against T_in + P / (rho c_p Q) + (q'' a / k) Theta from the dimensionless solve: the
anchored wall and fluid peaks within 1e-10 relative of their rise. G2: the dimensionless film at the
flow scaled by 1 and by 7.3, within 1e-12 relative. Usage: python probe_thermal_reduction.py [BETA]
"""

import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from lmhdx.core3d import ChannelProblem  # noqa: E402
from lmhdx.fully_developed import channel_flow_rate  # noqa: E402
from lmhdx.grid import CENTER, PERIODIC, BoundaryCondition, Field, Grid  # noqa: E402
from lmhdx.steady import solve_steady_state  # noqa: E402
from lmhdx.thermal import ThermalProblem, developed_heat, thermal_boundary  # noqa: E402

from ductopt.cases import case_p  # noqa: E402
from ductopt.physics import _build_problem, mesh_key  # noqa: E402
from ductopt.thermal import _dimensionless_thermal, film, thermal_box, thermal_objective  # noqa: E402

BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
CELLS, LAYER = 48, 6
box = thermal_box(case_p(), "duty")
u, w = float(np.log(2.0e-4)), float(np.log(BETA))
start = time.perf_counter()
design = thermal_objective(box, u, w, CELLS, LAYER)
print(f"beta {BETA}: a {design['a'] * 1e3:.3f} mm, Ha {design['Ha']:.2f}, theta wall {design['theta_wall']:.6f}"
      f" fluid {design['theta_fluid']:.6f}, Q_min {design['Q']:.4e} m^3/s, W {design['W']:.4e} W"
      f" ({time.perf_counter() - start:.1f} s with compile)")  # fmt: skip

# G1: the same design in metres.
ha_mesh = box.ha_mesh_for_beta(BETA)
beta_k, ha_mesh_k = mesh_key(BETA, ha_mesh)
unit = _build_problem(beta_k, ha_mesh_k, CELLS, LAYER)
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
velocity = tuple(
    c.replace_data(c.data * (design["Q"] / channel_flow_rate(flow, velocity[0].data[0]))) for c in velocity
)
insulated = thermal_boundary("heat_flux", conductivity=box.k)
heated = thermal_boundary("heat_flux", -design["wall_flux"], 0.0, conductivity=box.k)
thermal = ThermalProblem(
    flow, (BoundaryCondition(PERIODIC), insulated, heated), box.cp, box.k, 0.0, joule_heating=False
)
nothing = Field(jnp.zeros(grid.shape), (CENTER,) * 3, grid)
wall, fluid = (float(t) for t in developed_heat(thermal, velocity, nothing).anchored_peaks(box.t_in, box.L))
rise = design["T_wall_peak"] - box.t_in
g1 = max(abs(wall - design["T_wall_peak"]), abs(fluid - design["T_fluid_peak"])) / rise
print(f"G1: dimensional wall {wall:.10f} K, fluid {fluid:.10f} K; scaled {design['T_wall_peak']:.10f},"
      f" {design['T_fluid_peak']:.10f}; worst {g1:.2e} of the rise (gate 1e-10): {'pass' if g1 <= 1e-10 else 'FAIL'}")  # fmt: skip

# G2: the dimensionless film at two flow rates.
dimensionless = _dimensionless_thermal(unit)
shape = solve_steady_state(unit, forcing=(1.0, 0.0, 0.0), field_scale=design["Ha"] / ha_mesh_k).velocity
nothing_unit = Field(jnp.zeros(unit.grid.shape), (CENTER,) * 3, unit.grid)
films = []
for factor in (1.0, 7.3):
    heat = developed_heat(dimensionless, tuple(c.replace_data(factor * c.data) for c in shape), nothing_unit)
    films.append(np.array([heat.peak_wall_temperature - heat.bulk_temperature,
                           jnp.max(heat.temperature.data) - heat.bulk_temperature]))  # fmt: skip
g2 = float(np.max(np.abs(films[1] - films[0]) / np.abs(films[0])))
print(
    f"G2: film {films[0]} at 1x, {films[1]} at 7.3x; worst {g2:.2e} relative (gate 1e-12): {'pass' if g2 <= 1e-12 else 'FAIL'}"
)

# The derivative: d Theta / d Ha against central differences of the jitted value.
values, d_dha = film(BETA, design["Ha"], ha_mesh, CELLS, LAYER)
h = 1e-4 * design["Ha"]
central = (
    film(BETA, design["Ha"] + h, ha_mesh, CELLS, LAYER)[0]
    - film(BETA, design["Ha"] - h, ha_mesh, CELLS, LAYER)[0]
) / (2 * h)
print(
    f"d/dHa [q, theta_wall, theta_fluid]: {d_dha}; central {central}; relative {np.abs(d_dha - central) / np.abs(central)}"
)
