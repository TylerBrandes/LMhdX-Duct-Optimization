"""Stage 3: the thermal design of Case P (LMhdX plan step 5.5; results/stage3-thermal/README.md).

Derivation. LMhdX's thermally fully developed duct (``lmhdx.thermal.developed_heat``) solves
``rho c_p u gamma = k lap(theta)`` on the cross-section, with ``gamma`` the axial gradient from the heat
balance ``rho c_p gamma Q = P / L``. The velocity is ``u = Q u_hat``, with ``u_hat`` the shape of the flow
at unit flow rate. So ``u gamma = u_hat P / (rho c_p L)`` does not depend on ``Q``, and neither does the
wall-to-bulk field ``theta``. Lengths scale by ``a`` and the field by ``q'' a / k``. One dimensionless
solve per shape, at unit heated-wall flux, gives

    Theta(beta, Ha) = (peak - bulk) / (q'' a / k),

for the wall and for the fluid. At the outlet the peaks are then

    T_peak = T_in + P / (rho c_p Q) + (q'' a / k) Theta,

so each shape's cheapest feasible flow is ``Q_min = P / (rho c_p (T_max - T_in - (q'' a / k) Theta_max))``.
Its pumping power is ``W = Q_min dp = mu L Q_min^2 / (a^4 q)``, with ``q(Ha, beta)`` Stage 2's flow per unit
drive. The mesh is Stage 2's, one per aspect with the field traced
(``physics._build_problem``), and one jitted function returns ``q`` and both ``Theta`` with their
``Ha`` derivatives.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from lmhdx.fully_developed import channel_flow_rate
from lmhdx.grid import CENTER, PERIODIC, BoundaryCondition, Field
from lmhdx.steady import solve_steady_state
from lmhdx.thermal import ThermalProblem, developed_heat, thermal_boundary

from .cases import PBLI, CaseP
from .physics import _build_problem, mesh_key

K_PBLI = 20.4  # W/(m K), Martelli, Venturini & Utili 2019, Table 1 (D3), as Stage 2's other properties
T_IN, T_MAX = 573.0, 625.0  # K; 625 K is the top of the viscosity correlation's range
LAMINAR = 200.0  # the largest Re/Ha, Re on the hydraulic diameter and Ha on a
DUTY = 1500.0  # W per channel
AREA_MAX = 2.4e-4  # m^2: Stage 2's V_min-bound area, now a geometric limit
AREA_MIN = 2.4e-6  # m^2: Stage 2's smallest area (Q / V_max), kept as the lower bound


def _dimensionless_thermal(problem) -> ThermalProblem:
    """The developed duct on Stage 2's mesh: unit flux into the fluid through ``z = -b``, the rest insulated.

    The axes are Stage 2's: ``x`` the flow, ``y`` along the field (half-width 1, ``a``), ``z`` across
    it (half-width ``beta``, ``b``). A heat-flux value is the flux leaving the fluid, so heat entering
    through the lower ``z`` wall is -1.
    """
    insulated = thermal_boundary("heat_flux", conductivity=1.0)
    heated = thermal_boundary("heat_flux", -1.0, 0.0, conductivity=1.0)
    return ThermalProblem(
        problem, (BoundaryCondition(PERIODIC), insulated, heated), 1.0, 1.0, 0.0, joule_heating=False
    )


@functools.lru_cache(maxsize=8)
def _compiled_film(beta: float, ha_mesh: float, cells: int, cells_in_layer: int, centre: float = 0.0):
    """Return one jitted ``field_scale -> (d/d scale, value)`` of ``[q, Theta_wall, Theta_fluid]``, for one mesh.

    The value rides the reverse pass as its auxiliary output, so each mesh compiles once. The cache is
    smaller than Stage 2's: each entry holds a steady solve, a developed solve and their reverse pass.
    """
    problem = _build_problem(beta, ha_mesh, cells, cells_in_layer, centre)
    thermal, grid = _dimensionless_thermal(problem), problem.grid
    nothing = Field(jnp.zeros(grid.shape), (CENTER,) * 3, grid)

    def film(scale):
        velocity = solve_steady_state(problem, forcing=(1.0, 0.0, 0.0), field_scale=scale).velocity
        heat = developed_heat(thermal, velocity, nothing)
        q = channel_flow_rate(problem, velocity[0].data[0])
        wall = heat.peak_wall_temperature - heat.bulk_temperature
        fluid = jnp.max(heat.temperature.data) - heat.bulk_temperature
        value = jnp.stack([q, wall, fluid])
        return value, value

    return jax.jit(jax.jacrev(film, has_aux=True))


def film(
    beta: float, ha: float, ha_mesh: float, cells: int, cells_in_layer: int, centre: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``[q, Theta_wall, Theta_fluid]`` at ``Ha`` and their ``Ha`` derivatives, on the mesh for ``ha_mesh``."""
    beta_k, ha_mesh_k = mesh_key(beta, ha_mesh, centre)
    derivative, value = _compiled_film(beta_k, ha_mesh_k, cells, cells_in_layer, centre)(ha / ha_mesh_k)
    return np.asarray(value), np.asarray(derivative) / ha_mesh_k


@dataclass(frozen=True)
class ThermalBox:
    """Case P's thermal design: the channel, the heat, the fluid and the limits (the README's specification)."""

    framing: str  # "duty": P fixed per channel; "flux": q'' fixed on the heated wall
    B: float  # T
    L: float  # m
    duty: float = DUTY
    flux: float = 0.0  # W/m^2, set by thermal_box
    rho: float = PBLI["rho"]
    cp: float = PBLI["cp"]
    mu: float = PBLI["mu"]
    sigma: float = PBLI["sigma"]
    k: float = K_PBLI
    t_in: float = T_IN
    t_max: float = T_MAX
    laminar: float = LAMINAR
    area_min: float = AREA_MIN
    area_max: float = AREA_MAX
    beta_lo: float = 0.02
    beta_hi: float = 5.0

    @property
    def u_lo(self) -> float:
        return float(np.log(self.area_min))

    @property
    def u_hi(self) -> float:
        return float(np.log(self.area_max))

    def a_of(self, u: float, w: float) -> float:
        return float(np.sqrt(np.exp(u) / (4.0 * np.exp(w))))

    def ha_mesh_for_beta(self, beta: float) -> float:
        """As Stage 2's: the largest Ha the box reaches at this aspect, with the same 2 % margin."""
        a_max = float(np.sqrt(self.area_max / (4.0 * beta)))
        return 1.02 * self.B * a_max * float(np.sqrt(self.sigma / self.mu))


def thermal_box(case: CaseP, framing: str) -> ThermalBox:
    """Case P with the README's heat: 1.5 kW per channel, or the equal-area square's flux at that duty."""
    if framing not in ("duty", "flux"):
        raise ValueError(f"framing is 'duty' or 'flux', not {framing!r}")
    L = case.poloidal_length_m
    square_flux = DUTY / (2.0 * np.sqrt(AREA_MAX / 4.0) * L)
    return ThermalBox(framing=framing, B=case.B_center_T, L=L, flux=float(square_flux))


def thermal_objective(
    box: ThermalBox, u: float, w: float, cells: int, cells_in_layer: int, centre: float = 0.0
):
    """Return the shape's cheapest feasible flow, its pumping power ``W`` and ``dW/du`` at fixed ``w``.

    ``W`` is infinite where the temperature allowance is used up or ``Q_min`` breaks ``Re/Ha <= laminar``;
    ``feasible`` and ``limit`` say which.
    """
    beta, a = float(np.exp(w)), box.a_of(u, w)
    b, area = beta * a, float(np.exp(u))
    ha = box.B * a * float(np.sqrt(box.sigma / box.mu))
    values, d_dha = film(beta, ha, box.ha_mesh_for_beta(centre or beta), cells, cells_in_layer, centre)
    q, theta_wall, theta_fluid = (float(v) for v in values)
    hottest = 1 if theta_wall >= theta_fluid else 2
    theta, dtheta_dha = float(values[hottest]), float(d_dha[hottest])
    power = box.duty if box.framing == "duty" else box.flux * 2.0 * a * box.L
    wall_flux = power / (2.0 * a * box.L)
    scale = wall_flux * a / box.k  # K per unit Theta
    allowance = box.t_max - box.t_in - scale * theta
    # d(film)/d ln a: the duty's flux falls as 1/a, so its scale is constant; the fixed flux's grows as a.
    dfilm_dlna = scale * dtheta_dha * ha + (scale * theta if box.framing == "flux" else 0.0)
    record = {
        "u": u, "w": w, "beta": beta, "a": a, "b": b, "A": area, "Ha": ha, "q": q,
        "theta_wall": theta_wall, "theta_fluid": theta_fluid, "power": power, "wall_flux": wall_flux,
        "film": scale * theta, "allowance": allowance,
    }  # fmt: skip
    if allowance <= 0.0:
        return {**record, "W": np.inf, "dW_du": np.inf, "feasible": False, "limit": "temperature"}
    Q = power / (box.rho * box.cp * allowance)
    dlnQ_dlna = (1.0 if box.framing == "flux" else 0.0) + dfilm_dlna / allowance
    W = box.mu * box.L * Q**2 / (a**4 * q)
    dlnq_dlna = float(d_dha[0]) * ha / q
    dW_du = W * 0.5 * (2.0 * dlnQ_dlna - 4.0 - dlnq_dlna)  # d ln a / du = 1/2 at fixed w
    hydraulic = 4.0 * a * b / (a + b)
    reynolds = box.rho * (Q / area) * hydraulic / box.mu
    record |= {
        "Q": Q, "V": Q / area, "W": W, "dW_du": dW_du, "dp": W / Q, "D_h": hydraulic, "Re": reynolds,
        "Re_over_Ha": reynolds / ha, "T_bulk_out": box.t_in + power / (box.rho * box.cp * Q),
        "T_wall_peak": box.t_in + power / (box.rho * box.cp * Q) + scale * theta_wall,
        "T_fluid_peak": box.t_in + power / (box.rho * box.cp * Q) + scale * theta_fluid,
    }  # fmt: skip
    if reynolds / ha > box.laminar:
        return {**record, "W": np.inf, "dW_du": np.inf, "feasible": False, "limit": "Re/Ha"}
    return {**record, "feasible": True, "limit": "temperature"}


def feasible_u(
    box: ThermalBox,
    w: float,
    cells: int,
    cells_in_layer: int,
    centre: float = 0.0,
    scan: int = 13,
    steps: int = 30,
):
    """Return the feasible ``(u_lo, u_hi, lower_name, upper_name)`` at an aspect, or None if there is none.

    A coarse scan in ``u`` (warm: one mesh per aspect) finds the feasible run that holds its cheapest sample.
    Each end that is not the box's own bound is refined by bisection on feasibility, to its feasible side,
    and is named by the limit beyond it: ``Re/Ha`` or ``temperature``. The box's ends are ``area floor`` and
    ``area cap``.
    """
    us = np.linspace(box.u_lo, box.u_hi, scan)
    samples = [thermal_objective(box, float(u), w, cells, cells_in_layer, centre) for u in us]
    feasible = [s["feasible"] for s in samples]
    if not any(feasible):
        return None
    best = min((i for i in range(scan) if feasible[i]), key=lambda i: samples[i]["W"])
    lo = best
    while lo > 0 and feasible[lo - 1]:
        lo -= 1
    hi = best
    while hi < scan - 1 and feasible[hi + 1]:
        hi += 1

    def edge(inside: float, outside: float) -> tuple[float, str]:
        limit = thermal_objective(box, outside, w, cells, cells_in_layer, centre)["limit"]
        for _ in range(steps):
            middle = 0.5 * (inside + outside)
            if thermal_objective(box, middle, w, cells, cells_in_layer, centre)["feasible"]:
                inside = middle
            else:
                outside = middle
        return inside, limit

    u_lo, lower = (float(us[lo]), "area floor") if lo == 0 else edge(float(us[lo]), float(us[lo - 1]))
    u_hi, upper = (float(us[hi]), "area cap") if hi == scan - 1 else edge(float(us[hi]), float(us[hi + 1]))
    return u_lo, u_hi, lower, upper


def profile(box: ThermalBox, w: float, cells: int, cells_in_layer: int, centre: float = 0.0) -> dict | None:
    """The cheapest area at one aspect: Stage 2's exact bisection on ``dW/du`` inside the feasible interval.

    Returns the design there, the interval and the active bound with its multiplier (``dW/du`` signed so a
    positive value means ``W`` would fall further past the bound), or None if no area is feasible.
    """
    from .optimize import _bisect_u

    found = feasible_u(box, w, cells, cells_in_layer, centre)
    if found is None:
        return None
    u_lo, u_hi, lower, upper = found
    evaluate = functools.partial(thermal_objective, centre=centre)
    u = _bisect_u(box, w, cells, cells_in_layer, u_lo, u_hi, evaluate)
    design = evaluate(box, u, w, cells, cells_in_layer)
    if u >= u_hi - 1.0e-9:
        bound, multiplier = upper, -design["dW_du"]
    elif u <= u_lo + 1.0e-9:
        bound, multiplier = lower, design["dW_du"]
    else:
        bound, multiplier = "interior", 0.0
    return {"design": design, "u_range": [u_lo, u_hi], "bound": bound, "multiplier": multiplier}


def optimize_thermal(
    box: ThermalBox,
    w0: float,
    cells: int,
    cells_in_layer: int,
    *,
    max_rounds: int = 12,
    w_tol: float = 1.0e-3,
) -> dict:
    """Minimize ``W*(w) = min_u W(u, w)``: Stage 2's quadratic bracket in ``w``, each sample at its exact best ``u``.

    Stage 2's optimizer alternates the two steps; nested, they do not zig-zag along a valley where ``u`` and
    ``w`` are coupled, as the fixed-flux framing's is. An infeasible aspect is worth infinity.
    """
    from .optimize import _BRACKET_FACTORS, _quad_bracket

    w_lo, w_hi = float(np.log(box.beta_lo)), float(np.log(box.beta_hi))
    profiles: dict[float, dict | None] = {}

    def at(w: float) -> dict | None:
        if w not in profiles:
            profiles[w] = profile(box, w, cells, cells_in_layer)
        return profiles[w]

    def value(box_, u, w, cells_, layer_):
        found = at(float(w))
        return {"W": np.inf if found is None else found["design"]["W"]}

    w, rounds, interior = float(np.clip(w0, w_lo, w_hi)), [], False
    for _ in range(max_rounds):
        w_new, W_new, interior, evaluated, _ = _quad_bracket(
            box, 0.0, w, w_lo, w_hi, cells, cells_in_layer, _BRACKET_FACTORS, value
        )
        rounds.append(
            {"w_center": w, "w_next": w_new, "W": W_new, "interior": interior, "evaluations": evaluated}
        )
        moved, w = abs(w_new - w), w_new
        if interior and moved < w_tol:
            break
    final = at(w)
    return {
        "w": w, "beta": float(np.exp(w)), "converged": bool(interior and moved < w_tol),
        "rounds": rounds, "profiles_evaluated": len(profiles), **(final or {"design": None}),
    }  # fmt: skip
