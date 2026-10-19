# Stage 3: a thermal design for Case P (LMhdX plan step 5.5)

**Status: set-up, before it runs.** Everything below was fixed before any result was looked at. The
gates will not be relaxed after the run.

Stage 2 took each duct's area from a smallest allowed mean velocity, `V_min = 10 mm/s`, which stood
in for heat removal. Stage 3 asks the question directly. Which cross-section of the Case P channel
removes a first-wall heat load within a temperature limit, at the least pumping power? And what does
the velocity bound get wrong?

## The question, and why the flow must be free

At Stage 2's flow rate, `Q = 2.4e-6 m³/s`, the PbLi carries `ρ c_p Q = 4.48 W/K`. The temperature
allowance below, 52 K, then removes at most 233 W per channel, and the wall-to-bulk difference
across a 6 mm gap is only a few kelvin. A temperature limit at fixed `Q` would be almost pure energy
balance. At a fixed duty it would bound nothing, since the bulk rise does not depend on the shape.

So here **heat sets the flow**. For each shape, the flow rate is the smallest that keeps every
temperature within the limit, and the shape is chosen to make that flow cheapest to pump.

## Specification

The specification is illustrative. None of these values is an engineering requirement.

- **The channel** is Case P of Stage 2: the vertical poloidal duct at the outboard midplane,
  1.0 m long, in its uniform centre field `B_c = 0.4805 T`. The cross-duct `1/R` is neglected, as in
  Stage 2, where `case_p_correction` measured it. The walls are insulating, and the flow is fully
  developed and inertialess: Stage 2's evaluator.
- **The heat** enters through the wall `z = -b`, the one of width `2a` that faces the plasma, with
  `a` the half-width along the field. It is either:
  - **a fixed duty** of 1.5 kW per channel, whose flux `P/(2aL)` falls as the wall widens; or
  - **a fixed flux** of 96.82 kW/m², the equal-area square duct's flux at that duty, whose duty
    grows with `2a`.
- **The fluid** is Pb-17Li at 573 K, with Stage 2's properties (Martelli et al. 2019, Table 1)
  and `k = 20.4 W/(m K)` from the same table, so `Pr = 0.020`.
- **The inlet** is at 573 K.
- **The limits:**
  - the peak fluid and wall-interface temperatures at most 625 K, the top of the viscosity
    correlation's range;
  - `Re/Ha ≤ 200`, with `Re` on the hydraulic diameter and `Ha` on `a`, for laminar Hartmann
    layers;
  - an area of at most 2.4e-4 m². That is Stage 2's `V_min`-bound area, kept now as a geometric
    limit;
  - an aspect `b/a` in [0.02, 5], Stage 2's bounds.
- **The cost** is the hydraulic power `W = Q Δp`, with no pump efficiency.
- **Left out:** buoyancy (the flow is held), and Joule and viscous heating. Their sizes are measured
  on the candidate and reported.

## Method

The thermally fully developed duct of LMhdX (`lmhdx.thermal.developed_heat`, plan step 5.2) has
`ρ c_p u γ = k ∇²θ`, with `u ∝ Q` and the axial gradient `γ ∝ 1/Q`. The wall-to-bulk field `θ`
therefore depends on the shape alone, not on the flow rate. At the outlet:

```
T_peak = T_in + P / (ρ c_p Q) + (q'' a / k) Θ(β, Ha)
```

Here `Θ` is the dimensionless peak, of the wall and of the fluid, from one developed solve per
shape, on Stage 2's mesh family (one mesh per aspect, the field magnitude traced). Each shape's
cheapest feasible flow is then

```
Q_min = P / (ρ c_p (T_max - T_in - (q'' a / k) Θ_max))
```

and its cost is `W = Q_min² L μ / (a⁴ q(Ha, β))`. Stage 2's block-coordinate optimizer minimizes it
over `(u, w) = (ln A, ln β)`, with `dW/du` exact through `Θ`'s dependence on `Ha`. A shape whose
`Q_min` breaks `Re/Ha ≤ 200`, or for which the temperature allowance goes negative, is infeasible.

Each framing is also compared with Stage 2's design. That design is `A = 2.4e-4 m²` and
`b/a = 0.150`, at the same limits:
- **At fixed duty or flux:** its pumping power at its own `Q_min`, against the thermal optimum's.
- **At fixed pumping power:** the largest duty each sampled shape removes, at the Stage 2 design's
  power.

The plan row asks for both kinds of comparison.

## Gates

- **G1, scaling.** The dimensionless `Θ`, scaled, matches a dimensional `developed_heat` of the same
  physical design to 1e-10 relative.
- **G2, the reduction.** `θ` at two flow rates differs by at most 1e-12 relative, so `Q_min` is
  exact.
- **G3, the optimum.**
  - `dW/du` passes a Taylor test, including the path through `Θ(Ha)`.
  - At convergence, `dW/dw` is within Stage 2's stationarity tolerance.
  - An active bound carries a multiplier of the right sign.
- **G4, the mesh.** On the next mesh of Stage 2's family, 72/9 cells against the optimizer's 48/6,
  the candidate's `W` is within 1 % and its `β` within 2 %.
- **G5, an open duct.** The candidate's held flow, carried down an open duct from a 573 K inlet with
  upwind transport (`lmhdx.thermal.load_attribution`), must deliver at least 95 % of the duty at
  the outlet. Its peak must also stay within 625 K. The developed screening delivers every watt by
  construction, so this is checked on the candidate, as on LMhdX's design page.
- **G6, the neglected terms.**
  - The candidate's `Re/Ha` is at most 200.
  - Joule and viscous heating, measured against the duty, are at most 1e-3.
  - The buoyancy ratio `Ri/N` is reported, not graded: the flow is held.

## Outputs

`results.json` will hold every input, value and gate; figures go in `figures/`. This README then
becomes the write-up. LMhdX's plan row 5.5 points here.
