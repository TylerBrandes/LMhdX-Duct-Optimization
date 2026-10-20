# Stage 3 v2: the open duct refined, the wall face, a common feasible flow

**Status: set-up, not yet run.** The specification and gates below are committed before the run and
will not be relaxed after it.

Stage 3 ([results/stage3-thermal](../stage3-thermal/README.md), release `stage3-v1`) left three items
open. This study measures them. Its record stays separate, and v1's is read but never rewritten.

1. **The open duct on one axial mesh.** v1's G5 ran each optimum's held flow down an open duct on 12
   axial cells. That shows a discrete balance at one resolution, not a grid-independent delivered
   share or entrance temperature.
2. **The hottest cell, not the wall.** v1's G5 graded the hottest fluid cell. The developed screening's
   binding peak is the heated wall's face, and the open duct did not measure it.
3. **No comparison at a common flow.** v1 compared Stage 2's design with each optimum at each one's own
   cheapest feasible flow. A comparison at one flow that both shapes can carry keeps the bulk
   temperature rise equal and isolates what the shape changes: the wall-to-bulk film and the
   hydraulic power.

v1's fixed-flux stationarity gate (G3, −1.8e-6 against 1e-6) stays an open, recorded exception. This
study does not revisit it.

## Specification

Everything is v1's: Case P, Pb-17Li, the 573 K inlet, the 625 K limit, the two framings (1.5 kW per
channel, or 96.82 kW/m² on the first wall), and Stage 2's 48/6 cross-section mesh. The candidates are
read from v1's `results.json`:

- the fixed-duty optimum (b/a 0.1437);
- the fixed-flux optimum (b/a 0.2839);
- Stage 2's design (b/a 0.150, A = 2.4e-4 m²).

LMhdX is 1.11.0 (`4e1b6b3`), on which v1 reproduces bit for bit.

## Method

**A. The open duct, refined along its axis** (stages `refine_duty`, `refine_flux`)

Each optimum's held flow is carried down the 1 m duct from a uniform 573 K start, exactly as in v1's
G5:

- upwind transport;
- a Courant number of 0.5 along the duct;
- six residence times, by which the duct is steady;
- the delivered share from `lmhdx.thermal.load_attribution`.

The axial mesh is 12, 24 and 48 cells. The cross-section stays 48/6; G4 measured its own refinement.

At each level the run is made twice:

- **Wall flux as a source**, as in v1: `q'' A / V` in the row of fluid cells along the heated wall.
  `load_attribution` needs the load as a source.
- **Wall flux as a boundary condition**: the same flux, with no source. In the finite-volume balance
  a face flux enters its cell exactly as that source does, so the two must agree (C1).

From each final state the study takes:

- **the hottest cell**: v1's quantity, the largest cell temperature over the run;
- **the hottest wall face**: the largest face temperature on any transverse wall. A face's value is
  defined as in LMhdX's developed duct (`developed_heat`): the mean of the wall cell and its ghost
  under the wall's own condition, from `lmhdx.grid.pad`. On the heated wall that is
  `T_P + q'' h / (2k)`, with `h` the wall cell's width. On an insulated wall it is the cell value.
- **the outlet's mixed-cup temperature**: weighted by the axial flow. With the delivered share it
  checks the heat delivered at the outlet.
- **the heated wall's face temperature along the duct**: its largest value across the wall, in each
  axial cell. The figure for this is reported, not gated.

**Extrapolation.** Each quantity is extrapolated from its three levels (refinement ratio 2):

- **monotone convergence:** the observed order is `p = ln(e₃₂ / e₂₁) / ln 2`, with `e₂₁ = f₂₄ − f₄₈`
  and `e₃₂ = f₁₂ − f₂₄`. The Richardson value is `f₄₈ + (f₄₈ − f₂₄) / (2^p − 1)`, with Celik's (2008)
  band `1.25 |f₄₈ − f₂₄| / (2^p − 1)`;
- **otherwise** (the differences change sign or grow): no extrapolation. The value is `f₄₈` and the
  band is three times the larger difference (Roache's factor for an unconfirmed order).

**B. A common feasible flow** (stage `common_flow`)

Each framing compares its optimum with Stage 2's design at one common flow:

- **fixed duty:** the flow per channel;
- **fixed flux:** the flow per metre of first wall, `Q / (2a)`.

Either way, the two shapes then have the same bulk temperature rise. The common flow is the larger of
the two shapes' cheapest feasible flows, so both can carry it.

At that flow, and up to twice it, the study reports for each shape:

- the cost `W`, in v1's units: per channel for the duty, per metre of first wall for the flux;
- the pressure drop;
- the film, the wall-to-bulk difference;
- the peak wall temperature and its margin below 625 K;
- `Re/Ha`;
- for the flux, the load per channel and the channels per metre of first wall.

These follow from one developed solve per shape (v1's `Θ`, independent of the flow, G2), on 48/6
and on 72/9.

**The rule for a difference.** A difference between the shapes that is smaller than its own change
between 48/6 and 72/9 is reported as no difference.

## Gates and checks

| Name | What | Threshold |
|:--|:--|:--|
| C0, v1 reproduced | the 12-cell source run's delivered share and hottest cell against v1's G5 record | 1e-12 relative |
| C1, the two formulations | final temperature, wall flux as a source against wall flux as a boundary condition, at every level | 1e-9 of the temperature rise |
| G5a, delivered, refined | `f₄₈` and the extrapolated value minus its band | ≥ 95 % (v1's threshold) |
| G5b, the hottest wall face, refined | `f₄₈` and the extrapolated value plus its band | ≤ 625 K (the specification's limit) |
| C2, the common flow's closed form | a dimensional `developed_heat` of each shape at the common flow against the closed form, on 48/6 | 1e-10 of the rise (v1's G1) |

**Reported, not gated:**

- the hottest cell, refined;
- the observed orders;
- the outlet's mixed-cup temperature;
- the open duct's wall-face peak against the developed screening's 625 K (the entrance region's margin);
- the wall temperature along the duct;
- every quantity of B.

## Outputs

- `run.py`: the driver. `python results/stage3-v2/run.py <stage> | all | finalize`, one stage per
  process.
- `checkpoint.json` and `results.json`: strict JSON, with v1's SHA and LMhdX's recorded.
- `make_figures.py`: the figures, from `results.json` alone, in Stage 2's house style.
- `figures/`, `logs/`.

This README then becomes the write-up. LMhdX's plan row 5.5 and
[uwplasma/LMhdX#271](https://github.com/uwplasma/LMhdX/pull/271) point here.
