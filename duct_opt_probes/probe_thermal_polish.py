"""Post-hoc, after Stage 3's run: does the fixed-flux polish's residual dW/dw come from the quartic fit?

The run's G3 measurement (polish_w on the scaled-mesh family centred at the optimizer's w) read Richardson
-1.8e-6 and h/10 -1.9e-6 against the 1e-6 gate: a fail, and it stays one. This repeats the same
measurement centred at the polish's own minimum, a second quartic about a closer centre. A reading well
inside 1e-6 there says the first reading was the fit's offset, not a slope of W; it does not change G3.
Writes results/stage3-thermal/posthoc_polish_flux.json. Usage: python probe_thermal_polish.py
"""

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from ductopt.cases import case_p  # noqa: E402
from ductopt.optimize import polish_w  # noqa: E402
from ductopt.thermal import profile, thermal_box  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "stage3-thermal"
first = json.loads((RESULTS / "results.json").read_text())["optimum"]["flux"]["polish"]
box = thermal_box(case_p(), "flux")
w_centre = float(np.log(first["beta_star"]))


def value(w: float) -> float:
    found = profile(box, w, 48, 6, float(np.exp(w_centre)))
    return np.inf if found is None else found["design"]["W"]


second = polish_w(value, w_centre)
out = {"first": first, "recentred": second, "centre_beta": float(np.exp(w_centre))}
(RESULTS / "posthoc_polish_flux.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
print(
    f"first: b/a* {first['beta_star']:.6f} Richardson {first['dW_dw_rel_richardson']:.2e} h/10 {first['dW_dw_rel_small']:.2e}"
)
print(
    f"re-centred: b/a* {second['beta_star']:.6f} Richardson {second['dW_dw_rel_richardson']:.2e} h/10 {second['dW_dw_rel_small']:.2e}"
)
