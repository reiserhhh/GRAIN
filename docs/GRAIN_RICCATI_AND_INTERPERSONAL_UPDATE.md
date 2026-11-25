# Practitioner Manual: Constrained Riccati-Möbius Flows, Dynamic Set Inversion, and Interpersonal Updates (`grain_core.riccati_mobius`)

This manual covers the standalone numerical modules introduced in
`grain_core.riccati_mobius` (`suica_core/riccati_mobius.py`) as an exact
continuous-time and multi-time companion to the frozen liquid trajectory toolkit
([`GRAIN_LIQUID_TOOL.md`](GRAIN_LIQUID_TOOL.md)). For mathematical proofs, Lean 4
formalization references (`58` theorems in companion `project-persona`), and
public multi-corpus benchmarks, see
[`paper/GRAIN_LIQUID_RICCATI_SUPPLEMENT_EN_V2.md`](paper/GRAIN_LIQUID_RICCATI_SUPPLEMENT_EN_V2.md).

> **Scope notice.** `grain_core.riccati_mobius` provides deterministic trajectory
> integration and linear-programming set inversion over observed coordinate
> streams. It is **not a personality scale** or clinical diagnostic classifier,
> and formal soundness does not substitute for construct validation.

For a downstream decision, start with the [same-history comparison](GRAIN_EVENT_AND_VIEW_INCREMENT.md). A narrower interval than today's static band alone does not establish that a joint fit outperforms simple lagged contrasts.

## 1. API Inventory (`grain_core.riccati_mobius`)

| Exported Symbol | Purpose | Key Guarantee / Output | Boundary / Abstention Rule |
|---|---|---|---|
| `ConstrainedRiccatiCell` | Exact normalized $2\times 2$ ZOH Möbius matrix exponential integrating $\frac{dx}{dt}=\epsilon(u)(\alpha(u)+\beta(u)x+\gamma(u)x^2)$ | Strict $[-B, B]$ invariance, `dt = 0` identity preservation, subdivision consistency, and $O(\log T)$ Kogge-Stone parallel scan (`<= 1e-12`) | Piecewise ZOH input assumption; captures fast contractive relaxation unless paired with episodic memory |
| `DualPathwayRiccatiObserver` | Combines fast continuous Riccati decay across irregular idle delays with content-addressable cosine retrieval | Retains opening-event sensitivity across long empty gaps while supporting `mask_first` causal idle ablation | Episodic mixture weights reflect history similarity, not causal counterfactual token rewrites |
| `estimate_cross_view_lag1_rates` | Legacy clipped lag-one cross-view moment ratios | Returns three rates; the implementation falls back to zero on small denominators | Use independent source trajectories; nonpositive common/private variance estimates do not identify rates. Zero fallback does not establish zero persistence. |
| `solve_axis_trajectory_lp` | Exact HiGHS linear program over shared state trajectory $c_0, \dots, c_{T-1}$ | Collapses private initial offsets whenever $r_c \ne r_{p,i}$; preserves exact interval feasibility | Returns `None` on infeasible noise tolerances (never forces projection); widens under $r_c = r_{p,1} = r_{p,2}$ |
| `stack_transition_contrasts` | Builds calibrated $(A, b)$ linear system by stacking $A_t - P_{j,t}$ across steps and readers | Recovers full-rank $c_0$ from multiple window transitions even if individual steps are rank-deficient | Assumes calibrated regime maps `common_maps` and `private_maps` fitted on disjoint source data |

## 2. Runnable Practitioner Recipes

All examples below import directly from `grain_core.riccati_mobius` using NumPy
and SciPy (`linprog` with HiGHS) with no external data or neural model server
required.

### 2.1 Exact ZOH Möbius Flow: Subdivision Invariance & $O(\log T)$ Parallel Scan

Verify `dt = 0` identity preservation, time-subdivision semigroup invariance,
strict `[-B, B]` confinement, and exact equivalence between sequential rollout
and `rollout_parallel_scan`:

```python
import numpy as np
from grain_core.riccati_mobius import ConstrainedRiccatiCell

rng = np.random.default_rng(7)
cell = ConstrainedRiccatiCell(input_dim=5, hidden_dim=8, bound_B=1.25, s_min=0.05)

u = rng.normal(size=(5,))
x0 = rng.uniform(-0.9, 0.9, size=(8,))

# 1. Exact dt = 0 identity preservation
x_zero = cell.apply_mobius(cell.step_matrices(u, 0.0), x0)
assert np.max(np.abs(x_zero - x0)) < 1e-15

# 2. Exact time-subdivision semigroup law & [-B, B] containment
dt_total = 1.6
x_one = cell.apply_mobius(cell.step_matrices(u, dt_total), x0)
M_quarter = cell.step_matrices(u, dt_total / 4.0)
x_sub = x0.copy()
for _ in range(4):
    x_sub = cell.apply_mobius(M_quarter, x_sub)
assert np.all(np.abs(x_one) <= cell.bound_B + 1e-12)
assert np.max(np.abs(x_one - x_sub)) < 1e-12

# 3. Sequential rollout vs O(log T) Kogge-Stone parallel prefix scan
u_seq = rng.normal(size=(16, 5))
dt_seq = rng.uniform(0.05, 1.5, size=(16,))
xs_seq = cell.rollout_sequential(u_seq, dt_seq, x0=x0)
xs_par = cell.rollout_parallel_scan(u_seq, dt_seq, x0=x0)
assert np.max(np.abs(xs_seq - xs_par)) < 1e-12
```

### 2.2 Dual-Pathway Continuous Decay + Episodic Memory Over Long Gaps

Contrast fast-only decay against dual-pathway episodic retrieval across empty
inter-event intervals (`gaps`), using `mask_first=True` to isolate opening-event
sensitivity:

```python
import numpy as np
from grain_core.riccati_mobius import DualPathwayRiccatiObserver

obs = DualPathwayRiccatiObserver(dim=6, episodic_weight=0.35, gamma_scale=0.35)
events = np.full((16, 6), 1.0 / 6.0)
events[0] = np.array([0.70, 0.06, 0.06, 0.06, 0.06, 0.06])
events[-1] = np.array([0.70, 0.06, 0.06, 0.06, 0.06, 0.06])
gaps = np.full((16,), 2.0)
gaps[0] = 0.0

fast_diff = 0.5 * float(np.sum(np.abs(
    obs.rollout_with_idle_gaps(events, gaps, mask_first=False, enable_episodic=False)[-1]
    - obs.rollout_with_idle_gaps(events, gaps, mask_first=True, enable_episodic=False)[-1]
)))
dual_diff = 0.5 * float(np.sum(np.abs(
    obs.rollout_with_idle_gaps(events, gaps, mask_first=False, enable_episodic=True)[-1]
    - obs.rollout_with_idle_gaps(events, gaps, mask_first=True, enable_episodic=True)[-1]
)))
assert fast_diff < 1e-4 and dual_diff > 0.01
```

### 2.3 Cross-View Rate Estimation & Dynamic Multi-Time Set Inversion

Estimate $(r_c, r_{p,1}, r_{p,2})$ on calibration panels and compute exact
common-trajectory intervals `lo, hi` with `solve_axis_trajectory_lp`. Always
check for `None` (conformal/tolerance infeasibility) and treat `lo, hi` as
partial-identification bounds rather than point traits:

```python
import numpy as np
from grain_core.riccati_mobius import (
    estimate_cross_view_lag1_rates,
    solve_axis_trajectory_lp,
)

rng = np.random.default_rng(123)
N, T = 40, 6
rc_true, rp1_true, rp2_true = 0.85, 0.25, 0.40
c, p1, p2 = np.zeros((N, T)), np.zeros((N, T)), np.zeros((N, T))
c[:, 0] = rng.normal(0.0, 1.0, N)
p1[:, 0] = rng.normal(0.0, 1.2, N)
p2[:, 0] = rng.normal(0.0, 1.2, N)
for t in range(T - 1):
    c[:, t + 1] = rc_true * c[:, t] + rng.uniform(-0.02, 0.02, N)
    p1[:, t + 1] = rp1_true * p1[:, t] + rng.uniform(-0.02, 0.02, N)
    p2[:, t + 1] = rp2_true * p2[:, t] + rng.uniform(-0.02, 0.02, N)

y1 = (c + p1)[:, :, None]
y2 = (c + p2)[:, :, None]
est_c, est_p1, est_p2 = estimate_cross_view_lag1_rates(y1, y2, rank=1)

bounds = solve_axis_trajectory_lp(
    y1_t=y1[0, :, 0],
    y2_t=y2[0, :, 0],
    r_c=float(est_c[0]),
    r_p1=float(est_p1[0]),
    r_p2=float(est_p2[0]),
    D=0.30,
    E1=0.30,
    E2=0.30,
    static_half_widths=None,
    c_bound=10.0,
)
if bounds is None:
    print("Abstain: observation trajectory is infeasible under calibrated tolerances.")
else:
    lo, hi = bounds
    assert np.all(lo[:-1] <= hi[:-1] + 1e-9)
    print(f"Feasible trajectory interval median width: {np.median(hi[:-1] - lo[:-1]):.4f}")
```

### 2.4 Multi-Window Linear Contrast Stacking (`stack_transition_contrasts`)

When single-step transitions are rank-deficient, stack calibrated multi-reader
transition contrasts `(A_t - P_{j,t})` to eliminate arbitrary initial private
offsets and recover $c_0$:

```python
import numpy as np
from grain_core.riccati_mobius import stack_transition_contrasts

rng = np.random.default_rng(31)
c = np.zeros((3, 2))
c[0] = np.array([1.2, -0.7])
A = np.array([[[0.8, 0.1], [-0.05, 0.75]], [[0.7, 0.0], [0.1, 0.85]]])
P = np.array([
    [[[0.8, 0.1], [-0.05, 0.2]], [[0.8, 0.1], [-0.05, 0.3]]],
    [[[0.2, 0.0], [0.1, 0.85]], [[0.3, 0.0], [0.1, 0.85]]],
])
p = rng.normal(0.0, 0.8, size=(3, 2, 2))
g = np.array([[0.1, -0.05], [0.0, 0.15]])
for t in range(2):
    c[t + 1] = A[t] @ c[t] + g[t]
    for r in range(2):
        p[t + 1, r] = P[t, r] @ p[t, r]

y = c[:, None, :] + p
H, b = stack_transition_contrasts(y, A, P, common_drive=g)
c0_hat = np.linalg.lstsq(H, b, rcond=None)[0]
assert np.max(np.abs(c0_hat - c[0])) < 1e-12
```

## 3. Formal Guarantees, Known Failure Modes, and Companion Verification

- **Lean 4 Deductive Guarantee (Companion `project-persona`).** Seven Lean 4
  modules (`58` theorems verified with zero `sorryAx`) prove scalar Möbius
  domain confinement, composition associativity, equal-rate LP degeneracy, and
  five-operator interpersonal identities under explicit mathematical premises.
- **Equal-Rate Non-Identifiability.** When shared and view-specific retention
  rates coincide ($r_c \approx r_{p,1} \approx r_{p,2}$), multi-time LP widths
  remain wide (`3.8343`, `0/60` branch disjointness in synthetic controls). Do
  not force a unique latent trajectory when rates are inseparable.
- **Disjoint Multi-Cohort Context.** Companion public-cohort analyses in
  `project-persona` (`RealPersonaChat`, `ESConv`, `PSE-Database`,
  `Bar-Anan & Nosek`) establish domain-bounded behavioral dynamics alongside
  honest null regimes (`0/252` RealPersonaChat self-report links; `ESConv`
  emotion relief `p >= 0.309`). Never merge cohorts or infer clinical status.

## 4. Verification Commands

Run the standalone liquid and Riccati-Möbius test suites from the repository
root:

```sh
pytest -q tests/test_liquid_*.py tests/test_riccati_mobius.py
```

And run the trilingual manuscript/supplement consistency checks with:

```sh
pytest -q tests/test_trilingual_theory_manuscript.py tests/test_trilingual_liquid_riccati_supplement.py
```
