# Constrained Riccati-Möbius Flows, Dynamic Set Inversion, and Transition-Contrast Updates

This standalone release promotes the verified mathematical upgrades in `grain_core.riccati_mobius`
(`suica_core/riccati_mobius.py`), accompanied by the formal Lean 4 machine-checked proofs and public
multi-corpus validations maintained in the companion `project-persona` repository (`docs/GRAIN_THEORY_LEAP_RICCATI_DYNAMIC_DUAL_CLOCK_20261124.md`, `docs/GRAIN_PERSONALITY_PSYCHOLOGY_THEORY_SYNTHESIS_20261124.md`).

## Core Instruments Added (`grain_core.riccati_mobius`)

1. **`ConstrainedRiccatiCell` — Exact Zero-Order-Hold (ZOH) Möbius Flow & $O(\log T)$ Parallel Scan**:
   - Integrates the constrained scalar Riccati ODE $\frac{dx}{dt} = \epsilon(u)(\alpha(u) + \beta(u)x + \gamma(u)x^2)$ via an exact normalized `2x2` hyperbolic matrix exponential.
   - Guarantees strict `[-B, B]` domain invariance, `dt = 0` identity preservation (avoiding discrete Euler/`CfC` zero-time defects), exact time-subdivision semigroup consistency (`< 1e-12` error vs `30.68%` 2x subdivision error in standard `LTCCell` baselines), and exact agreement between sequential rollout and associative Kogge-Stone parallel prefix scan (`< 1e-12`).

2. **`DualPathwayRiccatiObserver` — Fast Continuous Decay Paired with Slow Episodic Lookup**:
   - Pairs a contractive Riccati state over continuous inter-event delays with content-addressable cosine retrieval over prior utterance representations so early high-signal events remain addressable even across long empty gaps.

3. **`estimate_cross_view_lag1_rates` & `solve_axis_trajectory_lp` — Dynamic Multi-Time Set Inversion**:
   - Replaces midpoint-biased private AR(1) estimates with an unbiased cross-view Method-of-Moments estimator separating shared vs. view-specific retention rates $(r_c, r_{p,1}, r_{p,2})$.
   - Uses HiGHS linear programming across consecutive transitions to eliminate unbounded initial private offsets whenever $r_c \ne r_{p,i}$, while honoring the proven equal-rate degeneracy barrier when all three rates coincide.

4. **`stack_transition_contrasts` — Multi-Window Linear Contrast Operator**:
   - Eliminates arbitrary view-specific initial coordinates step by step and stacks calibrated window differences $A_t - P_{j,t}$ to recover shared latent trajectories when single-step differences are rank-deficient.

## Usage

```python
from grain_core.riccati_mobius import (
    ConstrainedRiccatiCell,
    DualPathwayRiccatiObserver,
    estimate_cross_view_lag1_rates,
    solve_axis_trajectory_lp,
    stack_transition_contrasts,
)
```

Verify the standalone liquid and Riccati-Möbius suites with:

```sh
pytest -q tests/test_liquid_*.py tests/test_riccati_mobius.py
```
