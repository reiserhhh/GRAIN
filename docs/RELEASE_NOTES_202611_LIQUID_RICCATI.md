# GRAIN 2026-11 Liquid Trajectory & Constrained Riccati-Möbius Notes

**Date:** 2026-11-25  
**Package version compatibility:** `grain_core.__version__` remains `1.1.1` to preserve existing release tags and brand contracts verified by CI (`tests/test_grain_brand.py`).

## Scope of Additions on `main`

This release note indexes the additive updates on `main` plus their accompanying documentation and trilingual theoretical synthesis:

1. **Frozen Liquid Trajectory Suite (`grain_core.liquid_*`):**
   - Native deterministic node-map readers and optional continuous-time/common-range tools (`liquid_instrument`, `liquid_dialogue`, `liquid_common_range`, `liquid_event_time`, `liquid_task_paths`, `liquid_reference_error`, `liquid_opportunity`, `liquid_axis_series`, `liquid_frozen_bank`, `liquid_inverse`, `liquid_v6_author`).
   - Practitioner manual: [`docs/GRAIN_LIQUID_TOOL.md`](GRAIN_LIQUID_TOOL.md).
2. **Constrained Riccati-Möbius Flows & Dynamic Set Inversion (`grain_core.riccati_mobius`):**
   - `ConstrainedRiccatiCell`: exact normalized `2x2` Zero-Order-Hold (ZOH) Möbius matrix exponential guaranteeing `[-B, B]` domain invariance, `dt = 0` identity preservation, time-subdivision semigroup consistency (`1.73e-18`–`2.52e-18` 2x subdivision error vs. `30.68%` in baseline `ncps LTCCell` on 108 real trajectories), and associative $O(\log T)$ Kogge-Stone parallel prefix scan agreement (`<= 1e-12`, observed `1.388e-17` median/max).
   - `DualPathwayRiccatiObserver`: couples fast contractive decay across idle intervals with content-addressable cosine retrieval, recovering long-horizon opening-event sensitivity (`wall_first_at_end_median = 2.710e-4` vs. `0.0`).
   - `estimate_cross_view_lag1_rates`, `solve_axis_trajectory_lp`, and `stack_transition_contrasts`: clipped cross-view lag-1 moment estimates feeding HiGHS dynamic trajectory polytopes (`4.949 -> 0.1070`, `0.0216x` width under `distinct_rates`) while explicitly exposing the equal-rate non-identifiability barrier (`3.8343`, `0/60` disjoint) and conformal infeasibility (`None`).
   - Practitioner manual: [`docs/GRAIN_RICCATI_AND_INTERPERSONAL_UPDATE.md`](GRAIN_RICCATI_AND_INTERPERSONAL_UPDATE.md).
3. **Trilingual Supplement V2, Lean 4 Proof Correspondence, & Analyst Guardrails:**
   - Trilingual Supplement V2 (`EN`, `ZH`, `JA`) + Evidence Provenance + Crosswalk Section 8: [`docs/paper/GRAIN_LIQUID_RICCATI_SUPPLEMENT_EN_V2.md`](paper/GRAIN_LIQUID_RICCATI_SUPPLEMENT_EN_V2.md), [`docs/paper/GRAIN_LIQUID_RICCATI_SUPPLEMENT_ZH_V2.md`](paper/GRAIN_LIQUID_RICCATI_SUPPLEMENT_ZH_V2.md), [`docs/paper/GRAIN_LIQUID_RICCATI_SUPPLEMENT_JA_V2.md`](paper/GRAIN_LIQUID_RICCATI_SUPPLEMENT_JA_V2.md), [`docs/paper/GRAIN_LIQUID_RICCATI_EVIDENCE_PROVENANCE.md`](paper/GRAIN_LIQUID_RICCATI_EVIDENCE_PROVENANCE.md), [`docs/paper/GRAIN_THEORY_MANUSCRIPT_CROSSWALK.md`](paper/GRAIN_THEORY_MANUSCRIPT_CROSSWALK.md).
   - Companion Lean 4 verification (`58` theorems across 7 modules, `0` `sorryAx` in companion `project-persona`) and bounded validation across four disjoint public cohorts (`RealPersonaChat`, `ESConv`, `PSE-Database v1.0`, `Bar-Anan & Nosek 2014`) including honest null results and non-clinical boundaries.
   - Analyst guardrails G13–G16 added to [`docs/AI_ANALYST_GUIDE.md`](AI_ANALYST_GUIDE.md).

4. **Addressed events and two-view decisions (2026-11-25).** [Usage and examples](GRAIN_EVENT_AND_VIEW_INCREMENT.md). Both commands run on a fresh checkout without a corpus, weights, or API access. Full-text inference beyond the explicit correction patterns requires separately coded evidence. Joint path bounds are compared against both observed histories, not just the current midpoint. Cite the commit for these untagged additions to `main`.

## Compatibility & Integrity Guarantees

- All frozen legacy artifacts (`v0.2.0`, `v0.2.1`, `v1.1.0`, `v1.1.1`) and Version 1 trilingual manuscript locks (`tests/test_trilingual_theory_manuscript.py`) remain untouched.
- Verification suite:

```sh
pytest -q tests/test_liquid_*.py tests/test_riccati_mobius.py tests/test_trilingual_theory_manuscript.py tests/test_trilingual_liquid_riccati_supplement.py
```
