# Amendment 1 to GRAIN_R09_G_COEFFICIENT_PROTOCOL_V1 (append-only; V1 text unchanged, sha 77e37e56…)

2026-09-26 22:26 JST. This amendment was written after attempt 1 stopped at the pre-computation composite check. No
coefficient had been computed.

- **Scope of the balance check.** The requirement "w' X β must not depend on the fixed wording effects" applies to the
  48-event composites, where all 3 wordings are balanced.
- **Single-wording half.** The half (occasion 1, wording 1) cannot average over wordings. Its level and cell-mean
  composites load the fixed effect g_{1,k} with weight 1. This is a person-constant offset:
  - it is recorded in `composite_checks.json`;
  - it is excluded from every variance term;
  - under absolute decisions with fixed wordings it is a known constant bias of the universe mean, not error.
- **Unchanged.** The universe-score loading check (w' Z_θ = target loading) still applies to every composite.
  Composites, formulas, inputs, plug-ins and labels are unchanged.
- **History.** Attempt 1 is kept as `F04/generalizability_coefficients_r1_failed_attempt1/FAILED_REASON.md`.
