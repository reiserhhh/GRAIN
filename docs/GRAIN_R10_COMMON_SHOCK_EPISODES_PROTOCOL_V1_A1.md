# R10 common-shock episodes protocol V1, amendment A1: run settings from smoke timing only

Date 2026-09-26 JST. V1 (sha 5179228b…) is unchanged.
- **Checks.** `checks.json`: status PASS in 6.9 s. The archived replay, zero-common equivalence, no-future,
  independence and dense-filter checks all passed before any new episode output existed.
- **Smoke.** `smoke_timing.json`: 1 episode per world took 5.1 s. Outputs were discarded, and no scientific output
  was inspected. The projection is 2.7 min on 2 workers.
- **Frozen full run.** E = 32 episodes per world (seeds 20261001–20261032), both worlds, all budgets, 2 workers ×
  1 BLAS thread. No design element changes.
