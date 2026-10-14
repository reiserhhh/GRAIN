# R24 personal-law protocol V1, amendment A1: full design frozen from smoke timing only

Date 2026-09-26 JST. V1 (sha 5df0cdb0…) is unchanged.
- **Smoke.** `smoke_timing.json`: 1 seed, 20 application authors, 20 refits, mechanism none. It took 0.95 s per
  world, about 0.047 s per author-mechanism including source-validation scoring. No scientific output was
  inspected.
- **Projection.** Cost scales roughly linearly with the refit count. For 3 worlds × 3 seeds × 200 authors × 4 masks
  at 200 refits, the projection is about 15–20 min on 2 workers.
- **Frozen full design.** The V1 default target: seeds 20260915–17, all four masks (EB for none and MCAR only),
  200 source refits per world/seed, 200 application and 100 source-validation authors.
- **Cap rule.** If the 45-minute cap is reached, partial per-world/seed outputs are kept and nothing is enlarged.
