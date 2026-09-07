# S17: source continuation of the fixed second-order conditional expansion

2026-09-08. After S16, freeze the discovered formula before computing the other
189 source authors. This is a source-development continuation, not a newly
untouched population or a release benchmark.

## Exact scope

Same213 N source authors,3views, same S14 author-out common queries, S12 KV/PCA C
readers, prompt/model hashes and full attention support. Reuse the first24 S16
packets; collect only the remaining189,567 new prefix calls. No new model fit,
reader calibration, query choice, feature selection or human outcome selection.

Compare three fixed readouts:
exact pairing; exact comment mass times conditional covariance; and exact mass
times conditional covariance plus half the second directional derivative.
For each comment the latter is Cov(V,L)+E[(V-EV)(L-EL)^2]/2, L=q^TK.
This is second order in query strength, a third joint cumulant in (V,K,K).
The finite-exponential-family derivation is in theory V4 and the S16 report.

## What determines the next step

Report fixed-formula reconstruction r, MSE/exact-author-variance and bias on
new189 and all213 authors, retaining all three arms. Then freeze all source
scores before connecting already-exposed N outcomes. Report teacher-C fidelity,
source r, and paired source delta CI versus exact reading; never choose a formula
using those outcomes. Preserve T/F and the exact N pathway.

If second order recovers internal response without improving external association,
retain it as a mechanistic approximation, not an accuracy breakthrough. If it
also preserves the source connection, it supports a compact query-specific
statistical readout. If it fails beyond the pilot, locate the conditional
distribution regime rather than fit unrestricted higher orders automatically.

## Resource and provenance

567 new prefixes maximum,0 full suffix/backward calls,15-minute collection cap.
One neural collector. Reused72 prefixes remain attributed to S16; do not count
them again as new computation. Replays of exact and first-order arms must match
S15 within1e-8. Do not modify frozen S12-S16 code or open bridge/MEPS/lockbox.

Output `results/grain_vector_mechanisms/second_conditional_s17_r1/` (ignored).
Report `reports/grain_vector_mechanisms/S17_SECOND_CONDITIONAL_REPORT.md`.
Save all source hashes, cache/collection membership, before-outcome scores and
actual cost. No new paper or psychological factor-name claim is part of this run.
