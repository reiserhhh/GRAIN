# GRAIN R16 cross-reader intervention-effect concordance and single-law test, protocol V1

Frozen 2026-09-26 JST before computation. Cached F16 `text_transport_full_r1` reader
scores on 380 synthetic materials in 95 templates. No neural calls.

- Interventions are matched within template:
  - pole flip, with qualification fixed;
  - qualification flip, with pole fixed. Subject status moves with qualification by
    design, as recorded.
- Reader-targets: NLI pole readout, GRAIN pole, NLI qualification, GRAIN qualification.
- Effect = score(after) - score(before) for each matched pair.
- Concordance: for each intervention and each target, the Pearson and sign agreement of
  NLI vs GRAIN effects across matched pairs.
- Single-law test: one law with reader-specific linear links predicts effects that are
  proportional across interventions. The test statistic is the normalized determinant
  D = det(E) / (||E||_F^2) of the 2x2 matrix, interventions by readers, of mean effects
  per target. CIs come from 2,000 template-bootstrap draws (seed 20260926).
- Rule: single-law compatible when the CI of D covers 0. Rejected otherwise.
- A zero-variance reader effect is reported as ABSENT_CHANNEL, not as a correlation.
