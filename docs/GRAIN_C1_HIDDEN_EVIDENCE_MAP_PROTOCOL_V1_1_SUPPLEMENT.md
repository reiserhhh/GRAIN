# GRAIN-C1-01 protocol V1.1 supplement (r2): unanchored-nuisance projection

Frozen 2026-10-02 ~19:55 JST after r1 (`results/grain_construction_v1/C1/hidden_map_r1/`) and
before any r2 output. Everything in V1 (`docs/GRAIN_C1_HIDDEN_EVIDENCE_MAP_PROTOCOL_V1.md`) applies
unchanged: worlds, sample sizes, information boundary, kernel, frozen tolerances (anchor |t| >= 4,
design R^2 > .5 flag, tau_T = max(.15, 2 SE), tau_F = .20, nonconstancy), interventions and the
acceptance rule. r1's verdict is final for r1 and is not revised by r2.

## Why (post-r1, disclosed)

In r1 no arm passed W2 (Delta) in 3/3 seeds. The anchored arms failed on the style check:
the author-by-context register shift leaked into the conditional contrast (R_F style .04-.34).
The r1 stability arms recovered exactly that nuisance (unanchored, |t| < 4). This supplement tests
one construction rule motivated by r1, on fresh seeds so the rule is not tuned on r1 outcomes.

## Rule (estimator-side only)

Source data identify stable between-author structure; the pilot identifies which of it is the
target. Take the top 4 author-level and top 2 author-by-context (slope) generalized eigen-filters
of source between-author covariance (V1 definitions). A filter whose pilot calibration |t| < 4 is
unanchored; its source activation pattern (between covariance x filter, unit norm) enters a
nuisance set N. The supervised readout direction is projected onto the orthogonal complement of N,
then unit-calibrated on the pilot as in V1. Arms: `np_ridge` (base pilot_ridge, same lambda rule)
and `np_pcr` (base pilot_pcr, same k rule). Bootstrap: N fixed, base refit, unit re-estimated.

## Other changes

- Fresh seeds 20261005, 20261006, 20261007. All V1 arms are rerun (replication) plus the two new
  arms.
- Defect fix found in r1: in W3 the generic largest-loading sign rule overrode the public lexicon's
  declared signs for `keyed`. r2 keeps the declared lexicon signs (`--keyed-declared-sign`). r1
  outputs are unchanged and replay byte-identically under the current code with defaults.

## Decision use

If an np arm passes W1 and W2 in 3/3 fresh seeds, the construction rule "remove stable source
structure the anchor cannot claim" is supported in this declared linear world, and the
r1 failure is located (unanchored context-responsive fingerprint), not hidden. If not, the
failure stands and C2 must anchor context-responsive nuisance directly in the pilot design.
