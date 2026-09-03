# SUICA v0.2.0 Release Notes

Release date: 2026-07-15

> Historical release under the former project name. The current method is
> GRAIN; the release identity and scientific record remain unchanged.

## Ruling

`V7_THEORETICAL_CORE_CLOSED_WITH_EMPIRICAL_GATES`

Version 0.2.0 is a technical-method release, not a completed personality
instrument. It moves SUICA from named low-dimensional factor claims toward a
frozen operator-indexed, reference-relative geometry with explicit refusal and
transport boundaries.

## Closed in this release

- G1 implements a content-bound, identifier-free landmark profile with
  tie-safe landmark orbits and radial-envelope refusal. Held-out ready rates
  were 98.6% discovery, 92.2% calibration, and 79.2% confirmation.
- W2 separates relative nonlinear geometry from linear coordinate recovery;
  in the nonlinear calibration world, distance Spearman was .393 while linear
  transport R2 was .068. Context residualization is not universally valid.
- W3 uses 25 random partitions; chronological splitting did not outperform
  ordinary source sampling on the tested PANDORA cohort.
- W4/W4b finds distributed multiview capacity rather than a small confirmed
  factor count. Confirmation effective rank was about 22.8--22.9; this is a
  capacity descriptor only.
- W7 demonstrates that unknown rotations/affine maps require paired alignment;
  target-local reference fitting is the default without a bridge.
- A portable, deidentified evidence bundle and content-hash lockbox replace
  machine-local ignored-result dependencies.

## Still open

Reliability, MDD, state-trait decomposition, external personality validity,
same-person cross-context generalization, clinical utility, language/culture
invariance, and criterion prediction all require future data. Anonymous V7
coordinates remain unnamed until those gates are passed.

## Compatibility and governance

The v0.1 historical P5 opening remains unchanged: overall 2/7 failed the
predeclared 4/7 rule, H2/H6 passed individually, and one opening remains.
Quick/smoke reports are development diagnostics; corrected full reports and
the tracked v0.2.0 evidence bundle are authoritative for this release.

Release audit: Python compilation passed; full suite 302/302 passed; V7 and
release-lockbox subset 95/95 passed. Exact package versions are recorded in
`requirements-lock-v0.2.0.txt` and `release/v0.2.0/TEST_REPORT.md`.

## Corrections (2026-07-17)

The body of this document above is the v0.2.0 record and is left unedited;
corrections are appended here per house style.

- **Effective-rank range corrected: "22.8--22.9" should read "22.7--22.9".**
  The W4b artifact value for the WORD12 view is 22.7496, which rounds to 22.7;
  the shipped range understated the low end. All other W4b numbers are
  unchanged.
- **Rank bridge (TGEO-P3 rank-4 vs W4b effective rank ~22.7--22.9 — different
  objects, no conflict).** No V7 document had mentioned TGEO-P3's rank-4
  result, leaving readers free to misread W4b as overturning it. They measure
  DIFFERENT objects in different feature spaces: TGEO-P3 found exactly 4
  supra-edge components in the frozen 19-construct battery's person-level
  covariance against a Marchenko-Pastur (empirical within-column-shuffle)
  edge, all 4 replicating across person-disjoint halves (congruence
  .985/.980/.932/.939). W4b's effective rank ~22.7--22.9 is a TF-IDF/SVD-24
  multiview CAPACITY DESCRIPTOR on comment slices — a distributed-capacity
  summary of a high-dimensional text representation, explicitly "a capacity
  descriptor, not a finite factor count" (V7.3-W4 ledger row). Neither result
  overturns the other; TGEO-P3's ledger row is unedited and stands.
