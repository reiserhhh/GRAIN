# VM05: exact common eligibility for the missing neural/profile comparison

Round31 execution update,2026-09-07: the raw reference-free34-head stage is
implemented/tested (80bcb90) and its real supervisor90901 has started, waiting
for54 sensitivity conditions and the previous owner to finish. This is not an
actual ready-fit result yet. Output`vm05_raw_ready_primary_r1`, identity
`77d1d31511f51c54ffe98163835726d7ae9d68c71e5aff0ef8a4bdf676422f7c`.
The unchanged144-person Big5 profile was independently replayed across all5
saved folds (max8.33e-17), not refitted. See
[bounded raw stage](../../docs/GRAIN_VM05_RAW_READY_BATCH_V1.md). The historical
preflight report below remains separate from future empirical comparisons.

2026-09-06. **Label-free material and membership complete; comparator fitting
still pending.** This executes the original same-ready requirement, without
changing a frozen GRAIN support rule, opening bridge outcomes or using MEPS scales.

## Why this matters

The existing full-source sparse heads use624 Big5-only/6393 MBTI-only authors.
The frozen formal profile accepts only a subset. Comparing those full-source
scores to a ready-only GRAIN score would conflate the representation, population
and training sample. Even restricting a completed full-source OOF table to the
accepted test users is a different estimator from ready-only retraining.

The formal profile experiments already ran; do not rerun them merely because
the sparse/neural side was missing. Reuse their exact saved parents where their
cohort, head protocol and replay match. The new registry is the shared material
contract for completing TF-IDF/BGE/E5/Qwen and GRAIN-plus-vector comparisons.

## Verified populations

| Cohort | Big5-only | MBTI-only | Bridge, eligibility only |
|---|---:|---:|---:|
| Raw profile ready | 144 | 1566 | 62 |
| All three text arms ready | 143 | 1495 | 61 |
| Raw ready, historical reference excluded | 144 | 1539 | 62 |
| All arms ready, historical reference excluded | 143 | 1469 | 61 |

Only27 of the99 common-corpus historical reference members occur in raw-ready,
and26 in all-arms-ready. Subtracting99 from either ready sample would be wrong;
the exclusion is an author-set intersection, not arithmetic on headline counts.
No Big5-only or bridge eligible author is a reference member. Excluding those
members does not erase other historical exposure or create a new lockbox.

Across4 cohorts x5 tasks,840 membership points were reconstructed, including
outer folds, source-to-bridge provenance and retain/purge inner folds. The
smallest permitted training set after these operations is65 Big5 authors and486
MBTI authors. None falls below the existing dense-head minimum20. This is an
algorithm-support check only: no outcome values were read, so class presence,
target variance and inferential power remain unassessed here. Small Big5 samples
also imply an inner PCA cap sometimes as low as6; it is not an equal16-dimensional
readout in every fold. Report actual fit dimensions rather than hiding the cap.

Source groups and official task folds remain separate; bridge rows carry no
source training fold. Every author keeps all36 comments. Masks are derived from
actual ready statuses, not their reported counts, and all methods must share the
selected author IDs before any fitting or paired comparison.

## Fixed comparison interpretation

Primary g is the existing16 sorted landmark-distance profile, not a claim that
this vector instantiates the complete current GRAIN4W theory. The main internal
candidate remains Qwen block10 body mean/std. [g,q] versus g asks whether the
internal representation adds predictive information under the fixed reader;
[g,q] versus q asks whether g contributes anything back. [g,b] supplies the
generic-embedding counterpart. No validation/bridge result chooses weights or a
new layer; canonical geometry variants remain separately named sensitivities.

All scalers, PCA and supervised choices fit source-inner folds. Joint PCA can
privilege the larger block; a weak joint result is not automatically absence of
information. The predeclared no-PCA sensitivity remains separate. Comparisons
are about these representations and readouts, not a proof of LLM personality
reasoning. Full observer collection is required before neural comparisons.

## Evidence and next action

- Protocol and implementation frozen at750c6ac:
  `docs/GRAIN_VM05_SAME_READY_EXECUTION_V1.md`.
- `vm05_same_ready_registry_r1`: all7,266 material authors, private masks/folds,
  coverage.csv, training_support.csv and840 private membership records;
  identity `d125e4a3e6d5e20fabd71419bb9b31da5585517971a4f4857848aa92798dc079`.
- Preparation took3.65sec, zero model fits and zero neural forwards. Formal
  profile input artifacts are checksum-bound to their previously frozen outputs.
- `vm05_same_ready_registry_audit_r1`: independent explicit thread-set logic and
  RandomState/partition reconstruction, without the preparation selector. All12
  saved array fields,840 memberships,12 coverage rows and20 training-support
  rows match. Source/bridge outcome values were not read.
- Four synthetic tests cover exact cohort identity, mask type, official folds,
  accepted-profile finiteness, thin purges and independent splitter equivalence.
- Related regression selection:119 tests passed in33.48sec; two pre-existing
  SWIG deprecation warnings. Ruff passed. This was not the entire repository suite.

Private paths above are under`results/grain_vector_mechanisms/`. Run
`scripts/prepare_grain_same_ready_registry.py` with the frozen views/profile/
preflight directories, then`scripts/audit_grain_same_ready_registry.py` with the
same parents and a fresh audit directory. Do not publish author-level artifacts.

The comparator/increment runner is now implemented at`fae1273`, separately from
the active old source estimator. It supports the exact fixed sparse/neural and
concatenated feature objects, official nested source fitting, target-support
checks, source-only final head and mandatory saved-fold replay. All72 observer
shards must finish and pass checksum checks before a neural comparison.

Real label-free entrypoint preflights passed: `vm05_ready_big5_sparse_preflight_r1`
selects144 authors (6.37sec), and`vm05_ready_ei_sparse_preflight_r1` selects1,539
reference-excluded authors (9.88sec). Both record zero model fits, neural forwards
and source/bridge outcome conversion. They do not demonstrate predictive quality.
Twelve focused tests passed, including full synthetic Big5/MBTI output/replay;
the subsequent46-test ready/source/resource/domain selection also passed. Ruff
and whitespace checks passed; two historical SWIG warnings, not whole-repo tests.

Continue the running strong-source controls and full observer queue. Actual
same-ready refits and increments remain pending behind the single-Mac fit budget
and complete neural material. Audit those fits, then freeze source heads before
bridge evaluation. No source winner or program closure follows from preflight.
