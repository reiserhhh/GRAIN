# VM05 full-source material and independent audit

Date: 2026-09-06. Status: material preparation and audit complete; external
personality prediction has not been run on this material.

## Source and eligibility

[Protocol](../../docs/GRAIN_VM05_SOURCE_EXPANSION_PROTOCOL.md) commit 0405492;
extractor a972f55. Read only prepared-table IDs and source-specific official folds;
no psychological outcome values were parsed. File hashes read bytes for identity,
not questionnaire values for analysis. One 112.091-second raw scan covered all
17,640,062 comment rows; peak Mac RSS was 5,304,123,392 bytes.

| Disjoint source | Source authors | Eligible authors | Comments |
| --- | ---: | ---: | ---: |
| Big5-only | 1,037 | 636 | 22,896 |
| MBTI-only | 8,678 | 6,453 | 232,308 |
| Strict bridge | 375 | 254 | 9,144 |
| Other both-label, reserved | 0 | 0 | 0 |
| Total | 10,090 | 7,343 | 264,348 |

Each eligible author has 36 unmodified source-file comments, allocated into three
strictly chronological 12-comment views from distinct discussions. All comments
are source-labeled English and 40-100 whitespace words. The fixed per-author
192-comment hash reservoir precedes selection. Insufficient distinct comments
exclude 2,747 authors; no score-based resampling or criterion relaxation occurred.

Big5-only official fold counts after filtering are 123/127/133/122/131. MBTI
folds remain axis-specific; all five folds are populated for every axis.
Bridge's own fold counts are 62/56/60/32/44; these are not substituted for the
official axis folds. The prepared-source union, not 10,090 independent complete
personality-label vectors, is the denominator.

## Important corrections and findings

**Bridge membership is not simply a source intersection.** Official Big5 and
MBTI files intersect at 364 authors; strict bridge has 375. The extra eleven
are outside BOTH prepared task-source files, not eleven Big5-only authors.
The earlier in-progress suggestion that they would contaminate the Big5 head
was incorrect. Explicit bridge-priority exclusion is a defensive invariant;
it did not change the observed Big5-only count (1,037), and no head was trained
from the earlier inventory. This is not evidence invalidating older corrected H2.
All eleven fail the present 36-comment eligibility, so the 254 bridge materials
also have official-task membership. Label completeness still awaits VM05 coding.

**No exact material leakage, but shared contexts remain.** Independent audit
verifies the material hash, complete source ledger, per-source folds, author
separation, chronological boundaries, zero repeated comment IDs and zero
normalized duplicate bodies across authors. Fourteen cross-author text keys
were removed within retained reservoirs, not by scanning all possible near
duplicates. There are 2,041 discussion threads shared between source groups,
including 1,096 shared between bridge and a head-training source. Different
comments in the same discussion are not identical samples, but correlated
context and quotations require an explicit sensitivity analysis.

**Explicit personality-related cues differ by source.** The automatic flag is a
clue, not a confirmed label report; it combines type terms, trait terms and report
cues and can over/under-detect. Counts below count a comment once if any flag fires.

| Source | Flagged comments | Authors with at least one flag |
| --- | ---: | ---: |
| Big5-only | 216/22,896 (.94%) | 146/636 (23.0%) |
| MBTI-only | 1,928/232,308 (.83%) | 1,158/6,453 (17.9%) |
| Bridge | 441/9,144 (4.82%) | 142/254 (55.9%) |

This imbalance could make bridge linkage appear easier through self-description
or report terms. Retain original text for one arm; a matched cue-masked arm and
within-budget, shared-thread sensitivity must precede a denoising/personality
mechanism claim. Do not silently remove whole authors or choose replacement text
after reading outcomes. Source-file raw text can itself contain upstream cleaning
placeholders; unmodified here does not mean untouched human originals.

## Coverage limits and next execution

Median observed author span is about 1,170 days, not a contemporaneous session.
Chronological views test accumulated-history correspondence, not pure stable
trait or known changes relative to questionnaire time. Current three-view data
also contain 239 of the old 240 mechanism authors; this is not a new lockbox.
The single missing old author is not replaced by relaxing criteria. Active-user,
English, length and discussion diversity selection limit representativeness.

Next freeze the VM05 same-evidence protocol before reading outcomes: local
TF-IDF, fixed encoder vectors and pre-answer LLM vectors, train-fold-only
projection/readout fitting, full source-specific CV, and bridge-only posterior
evaluation. Apply the same retained comments to each representation. Preserve
GRAIN's frozen applicability gate and label any unrestricted intermediate
features as such; do not compare 12-comment LLM against 36-comment GRAIN as
representation ability or use new results to replace historical headline scores.

Canonical ignored artifacts under `results/grain_vector_mechanisms/`:

- `vm05_membership_inventory_r2`: explicit bridge-priority source inventory;
  `r1` preserved, current 240-author subgroup counts unchanged.
- `vm05_full_source_cohort_r1`: all-author eligibility, fold/source counts,
  per-source and combined raw views, source identities, code snapshot, runtime.
  Material SHA256: `69d67f463e6b04dbcde41b0f1b0502d67b9b9b4a1521ef76962c0cf83b092e75`.
- `vm05_full_source_audit_r1`: independent checks, flags/coverage, exclusions
  and aggregate source-overlap counters. No psychological outcomes.

Public git contains only implementation, tests, protocol and aggregate report.
Participant text, identifiers, membership ledgers and caches remain ignored.
