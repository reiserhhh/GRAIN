# GRAIN R17 natural-material geometric sensitivity, protocol V1

Frozen 2026-09-26 (JST) under `reports/claude_research_loop_20260926/SEMINAR_FABLE_03_DECISION.md`
(coordination repo), Packet 1. It is written before any reader scores on the material.
The synthetic precursor in `GRAIN_R17_EDIT_BATTERY_PROTOCOL_V1.md` stays deferred; this
protocol does not run it.

## Scope and wording

- Three declared **mechanical** edits of existing natural units. Mechanical edits do not
  certify unchanged psychological content. There is no construct contract, so every
  result is labelled **geometric sensitivity only**. No personality selectivity claim.
- R17 remains PARTIAL after this packet: construct edits, LLM readers and human
  confirmation are absent.
- The F16/F11 synthetic plan-support head is NOT applied to PANDORA or Essays.

## Material (metadata first, no new collection)

- Units and cached reads come from
  `results/grain_vector_mechanisms/all_plan_closure_v1/local_bilinear_r2/{pandora,essays}`.
  PANDORA unit = one comment of the 36-comment Big5-only author material (local restricted
  copy, research use, no redistribution). Essays unit = one 128-whitespace-token chunk of
  the single essay (local research copy). No MEPS, no X, no labels are read.
- Only held-split authors (PANDORA official fold 0, Essays fold 9), so the historical heads
  never saw these authors.
- Author order: seeded permutation (SeedSequence([20260926, corpus code])). At most one unit
  per author, 50 authors per corpus, 100 base units in total.
- Unit within author: the first unit in address order that is eligible:
  1. at least 12 whitespace tokens;
  2. all three edits applicable (below);
  3. base and every edit at most 510 tokens under both tokenizers (E5 with its `query: `
     prefix), so no version is truncated and none displaces original evidence;
  4. a natural partner unit exists: the next other unit of the same author in address
     order (wrapping), with at least 12 whitespace tokens, at most 510 tokens under both
     tokenizers and a different text hash.
- Every scanned author and every rejected unit is recorded with its first failing reason.
  The unit definition is never changed to reach a count. If fewer than 50 eligible authors
  exist, the shortfall is reported.

## Declared edits and intended (uncertified) invariance

| Edit | Operation | Applicable when | Retention check | Intended invariance (assumed, not certified) |
|---|---|---|---|---|
| case_fold | lower-case the unit | any character changes | case-folded whitespace-token multiset identical | capitalisation as presentation |
| contraction_expand | fixed 27-entry unambiguous map (n't, I'm, 're, 've, 'll forms; no 's or 'd) | at least one match | text outside matched spans identical | contraction as register |
| sentence_reverse | split after [.!?] + whitespace, reverse order, join by one space | at least two segments with at least two word tokens each | whitespace-token multiset identical | sentence order inside a unit |

All versions are validated (applicability, retention, token counts, truncation) before any
model call. Refusals are recorded.

## Readers and coordinates

- Neural readers, both already cached locally, offline, no download:
  - `intfloat/e5-large-v2` rev f169b11e, prefix `query: `, attention-mask mean, L2, MIT;
  - `BAAI/bge-large-en-v1.5` rev d4aa6901, no instruction, CLS, L2, MIT.
  Device MPS, float32, one item per forward (no padding effects).
- Coordinates:
  - raw geometry of each reader (1 - cosine);
  - E5 frozen IPIP-NEO-300 keyed facet projection (30 facets, cached item vectors, no
    outcome fitting), plus the five domain means;
  - GRAIN deterministic lexical runtime (24 coordinates, no neural call);
  - supported historical readouts: the source-selected Big5 ridge heads `semantic` (E5
    author-mean) and `grain` (GRAIN mean and SD) of local_bilinear_r2, applied by
    single-unit substitution in the author block. Provenance: fitted on each corpus's
    source split with self-report labels. They are reported separately and are not
    personality scores for the edited unit.

## Forward budget (enforced before each call)

- Hard cap 1,000 item forward attempts across both readers, counting base, edit, partner,
  smoke and any failed attempt. Items are counted, not batches.
- Planned: E5 = 100 x (base + 3 edits) = 400; BGE = 100 x (base + 3 edits + partner) = 500;
  total 900. The E5 partner comes from the cached compatible E5 read, and the recomputed
  base measures device drift against that cache.
- The first smoke call is part of the run and of the cap. No automatic retry. A failed or
  interrupted item stops the run and is recorded.

## Comparisons (after all forwards, no further calls)

Per corpus, reader-coordinate and edit, with one unit per author (author clustering):

- edit distance per unit; within-author natural distance (base vs partner); between-author
  natural distance (mean distance of the base unit to the other base units of the corpus);
- ratios of means edit/within and edit/between with 2,000-draw author bootstrap
  (seed 20260915); share of units where edit > within;
- systematic share ||mean delta||^2 / mean ||delta||^2;
- signed mean change per standardised coordinate (between-author SD units) with CI;
- device-drift floor for E5 (local base vs cached base);
- generalized eigenproblem Sigma_B v = lambda Sigma_N v on the 30-facet and GRAIN
  coordinates, Sigma_B = between-author covariance of base units, Sigma_N = second moment of
  declared-edit deltas plus 1e-3 trace ridge. Sigma_B is natural between-author variation,
  not certified construct variance.
- readout deltas in held-out prediction SD units: edit substitution versus natural partner
  substitution.

## Decision rule (descriptive)

- An edit is called geometrically small for a reader-coordinate when the upper 95% bound of
  edit/within is below 0.5; comparable when the interval covers 1; large when the lower
  bound exceeds 1. Otherwise intermediate.
- No result upgrades R17 beyond PARTIAL.

## Amendment A1 (2026-09-26 03:50 JST, after a 2-item smoke, before any comparison)

- Finding: both readers use uncased BERT tokenizers (`do_lower_case: true`). `case_fold`
  therefore yields identical input IDs, and the smoke confirmed a bitwise-identical E5
  vector for the one case_fold item already forwarded. This is invariance by
  preprocessing, not a model property. The pre-scoring validation should have compared
  input IDs; that omission is recorded here.
- Change 1: for every case_fold item, input IDs are compared with the base before any
  call. Identical IDs reuse the base vector with status `TOKENIZER_IDENTICAL` and cost
  zero forwards. Non-identical IDs are forwarded and counted. The one forwarded smoke item
  stays in the accounting.
- Change 2: a fourth declared edit, `punctuation_strip`, is added (the decision allows at
  most four edits per unit). Operation: replace every character outside word characters,
  whitespace and apostrophes with a space, then collapse runs of spaces and tabs.
  Retention: the `\w+` token sequence is identical. Intended (uncertified) invariance:
  punctuation as presentation; exclamation and question marks can carry affect, so this
  is not certified content-neutral. Applicability and truncation are validated for all
  100 frozen units before scoring. The frozen eligibility rule and ID manifest do not
  change; a unit where punctuation_strip is not applicable or fails validation is
  recorded as a refusal for that edit only.
- Budget: planned E5 = 100 x (base + contraction + reverse + punctuation) + 1 smoke
  case_fold = 401; BGE = 100 x (same + partner) = 500; total 901 under the unchanged
  1,000 hard cap. Per-reader ceilings 500 (E5) and 600 (BGE) guard any non-identical
  case_fold item; the total cap binds first.
- GRAIN lexical coordinates for case_fold are computed and reported as they are.
