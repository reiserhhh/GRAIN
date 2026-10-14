# R17 nuisance vs construct edit battery protocol v1 (frozen 2026-09-26 before running)

Worker: Claude Opus 5.5. Source demand: P01 pp.11-14 and 23, and P02 p.20.
Compare nuisance edits with construct edits across readers, then solve
Sigma_P v = lambda Sigma_N v. This first packet uses the 80 source-role F16
text-transport materials, which are controlled synthetic texts. Real-corpus
texts are a follow-up.

## Edits (deterministic string edits; frozen list)

**Nuisance** (author stance and ownership unchanged):

| edit | change |
|---|---|
| N_prefix | prepend "Just to note: " |
| N_length | append " The room had been booked for an hour." |
| N_order | move the final context sentence to the front |
| N_paraphrase | supported -> backed, rejected -> turned down, everywhere |

**Construct:**

| edit | change |
|---|---|
| C_pole | supported <-> rejected everywhere |
| C_own | actual <-> quoted frame, on actual and quoted materials only |

- C_pole flips the author stance on owned statuses (actual, negated).
- On unowned statuses (quoted, hypothetical), C_pole flips only non-owned
  content. It is scored as a separate "unowned flip" check, where a correct
  author reader should not move.

## Readers (no training; no API)

- **GRAIN:** the historical lexical runtime (vectorizer + SVD, no neural
  forwards) plus the F11 readout head, producing grain_pole and
  grain_qualification.
- **NLI:** the pinned local cross-encoder/nli-deberta-v3-base, with
  local_files_only and offline mode, and the F08 queries.
  pole_readout = p_ent(pos) / (p_ent(pos) + p_ent(neg));
  nli_qualification = p_ent(qualification).

## Budget

- At most 1,500 NLI pair forwards, checked before the call.
- Expected: (80 base + 400 edited) x 3 queries, about 1,440.
- The base texts are re-read and must replay the archived `reader_scores.csv`
  (GRAIN exactly; NLI to 1e-4).

## Metrics

- Per reader and score, mean |delta| under each nuisance edit and under
  owned construct edits.
- CSNR = mean |delta_construct| / mean |delta_nuisance|.
- Unowned-flip movement.
- The generalized eigenproblem on the GRAIN 24-d features:
  - Sigma_N = covariance of feature deltas under nuisance edits;
  - Sigma_P = covariance of feature deltas under owned C_pole;
  - Sigma_N is ridged by 1e-6 * trace / 24.
  - Report the top eigenvalue and the cosine of its eigenvector with the
    GRAIN pole readout direction.

## Scope

Synthetic templated texts and two readers. There is no human confirmation of
nuisance status (human-stage) and no real-corpus edits yet.

## Pre-run correction (2026-09-26 03:36 JST, before any forward; not executed)

- **Dry count (no model call):** 80 base + 440 edited texts = 520 texts; x 3
  queries = **1,560 NLI pairs**.
  - The 440 edited texts are 320 nuisance, 80 C_pole and 40 C_own.
  - The "expected ~1,440" above was wrong. The frozen cap of 1,500 would
    correctly refuse the run.
- **Proposed amendment, pending the seminar and supervisor:** cap 1,560 NLI
  pair forwards.
  - Model: `cross-encoder/nli-deberta-v3-base` at revision 6c749ce3, local,
    offline, CPU, 4 threads.
  - The GRAIN reader needs no neural forwards.
- **What it would satisfy:** only a controlled synthetic precursor of R17.
  That is edit mechanics, CSNR per reader, the unowned-flip check, and
  Sigma_P v = lambda Sigma_N v on the GRAIN features.
- **What it cannot satisfy:** the real-text edit battery (P01 p.23, "existing
  data"), LLM/vector/supervised readers, and human confirmation of nuisance
  status. R17 stays open after it.
- Status: NOT RUN.
