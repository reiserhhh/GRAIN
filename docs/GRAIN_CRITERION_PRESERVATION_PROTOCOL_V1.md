# GRAIN criterion-preservation analysis, protocol V1 (seminar-05 queued item 2)

Frozen 2026-09-26 JST before any label value is read. It uses existing cached, previously
used, public-corpus numeric labels only: PANDORA Big5 (continuous 0-1) and Essays Big5
(binary). No MEPS, no lockbox, no new neural scoring, no fitting to evaluation labels, no
model, threshold or trait selection. **Exploratory reuse, not confirmatory.** Every label
access is logged before it happens.

## Eligibility (checked before labels)

- F12/heldout_map_r1 (GRAIN to NLI): acts on synthetic F16 items only. **INCOMPATIBLE**
  with labelled persons, so it is not analysed.
- F12/encoder_gauge_r1 (BGE to E5 maps): eligible. The maps act on the R17 units of 50
  held PANDORA and 50 held Essays authors, all labelled.
- F10/shortcut_r1 and FU1-c (topic-residualized historical E5 semantic head): eligible.
  The 120 held PANDORA authors are labelled.

## Part A: criterion preservation under the fitted encoder map (R21/R23)

- For each R17 author, form the 2-unit mean embedding (base plus partner):
  - E5 side: cached E5 base and partner vectors.
  - BGE side: the BGE base and partner vectors mapped to E5 space by the map fitted on
    the OTHER corpus's items (as in encoder_gauge_r1). Minimum-norm least squares is the
    primary map, declared in advance. Procrustes is also reported.
- Substitute the 2-unit mean into the author's block row and score it with that corpus's
  frozen historical semantic head (local_bilinear_r2).
- Criterion: Pearson r with each of the 5 traits (point-biserial for Essays), per corpus.
- Report r_E5, r_mapped, delta r, and the correlation of the two predictions. Paired
  author bootstrap, 2,000 draws, seed 20260926. All 5 traits, n = 50 per corpus. Missing
  labels are counted and reported.

## Part B: criterion preservation under topic residualization (R31)

- Held PANDORA authors (120).
- The original historical semantic-head predictions are compared with their
  leave-one-out topic-residualized versions: the FU1-c construction with 5 PCs of top-30
  subreddit shares, re-computed identically.
- Report r with each trait for original vs residualized, delta r, and a paired author
  bootstrap CI.

## Reading

- Descriptive only. A preserved criterion relation under the map or residualization is
  consistent with functional agreement at the criterion level. A drop shows what the map
  or residualization removes.
- n = 50 per corpus gives wide intervals, which are reported as they are.
