# VM02 Independent Multi-comment Cohort

Fixed before extraction on 2026-09-06. This extends the original VM02 plan;
it does not reset historical PANDORA exposure or open MEPS questionnaires.
This is an internal execution specification, not an independently registered
preregistration or a new corpus lockbox.

## Material and Split

- Select up to 1,024 candidate official author IDs by a new deterministic hash;
  exclude the previous 8 technical and 24 pilot authors. No label-value selection.
- Stream raw comments once. Eligible material: English, 40-100 whitespace words,
  valid timestamp and discussion ID. Preserve body, quotation, source and time.
  This narrower length band is an explicit computational/material restriction,
  not a representative sample or a linguistic denoising treatment.
- Retain a 192-comment hash reservoir per candidate. Remove normalized identical
  text shared across candidate reservoirs. Select 36 distinct discussions and
  bodies, then split chronologically into three independent 12-comment views.
- First 240 eligible hash-ordered authors: 144 discovery, 48 calibration,
  48 mechanism holdout. Discovery comprises 48 reference-fit and 96 comparison
  authors. Save split-specific files so default analyses do not open holdout.
- Report eligibility, time spans, text volume, source coverage and report-clue
  flags. Corpus-level near-duplicate absence is not established by exact checks.

## Comparison and Decisions

1. Frozen GRAIN unit projection, complete author mean/std geometry and canonical
   residual signatures, with the frozen minimum-unit and radial support rules.
   Unsupported scores are marked, never imputed as valid. Using raw bodies is a
   new observation design, not an exact rerun of the historical cleaned operator.
2. Reference-only word/char TF-IDF and format comparators. Frozen BGE/E5 record
   encoders followed by independent pooling vs context-joint record composition.
   All methods consume the same complete comments. Record all token lengths;
   no silent truncation. Different model receptive fields remain a limitation.
3. Whitebox Qwen history and decision states at the already-inspected sites;
   do not select layers on calibration/holdout. Context memory interventions and
   label-linked readouts remain subsequent packages, not replaced by matching.
4. Primary diagnostic: same-author rank against impostors across the three
   views, with paired author-node bootstrap. It is not personality prediction
   AUC. Include time/interleaving, clue and source/length sensitivities.
5. Freeze at most two channels after discovery and calibration before mechanism
   holdout. A calibration or holdout result must not silently become discovery.

Full GRAIN applicability and equal-material comparison are the immediate gates.
No fixed author-rank threshold constitutes psychological validity. The next
scientific step remains natural relational evidence and external linkage, not
optimizing author re-identification indefinitely.

Implementation clarification before the corrected comparison: chronological
reference vocabularies and scalers remain fixed for interleaved views. Token
spans are identified from explicit constructed character intervals, so a short
comment quoted inside another comment is not ambiguous or silently excluded.
The executed full GRAIN scorer refuses all 12-comment views on radial support;
see the [result report](../reports/grain_vector_mechanisms/VM02_MULTICOMMENT_DISCOVERY_REPORT.md).
