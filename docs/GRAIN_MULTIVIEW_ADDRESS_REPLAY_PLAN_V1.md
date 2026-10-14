# Full Multiview and Original-Address Replay V1

2026-09-13. Frozen before current-material results. All-plan C00/C06, with the
source-address interface needed by C03/C14/C16. Reconcile and exercise an actual
historical GRAIN construction, not another semantic regression grid.

## Purpose

Recent readers used 24 lexical unit coordinates, not the complete V7 multiview
operator. Reopen the E1/E4/G1/W4 evidence with checksums and replay the frozen E1
multiview model on the existing 624-author material. Keep shared reconstruction
and remaining variation separate. Add a lossless original-character sidecar
without changing any historical unit text, model, scorer, threshold or result.

## Material and Protocol

- Only predecessor material's 624 Big5-only authors, each36 comments. No new
  author selection, external labels, MEPS, bridge, remote upload, neural calls
  or fit. Verify zero overlap with original E1 240authors before replay.
- Freeze existing E1 representation/scalers/rank23/encoders/decoders/direct
  and permuted maps. Verify all18 original inventory files and numeric replay.
- Native common source protocol:32 comments,4096 project tokens, min12 tokens,
  max128 output units. Five views:whole,native,sentence_pack160,fixed128 cross,
  fixed128 within-comment. Do not silently change tokenization to repair old
  wording of raw-text preservation.
- Full panel plus one source-comment-disjoint A/B partition (seed916), using
  the already bounded same source panel. No overlap in original source_row;
  half panels have less support, not replicas of the original32-comment design.
- Save exact view unit -> source comment -> original token/character segments.
  Legacy budget processing joins project tokens with spaces: match token
  sequence, preserve original formatting in the sidecar. Restrict alignment to
  current nonoverlapping five views; fail on ambiguity/unsupported operators.
  Never silently locate a repeated phrase by first substring match.

## Fixed Analyses

1. E1 source inventory and fixture replay, plus exact historical conclusion
   extraction from E1/E4/W4/G1 files. Inspect evidence tier: E4 alignment was
   corrected to post-hoc exploratory; preserve that correction.
2. On all624 and separately historical heldfold0, frozen same-source cross-view
   R2/direct/broken-map comparisons. These share original text and cannot by
   themselves establish author invariance.
3. On A/B source-disjoint panels: per-view raw48, shared23 and retained48
   own-vs-other cosine rank AUC, with author-bootstrap intervals500 seed916.
   Rank unit is one anchor author versus other authors in the same sample;
   pairs are not independent observations. No topic-matched causal or trait
   inference; same-author context habits can contribute. Zero-norm rows excluded.
   No new scorer fit or choice of a winning view.
4. G1 radial support on full native material must match existing native_grain_r1;
   do not change or bypass its envelope to expand the count of scored people.
5. All29,075 candidate sentences and existing80 source-model annotated pairs:
   compare retention within common panel and full containment in one view unit.
   Partial cross-slice evidence remains addressable, not negative evidence.
   Original annotations stay unchanged and are not expert truth. Count only
   support/fragmentation, not statement entailment, attribution quality or
   causal rationale faithfulness.

## Mathematical Objects

Unit encoder phi is frozen. For viewv, author features are mean/std of phi;
X_v uses original discovery normalization. S_v=E_v(X_v), J_v=S_v D_v,
R_v=X_v-J_v. R_v is an operational reconstruction remainder, not independent
noise or a fitted JIVE individual component. Exact X_v=J_v+R_v is bookkeeping;
recurrence and shared support are separate empirical questions.

For evidence span e and token provenance tokens T_e, coverage is
|T_e intersect retained source tokens|/|T_e|. Unit completeness is the maximum
fraction of T_e contained in one unit. Token agreement is not byte identity:
the sidecar explicitly retains raw offsets and records formatting changes.

## Delivery and Bound

No broad dependency installation: spaCy/stanza/NLTK and their English parsers
are not installed. Do not replace open-domain ownership parsing with another
small quote regex and call it solved. This batch builds its needed addressing
interface and returns the substantive parsing question to C03/C16.

Local CPU,2threads,20min. Fresh ignored multiview_address_r1 directory with
contract, inventories/replay, observation/support/crosswalk tables, shared and
remainder arrays, all comparison results, audit and summary. Public source
graph/report, no raw quotes or author IDs. Goal remains ACTIVE. No commit/push.

## Prior Work

[Lock et al. 2013 JIVE](https://pmc.ncbi.nlm.nih.gov/articles/PMC3671601/): read
framework and separation of joint, data-type-specific and residual variation.
The frozen V7 method is inspired by JIVE, not exact JIVE; keep that distinction.
[ERASER, DeYoung et al. 2020](https://aclanthology.org/2020.acl-main.408/): explicit
supporting spans permit rationale evaluation, but span retention alone does not
measure comprehensiveness/sufficiency or agreement with human rationales.
