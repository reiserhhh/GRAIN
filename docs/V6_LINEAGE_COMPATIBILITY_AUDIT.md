# GRAIN V6 Lineage Compatibility Audit: Frozen Protocol
## Purpose

This is not a contest between GRAIN versions. It asks whether three distinct
objects that were easy to conflate remain distinguishable under V6 safeguards:

\[
\begin{aligned}
Q_u^{\mathrm{fixed}} & = \mathcal L(X_{ui}\mid A_{ui}=a_u),\\
Q_u^{\mathrm{natural}} & = \mathcal L(A_{ui},X_{ui}),\\
Q_u^{\mathrm{resid}} & = \mathcal L(X_{ui}-E[X\mid A_i]).
\end{aligned}
\]

`A` is an observed subreddit selection and `X` is text-derived expression.
The first object is the V3/V4 fixed-rind design claim; the second is V6 J1's
natural selection-expression geometry; the third is a sensitivity object, not
a purified truth. They answer different questions.

## Frozen source and restrictions

- Local PANDORA Tier-U comments only: author, body, timestamp, subreddit.
- Explicit personality-report text is excluded with `PERSONALITY_LEAK_RE`.
- No Big5, MBTI, Enneagram, clinical, market, or other endpoint field is read.
- Raw text and per-author rows are retained only in process memory. Artifacts
  contain aggregate counts, aggregate metrics, and no text.
- Authors use the existing J1 `suica-v6-joint-process-j1` 50/25/25 hash
  partition. Confirmation is the reporting cohort; it is not used to choose
  the protocol parameters below.

## A. Fixed-rind compatibility endpoint

For every eligible confirmation author, choose their most frequently selected
subreddit `a_u`. Construct two text-disjoint, time-spread views from:

1. comments within `a_u` (`fixed`); and
2. comments from at least four other subreddits (`mixed`).

Each arm uses four evenly spread, non-overlapping three-comment blocks per
technical view. Each view is converted to eight 128-token slices. Thus every
author/arm/view is matched at the evaluated token budget and shares no raw
comment with its paired view. Authors with an overall fixed-versus-mixed time
span ratio below `0.50` are excluded before scoring.

For each frozen V2 scalar construct, compute across-author technical-view
reliability and the paired contrast

\[
\Delta_c = r\!\left(s^{\mathrm{fixed},L}_{u,c},
s^{\mathrm{fixed},R}_{u,c}\right)
-r\!\left(s^{\mathrm{mixed},L}_{u,c},s^{\mathrm{mixed},R}_{u,c}\right).
\]

Cluster bootstrap resampling is over authors. The historical claim can be
called upheld under this **new event-disjoint implementation** only if the
bootstrap lower interval bound is positive for all three previously supported
constructs: `first_person_usage_v2`, `directive_action_v2`, and
`novelty_play_v2`. `tension_core_v2` is descriptive only because its former
positive result was overturned as a temporal-clustering artifact.

This is a V6-compatible replication, not a literal rerun of the old
interleaved-token procedure. A result does not identify a personality trait.

## B. Natural selection-expression reference

J1's existing frozen result remains a separate reference:

\[
Q_u^{\mathrm{natural}}=\mathcal L(A_{ui},X_{ui},\Delta t_{ui}).
\]

Its high author retrieval is expected to include stable choice and community
footprints. It cannot be compared numerically to the fixed-rind scalar
reliability contrast as if they were estimates of the same latent variable.

## C. Frozen residualization sensitivity

On J1-compatible, disjoint three-event views, calculate four frozen legacy
anchor coordinates per event. Fit a population condition mean using **only
discovery authors**, with a minimum of 20 reference events per subreddit:

\[
\widetilde X_{ui}=X_{ui}-\widehat E_{\mathrm{discovery}}[X\mid A_i].
\]

Compare confirmation same-author AUC for the raw and residual vectors using a
shuffled-alignment randomization. This is intentionally a sensitivity test:
it can show that subtraction changes or weakens the natural author footprint,
but it cannot establish that either version is the true personality signal.

## Interpretation matrix

| Possible result | Correct reading |
|---|---|
| Fixed-rind contrast survives | V3/V4 design-control insight remains compatible with V6 as a conditional-expression precision result. |
| Fixed-rind contrast is inconclusive | Do not overturn the old result automatically; check support, event-vs-token construction, span restriction, and scorer equivalence first. |
| Fixed-rind contrast reverses with CI above zero in the opposite direction | A real implementation-level contradiction requiring a separate audit. |
| Residualization lowers natural author retrieval | Compatible with the old mediator/choice warning; it does not prove that the removed information is personality. |
| Residualization improves retrieval | An estimand or implementation discrepancy requiring inspection; do not call it denoising success by default. |
| J1 remains high while ordered J3 is refused | Not a contradiction: first-order natural process stability does not imply an identified order-specific operator. |

## Required artifacts

- `results/v6_lineage_compatibility/fixed_rind_constructs.csv`
- `results/v6_lineage_compatibility/natural_residual_sensitivity.csv`
- `results/v6_lineage_compatibility/assumption_delta.csv`
- `results/v6_lineage_compatibility/lineage_result.json`
- `reports/V6_LINEAGE_COMPATIBILITY_AUDIT.md`
