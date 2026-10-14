# C12/C13: Semantic Response Accounting V1

2026-09-13. Frozen before outcomes. Uses C14's existing semantic-link reader,
not an accuracy-search extension. Traces RQ-C, V5 G2/G3 and V7 condition/time
work while preserving their different measurement objects.

## Question and Decision

Does the apparent change of an author's semantic profile reflect a change
within observed conditions, a shift in condition composition, or entry/exit
of observed conditions? Does teacher-function imitation preserve those parts
equally well? Use the answer to define the next conditional-response interface.

## Data and Scope

Reuse 624 Big5-only authors, 36 comments each, and C14's cached 60 affinity
targets/readers. Read no questionnaire outcomes. All comments have timestamps
and subreddit metadata. No MEPS/bridge fit, lockbox, new inference or model fit.
Main reader comparison remains 120 held authors; 504 source authors only give
support descriptions. Fold0 is historical development, not fresh confirmation.

Condition means recorded subreddit, not a fixed stimulus or full situation.
Scores remain model-derived semantic affinities, not verified personal events.
All 80 prior model/NLI judgments stay attached to source addresses; no new
ownership inference or gold labels is manufactured. Known model-coded other
statements remain distinguishable in the evidence inventory, not auto-corrected.

## Units and Formula

1. Order each author's original 36 comments by actual timestamp, with comment
   ID as deterministic tie-break. First18 vs last18. If the boundary timestamp
   is tied, record it and exclude that author from strict chronological claims.
2. Average candidate sentences inside each comment, then weight supported
   comments equally. A comment with no candidate is missing, never score zero.
3. Within each half compute condition proportions p and mean scores m. Over
   the intersection S of observed conditions use the symmetric product identity:

   W = sum_S (p_early+p_late)/2 * (m_late-m_early)
   C = sum_S (p_late-p_early) * (m_early+m_late)/2
   U = sum_new p_late*m_late - sum_lost p_early*m_early
   Delta = W + C + U.

W is observed within-condition change, C shared-condition composition change,
U unsupported condition entry/exit. No absent-cell response is imputed.
Use the existing source-only teacher center/scale for all reader channels.
With changing support, C and U separately depend on score origin; W and C+U
do not. Preserve the convention and test this algebraically.

4. Define coefficients once for each author/partition. Apply the exact same
   linear accounting operator to teacher t, reader p and error e=p-t. Then
   A(p)-A(t)=A(e) for each part. Thus a total-score agreement cannot stand in
   for within-condition agreement. This is standard linear error propagation.
5. Ten fixed seeded random18/18 partitions of the identical comments provide
   a sampling comparator, not a psychological null or longitudinal occasions.
   No selection of a favorable seed, radius, condition aggregation or degree.

## Analyses and Outputs

- Support: all/source/held counts, candidates per half, shared conditions,
  common mass, cells with >=2 supported comments in each half, span and gap.
- Primary: all60 teacher/reader Delta/W/C/U vectors; held-author correlation
  and MSE, by mean/max affinity family. W requires common support; repeated-
  cell sensitivity uses only predeclared support, not observed score quality.
- Signed component projections onto Delta plus all cross terms; never call
  them positive explained-variance percentages. List cancellations.
- Chronological squared change versus ten random allocations of the same
  comments, paired author bootstrap 500 draws, seed914. Descriptive only.
- Verify direct pooled mean identity, reverse-half antisymmetry, no-common-
  support handling, missing-data refusal and reader-error identity. Replay
  summary arithmetic through independent loops, not only test the report text.
- C00: re-open V7 E2 r7 and corrected W3 metric artifacts and source inventory;
  bind their evidence and remaining questions without rerunning old queues.

Output: ignored `results/grain_vector_mechanisms/all_plan_closure_v1/response_accounting_r1/`.
Report: `reports/grain_measurement_foundation/RESPONSE_ACCOUNTING_REPORT.md`.
Save source/code/plan hashes, per-comment support, decomposition coefficients,
all private author outputs, public aggregate tables, errors and an audit.
Stop after the declared comparison; local CPU only, <=20 minutes, no training.

## Method Source

Standard symmetric decomposition: Kitagawa (1955),
[DOI](https://doi.org/10.1080/01621459.1955.10501299); practical derivation in
Das Gupta (1993), [Census manual](https://www.census.gov/library/publications/1993/demo/p23-186.html),
section2.2. Here we apply the product identity to a mean and keep unshared
support as an explicit term. This is an adaptation, not a novel theorem or
a causal identification result. V7's earlier T=K+A and temporal/random checks
provide local predecessors, not interchangeable outcomes.
