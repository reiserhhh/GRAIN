# S14: a common query preserves the relational response

2026-09-08. COMPLETE. Same213 source N authors,3views, five author folds.
639 prefix calls,0 full suffix calls,0 backwards,590.30seconds. No new fitting.

## Decision

Proceed with a fixed author-out reference query as the next compression target.
Most of the recovered source response remains readable from author-specific KV
under this common query. Author-conditioned query variation is a small additional
term and its local second-order formula is highly accurate on this source.

| Query dose | Teacher-C r | MSE/train-mean-null | Source N r |
|---|---:|---:|---:|
| Common reference,alpha0 |.315627|.908661|.133706|
| Fixed half interpolation,alpha.5 |.318585|.906343|.138944|
| Native author query,alpha1 |.320283|.906267|.142869|

Common-minus-native target delta r=-.004656, paired95%CI[-.030492,.036014].
Source delta r=-.009163, CI[-.027605,.010624]. These data support retaining the
simple common-query route for mechanism development, not formal equivalence
of all query strategies. All doses are reported; none was selected on source Y.

The common query is the mean of training authors' three-view raw queries, with
the held-out fold entirely excluded. It is rotated at each test prompt's actual
position. Median raw-query cosine to reference=.999191,10-90%[.998221,.999656];
median relative displacement=.040593. This is a small natural perturbation,
not evidence that arbitrary or unrelated queries work equally well.

## Shared readout and local query increment

Cov(Y,native)=.000322346=.000296975 shared+.000025371 query increment.
The query-increment covariance CI[-.000017623,.000067941] crosses zero;
its source r=.079032,p=.2508. About92.1% of this **signed sample covariance**
sits in the shared term. It is not a92% personality-variance claim; the terms
are not orthogonal and the denominator can vary across samples.

Common-query source r=.133706 has FisherCI[-.000739,.263403] and paired author
bootstrapCI[-.019653,.260968]. Native source r remains the S12 result. The main
advance is common-query recovery of C, not a new external-accuracy gain.

## Query response has a compact local mathematical description

At each head, fixed KV and reader beta give R=beta^T Vtilde and
U=(q_native-q_reference)^T K. Under reference attention, calculate
L=Cov(R,U), Q=E[(R-E R)(U-E U)^2]. Compare alpha L and alpha L+alpha^2 Q/2
with the actual paired score change, summing heads and averaging views.

| Dose | Approximation | Increment r | MSE/zero-increment |
|---|---|---:|---:|
|.5|First order|.997255|.005967|
|.5|Second order|.999957|.000088|
|1|First order|.989699|.021955|
|1|Second order|.999416|.001185|

These are internal increments, not questionnaire predictions. No coefficient
was fitted to these increments. Their high reconstruction is evidence for a
local covariance/cumulant account around the common query, consistent with the
small observed query displacement. It does not extend S8's local curvature
approximation to arbitrary large token-collapse interventions.

Mathematical provenance: finite exponential-family derivatives
([Wainwright & Jordan2008](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/WainwrightJordan2008.pdf)).
The new [CCQ preprint, Le et al.2026](https://arxiv.org/abs/2606.01294) is related
read-side curvature work in linear attention; no CCQ modification was applied.
Full derivation and joint-relation witness: `docs/GRAIN_RELATIONAL_READOUT_THEORY_V4.md`.

## Execution evidence and next question

Native vector replay maxerror0, reader maxerror1.02e-16. Centered-expectation
identity error1.94e-17; covariance accounting error2.37e-20. Three-dose scores
and Taylor components were frozen before joining the previously exposed source
questionnaire. No bridge/MEPS/lockbox, external API, new model or outcome fitting.

The next finite experiment separates exact comment attention mass from
within-comment covariance reading. If exact masses plus covariance work,
joint moments may replace much of the token-level operation. If they fail,
we can locate whether higher conditional moments or normalization is needed,
rather than restarting a broad ordinary-embedding search.

Artifacts: `results/grain_vector_mechanisms/shared_query_s14_r2/` (ignored).
`manifest.json`, `run_status.json`, `summary.json`, `metrics.csv`,
`paired_intervals.csv`, `covariance_split.csv`, `query_taylor.csv`,
private reference arrays, per-author packets and scores-before-outcomes CSV.
Protocol/code was committed as d4b0948 before collection. r1 was prepared only
and superseded without any neural collection; do not resume it.

Commands: `scripts/run_grain_shared_query.py prepare|run|evaluate --output-dir
results/grain_vector_mechanisms/shared_query_s14_r2` with the existing project
venv. Prepare refuses existing directories; completed results are not overwritten.
