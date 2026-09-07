# S18: from a decoding weight to an intervention pattern

2026-09-08. Discovery question: does the population-trained KV reader identify a
direction that participates in the frozen decoder's final response? S12-S17
establish a useful readout; now connect that readout to an actual intervention,
instead of adding more approximation orders or prediction hyperparameters.

## Literature and the exact mathematical step

[Haufe et al., NeuroImage2014](https://pubmed.ncbi.nlm.nih.gov/24239590/)
distinguish extraction weights from forward activation patterns. A large decoder
coefficient need not locate the generating signal. Their linear-model perspective
motivates converting our frozen extraction weight into a data-covariance pattern.
This is a transfer of linear statistical reasoning, not a claim that transformer
heads are EEG sources or that a covariance pattern alone establishes causation.

[Park, Choe & Veitch, ICML2024](https://arxiv.org/html/2311.03658v2)
separate measurement and intervention representations and show that the metric
matters. Their causal-inner-product identification uses assumptions about
separable concepts and unembedding geometry. Those assumptions are not established
for our intermediate KV reader. S18 therefore compares Euclidean and empirical
covariance patterns experimentally; it does not call the latter a causal metric.

Here is the independently checkable calculation. For centered training-author
pairing vectors X, covariance Sigma, and fixed linear score s=beta^T x, the
least-squares forward pattern for predicting x from s is

\[
a=\frac{\operatorname{Cov}(X,s)}{\operatorname{Var}(s)}
 =\frac{\Sigma\beta}{\beta^T\Sigma\beta}.
\]

When the denominator is positive, beta^T a=1. For a held-out author's pairing
vector d, let d_selected=a(beta^T d), d_null=d-d_selected. Then
beta^T d_selected=beta^T d and beta^T d_null=0, exactly. This is a score-preserving
oblique decomposition, not a claim that the null part is uninformative for the
LLM or for psychology. The Euclidean comparator uses a_E=beta/(beta^T beta).

The covariance pattern is also the minimum-Sigma-pseudoinverse-norm displacement
in range(Sigma) having the specified beta score. Under x'=T x, beta'=T^-T beta,
Sigma'=T Sigma T^T, it transforms as a'=T a. No dense covariance inverse is needed:
compute X^T(X beta)/(n-1). Singular training covariance is allowed; zero score
variance is an explicit failure, not an arbitrary regularization choice.

These statements concern centered training covariance. The intervention itself
decomposes the full d relative to the meaningful zero-pairing baseline, not
relative to the population mean. It erases the beta*d component, not the reader
intercept. Both conventions are saved rather than conflated.

## Fixed experiment

Use the first24 authorized N source authors, view0 only. No questionnaire access.
Same model, compact prompt, device and native layer17 head capture as S12. This
experiment uses each text's native query, not the S14 common query, to replay
exactly the S12 features on which the reader was trained.

For each existing outer fold, estimate covariance from the other authors' S12
three-view-averaged native KV pairing vectors. The reader is the frozen S12
KV/PCA C reader for that fold. No new response fitting, directions selected on
endpoint scores, or use of the held-out author's vector in covariance fitting.

At the final-token concatenated head vector h before layer17 W_O, measure:
native; self patch; h-d; h-d_selected and h-d_null under each of the two metrics.
Seven full forwards per author,168 total,0 backwards. Report patch norms and
attention/feature/native-score replays. Stop collection at15minutes; retain
completed author units. Do not run alongside the S17 neural collector.

For each geometry report full-removal, selected-removal and null-removal effects,
their RMS, correlations across24 authors and paired-author bootstrap intervals.
Keep absolute and relative errors, not correlations alone. The scalar reader
necessarily ignores d_null, but the decoder might not: that is the question.

Also retain the nonlinear interaction

\[
I=[F(h)-F(h-d)]-[F(h)-F(h-d_s)]-[F(h)-F(h-d_n)]
 =-\int_0^1\int_0^1d_s^T H_F(h-u d_s-v d_n)d_n\,du\,dv.
\]

The finite difference is measured exactly; the Hessian identity explains it for
a twice differentiable fixed suffix. No Hessian is estimated in this pilot.
It prevents mistakenly expecting two endpoint effects to add in a nonlinear model.

## Decision value

A selected component that preserves substantial decoder effect offers a concrete
bridge from a learned population readout to a manipulable vector pathway.
If the covariance pattern improves over the Euclidean projection, geometry is a
useful mechanism, not just a different plotting convention. If both selected
components explain little while null effects persist, the decoder's operative
directions are broader than the one-dimensional score; then estimate a bounded
functional response subspace rather than claiming every large beta is causal.

No release-score update or human accuracy claim. No bridge/MEPS/lockbox, external
API, LoRA or unbounded queue. T/F positive pathway stays unchanged.

Code: `reader_pattern.py`, `run_grain_reader_pattern.py`, corresponding tests.
Private output: `results/grain_vector_mechanisms/reader_pattern_s18_r1/`.
Report: `reports/grain_vector_mechanisms/S18_READER_PATTERN_REPORT.md`.
