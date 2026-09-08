# GRAIN autonomous research: functional information left behind

> **Latest goal, 2026-09-08:** a new user-requested goal-tool object is ACTIVE:
> complete the whole GRAIN reference-function framework, not another local accuracy
> packet. Two Claude Fable5.1/max read-only seminar rounds complete. M8 has connected
> frozen P/Q matrices to complete scores and natural edits on12 new evaluation authors.
> The outcome-blind prefix-only source connection is RUNNING; after completion,
> integrate with author configuration, old external evidence and the4W theory.
> Governing plan: `GRAIN_FRAMEWORK_CLOSURE_PLAN_V1.md`.
> Historical goals/next experiments below do not override this route.

> **2026-09-08 execution update:** query-region functional matrices are COMPLETE:
> 2 reference / 2 evaluation authors, 72 suffix forward/backward pairs plus six
> preserved endpoint-preflight pairs, 77.30 seconds total. Frozen matrices read
> 6/6 held-author effect directions correctly; token averaging loses useful action.
> [Report](../reports/grain_vector_mechanisms/QUERY_REGION_FUNCTIONAL_REPORT.md),
> [milestones](GRAIN_MAINLINE_MILESTONES_20260908.md).
> This proposed P-only transport is now superseded by the complete M8 packet.
> The earlier statement that no new goal object existed is historical.
> Do not resume S27 or a source-label accuracy search.
> Historical execution instructions below are not the current queue.

2026-09-08. New user-authorized goal, following completed S0-S4. Discovery first;
not a restart of the old neural queues. No bridge, MEPS questionnaires, or lockbox.

## Question and decision

The teacher-only E5 student preserves most T/F label covariance, but only about
26% of neuroticism teacher-label covariance. This accounting does not identify
which information the encoder lacks. First distinguish an inadequate projection
or estimator from a missing representation. Keep T/F as a positive control.

S5A asks whether unsupervised PCA64 discarded teacher-relevant directions. On
the same exposed D1 authors and S4 outer folds, compare PCA64/Ridge, full Ridge,
and PLS/Ridge. Fit means, scales, subspaces and coefficients on training authors
only. Select rank and regularization using inner-author teacher MSE, never the
questionnaire. Freeze all OOF scores before joining existing source outcomes.
Report all candidates, teacher fidelity, external association, paired differences,
and the exact teacher/student/remainder covariance identity.

Fixed search: PLS ranks 2/4/8/16/32; Ridge alpha 1/10/100. Five outer and three
inner author folds. Also report the candidate chosen wholly inside each outer
training fold. Budget: zero new neural calls, two CPU threads, ten-minute cap;
retain partial outputs on timeout. No extra search based on external scores.

If supervised or full coordinates recover the remainder, develop the functional
projection. Otherwise test a bounded nonlinear/distribution representation before
attributing the remainder to unavailable encoder information. Further neural work
must distinguish a new mechanism, not merely fill an existing table.

## Mathematical sources and actual use

- Cook, Helland & Su (2013), *Envelopes and partial least squares regression*,
  JRSS B 75:851-877, [publisher](https://academic.oup.com/jrsssb/article/75/5/851/7075921),
  DOI 10.1111/rssb.12018. Response-relevant subspaces need not be the largest
  variance subspaces. We implement training-centered PLS coordinates, not their
  likelihood envelope estimator. Finite-sample improvement is an experiment,
  not a consequence guaranteed by the envelope assumptions.
- Fukumizu, Bach & Jordan (2004),
  [JMLR 5:73-99](https://www.jmlr.org/papers/v5/fukumizu04a.html).
  The target notion is preserving the response relationship, expressible as
  conditional sufficiency, rather than reconstructing all input variance.
  No claim that finite linear PLS establishes conditional independence.
- Szabo et al. (2016),
  [Learning Theory for Distribution Regression](https://jmlr.org/papers/v17/14-510.html).
  If needed, represent a user's empirical comment distribution with a kernel
  mean rather than just the mean of original embeddings. This separates two
  sampling levels: authors and comments. The theorem's sampling assumptions are
  not automatically satisfied by dependent Reddit comments.
- Rahimi & Recht (2007),
  [Random Features for Large-Scale Kernel Machines](https://proceedings.neurips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html).
  A bounded random Fourier map can approximate a Gaussian kernel without neural
  inference. Finite features are an approximation, not a universal sufficient
  statistic or a new invention of GRAIN.

## Working equations

For training-standardized author vectors X and centered teacher score s, compare
variance coordinates with a PLS response subspace related to the Krylov sequence
span(c, Sigma c, ..., Sigma^(k-1)c), c=Cov(X,s), Sigma=Cov(X).
Estimate s_hat=mean(s)+b' P' X by regularized training-only regression.

For a possible distribution extension, use
m_u = (1/T_u) sum_t phi(h_ut), not phi((1/T_u) sum_t h_ut).
These coincide for linear phi, but generally differ for nonlinear kernels.

At evaluation, retain the exact identity
Cov(Y,s)=Cov(Y,s_hat)+Cov(Y,s-s_hat).
Better teacher MSE and retained psychological association are different outcomes;
neither is used to hide the other. Results here concern exposed development
authors, not a new confirmatory sample or the released GRAIN scoring model.

## Research loop

State the question -> verify a primary mathematical source -> implement the
smallest distinguishing experiment -> test code -> run -> interpret every arm ->
record the next question and resource value -> commit locally. Stop only at a
real dependency requiring the user, or after the active objective is achieved.

## S5A result and S5B decision

S5A completed. N teacher fidelity rose .4614 -> .5059 with PLS, while external
r was .0673 -> .0467; full Ridge gave .0494. The missing psychological association
was not recovered by this linear-coordinate change. T/F remains a positive case
(PCA64 AUC .7173, PLS .7070). Better teacher prediction is not the same objective
as retaining teacher-label covariance.

S5B will compare a linear mean, a Gaussian random-feature map of that mean,
a mean of Gaussian maps of individual comments, and their equal-block sum.
Use 256 fixed random features, seed 731, training-only median pair distance,
training-only centering and scalar block scaling, and risk-normalized Ridge
lambda .001/.01/.1/1/10. The linear control also tests broader regularization
than S5A (whose alpha often hit its upper boundary). Same authors/folds and
teacher-only nested selection; no selection using the external associations.
No neural calls. Twenty-minute cap, two CPU threads, no expansion on these
labels. This separates nonlinearity of a mean readout from distributional inputs.

## S5B result and S6 mechanism experiment

S5B completed in the same source population. N mean-plus-bag teacher r=.5096,
external r=.0663; linear mean .5028/.0593. No recovery of the missing association.
The T/F linear control remains AUC=.7158. Stop enlarging this ordinary student
family. The useful next distinction is token arrangement versus pooled state.

S6 fixes eight existing S1 authors (panel indices 0-7, view0), three existing
queries and the same E1/E2 comment groups. At block16, replace each selected
comment's token states H_t along H_t(a)=H_t+a(mean_comment(H)-H_t). Every comment
mean is exactly preserved; token IDs, positions, other comments and the query
state are unchanged. This is a controlled hidden-state intervention, not text
deletion, natural paraphrase or a claim about measured questionnaire change.

Let D be the mean-preserving displacement. Measure actual score difference and
the path integral int_0^1 <grad_H s(H+aD), D> da. The sum of D within each comment
is zero: any effect is an explicit witness that this readout uses information
in the kernel of this particular averaging map. It does not prove E5 has no
such information, nor that all pooled encoders are insufficient.

Use Simpson nodes 0/.5/1, adding .25/.75 only when the completeness gap exceeds
max(.001, .05*abs(effect)). Record both rules and endpoint scores, without hiding
large quadrature errors. Reuse the origin gradient across E1/E2. Maximum216
forward/backward pairs, eight complete author units,30-minute compute cap.
No extra neural experiment is auto-started if these data already decide whether
mean-null token geometry has a usable effect.

Sources: [Sundararajan, Taly & Yan, ICML2017](https://proceedings.mlr.press/v70/sundararajan17a.html)
guides integrated-gradient completeness. Here the input is a fixed decoder suffix
and the path is a declared internal-state intervention, not an input-word
explanation. [Janizek, Sturmfels & Lee, JMLR2021](https://www.jmlr.org/papers/v22/20-1223.html)
motivates keeping interactions distinct from single-feature effects; S6 does not
implement Integrated Hessians or estimate a full Hessian.

## S6 result and S7 group-action split

S6 completed in318seconds with154 forward/backward pairs. Mean-preserving token
collapse has TF/N/confirmation effect RMS=.01355/.02471/.00900. Integrated-path
r=.9980/.9998/.9994. Matched N/confirmation effect r=.9717 (author bootstrap
CI .8094-.9928). This establishes a useful configuration response at the fixed
decoder site. It does not yet separate order from within-comment dispersion.

S7 uses the same eight authors and all three queries. Permute states only within
each selected comment, preserving the complete multiset, all moments, count,
and other/query states. Three uniform seeded permutations per E1/E2,168 total
forwards including24 self checks, zero backwards, ten-minute cap.

For finite permutation group G, define Rf(H)=mean_g f(gH) and Pi H=mean_g gH.
The exact identity is f(H)-f(Pi H)=[f(H)-Rf(H)]+[Rf(H)-f(Pi H)]. The first term
is location on the permutation orbit; the second is nonlinear response to a
distribution rather than its averaged state. S7 estimates Rf with only three
draws, so its numerical split is exploratory Monte Carlo, not an exact full-group
average or orthogonal psychological variance partition.

Sources: finite-group averaging in
[Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/reprbook.pdf)
and the invariant-function perspective in
[Zaheer et al., Deep Sets, NeurIPS2017](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html).
Do not infer that a mean is sufficient for every invariant nonlinear function.
This experiment can decide whether the next lightweight representation should
prioritize ordering/positions, distributional spread, or both.

## Historical checkpoint after S7

S5A, S5B, S6 and S7 are COMPLETE. No child experiment remains running. S7 took
239seconds/168 forwards and localized the shared N/confirmation response to
the nonlinear distribution term: r=.9730 [.8575,.9928]; ordering term r=-.0776.
Together S6/S7 cost322 forwards and154 backwards, about9.3minutes after loading.
The active goal remains open; this is a completed cycle, not universal closure.

Next: S8 tests the curvature-times-orbit-covariance expression in theory V3.
The exact matrix-free permutation covariance and its quadratic contraction now
pass four synthetic full-group enumeration tests (`orbit_geometry.py`). These
are mathematical witnesses, not new neural data. Next use a bounded fixed
author/query panel and symmetric finite differences or
supported Hessian-vector products to compare the quadratic approximation with
actual distribution response at multiple radii. Keep the same queries and
authors, no outcome-driven reselection. Distinguish small-radius approximation
from full-radius performance. If useful, fit a low-rank query-specific operator.

After that mechanism step, connect it to N's actual teacher-minus-student
covariance on the authorized D1 source authors. Verify D1 observer/text identity
before making any new neural measurements; match the same three views. S6/S7
used different authors, so do not claim they have already explained the D1
external remainder. Do not open bridge/MEPS/lockbox or resume old queues.

Deliverables at that checkpoint: S5/S6/S7 reports, theory V3, four runners and their tests.
189 focused tests plus four exact-geometry tests pass. All private artifacts remain ignored. Continue from
these paths, not the dated training instructions in the parent AGENTS file.

## Historical checkpoint: S8 complete, S9 source experiment running

S8 completed384 forwards,0 backwards in544.8seconds. A small-radius quadratic
predicts N full-radius internal effects with r=.7537 and MSE/zero=.7177, but
not T/F (r=-.2056; MSE/zero=6.2142). Shared N/confirmation curvature r=.7117.
The exact general formula retains a Hessian that varies along each path; its
one-point covariance contraction is a useful local approximation, not universal.
See [S8 report](../reports/grain_vector_mechanisms/S8_ORBIT_CURVATURE_REPORT.md).

S9 now directly targets the actual213 N source authors and64 fixed T/F controls.
The old teacher used CUDA/BF16/padding3072; the paired new collector uses the
same model checkpoint on MPS/FP32 without padding. Decompose the frozen student
remainder into observer-condition difference, all-comment configuration response,
and projected-suffix/student remainder. Keep the query state unchanged and state
that it can carry earlier-layer contextual information. This is not a mean-only
encoder sufficiency test.

Prepared277 authors/831 views with exact old token hashes. Run48 authors as a
resource-only checkpoint, then decide whether to finish the fixed cohort. At most
1662 forwards,0 backwards, one-hour cap; no parallel GPU collectors. No Y connection
until all cohort components have been written and hashed. Existing source outcomes
only; no bridge/MEPS/lockbox, retraining, external API, or old-queue restart.

S9 output: `results/grain_vector_mechanisms/source_remainder_s9_r1/`.
S9 protocol/code are locally committed as431db14. Use `run_status.json` for live
progress, not the historic checkpoint above. The overall goal remains active.

## Completed checkpoint: S9/S10/S11, source configuration link found

S9 completed all277 authors/1662 forwards in2431.4seconds. N configuration
contrast C=S_new-M has r=.2116 with source Y, p=.00191; Cov(Y,C)=.001046 with
paired bootstrap CI[.000284,.002054]. Native-versus-projected N delta r=.0773
[.0236,.1330]. On64 T/F controls, native-versus-projected delta AUC=.0380
[.0049,.0774]. This is the first direct source-outcome connection for this
mean-null intervention, not just internal response fidelity. Old/new observer
differences remain explicitly separated; no bridge/MEPS/lockbox was opened.

S10 completed72 forwards/0 backwards in104.8seconds. The exact fixed-head
query-response derivative is the KV cross-covariance. Its numerical integral
reconstructs captured head changes (maximum relative MSE6.46e-13). Removing
within-comment KV pairing while holding keys, group mass and query fixed changes
final scores; N/confirmation effects correlate .8070 [.5252,.9710]. S10's
eight-author panel is not the same as S9's full source cohort.

S11's zero-forward contrast students completed in6.4seconds. PCA64/PLS2 E5-mean
students cannot recover C here (target r=-.0187/-.0183; external N r=-.0058/.0311),
while projected-background target r=.6338/.6110. Fixed-operator additivity error
2.22e-16. Stop this finite mean-student family, retain the positive native C result.

No experiment remains running at this checkpoint. Next is a **bounded source
representation collector**, not more ordinary hyperparameters: same213 N authors,
three views, body-state mean versus contextualized query state versus layer17
KV-pairing vector. Reuse S9 origins and targets, record observer identity; save
representations in the same pass, then compare frozen author-out learners. Inspect
the current capture APIs and write the finite protocol before collection. Avoid
claiming that a model-derived prefix encoder is fully LLM-independent.

Read the S9/S10/S11 aggregate reports under `reports/grain_vector_mechanisms/`.
All code/result-source references are committed locally; no remote push. The
active objective remains open because compact recovery of the useful component
has not yet been established. Proceed independently from this mainline question.

Final focused validation for this cycle:216 tests passed in18.23seconds; Ruff
passed for all newly added/modified mechanism code. Only two existing SWIG
deprecation warnings were reported. Unrelated worktree changes were preserved.

## Historical resource checkpoint: S12 collection

The frozen source representation protocol is locally committed as0c3c046 and
its paired readers as8f866f9. Same213 N authors,3views; one forward captures
block16 equal-comment body mean, final query state, and layer17 KV-pairing vector.
The24-author pilot completed72 forwards in105.36seconds, origin replay error0,
head reconstruction error2.83e-6. Full213 collection resumed; inspect
`results/grain_vector_mechanisms/query_representation_s12_r1/run_status.json`.
Do not launch another neural collector. Analysis runs after COMPLETE via
`scripts/analyze_grain_query_representation.py`, using all18 frozen teacher-only
arms, then the same exposed source outcomes. Source T/F success remains frozen.

Parallel theory work adds Fisher-metric query accessibility and its weighted
projection derivation in`GRAIN_QUERY_INFORMATION_GEOMETRY_V1.md`;3 new finite
tests pass. This describes token-weighted local value access, not personality
variance. Source/outcome conclusions await S12; no new protected data are needed.

## Historical checkpoint: S12 complete, compact relational readout found

All213 N authors/639views collected in953.63seconds,0 backwards. Native S9
score replay error0; head reconstruction max4.42e-6.90 fixed affine readers,
3834 OOF rows,18 declared arms, no questionnaire fitting. KV/PCA64 recovers the
source configuration contrast at target r=.3203, MSE/null=.9063; body/PCA
target r=.0074 and E5/PCA -.0187. Paired target delta versus body .3128
[.1177,.4878], versus E5 .3390 [.0919,.5246]. This is the positive localization
needed to leave the ordinary-mean failure behind.

KV/PCA's source N r=.1429 (FisherCI[.0086,.2721], bootstrapCI[-.0055,.2702]);
query/PLS's N r=.1695. External gains over matched body/native controls are not
established, and all arms are displayed. Choose KV/PCA for the next mechanism
shortcut by target fidelity, not the largest external r. The configuration
contrast itself retains S9's r=.2116; full-source T/F control is unchanged.

No background collection remains. Next implement a prefix-only KV extractor
and replay the existing affine readers on a fixed small source subset, checking
that upper decoder blocks/lm_head are not executed and vectors/scores match.
Then inspect joint/query interaction and lower-rank response compression from
existing caches. Do not reopen ordinary mean grids, old queues, or protected
labels. The research goal remains active; no user decision is needed now.

Read`reports/grain_vector_mechanisms/S12_QUERY_REPRESENTATION_REPORT.md` for
all comparisons, exact formula, source references and reproduction commands.

Final focused validation after S12:225 tests passed in18.72seconds, with only
two existing SWIG deprecation warnings. Ruff and whitespace checks passed.
S12 finding commit:f40a650; parent handoff sync:5f5a94c. No remote push.

## Historical checkpoint: S13 complete; S14 shared-query collection

S13 replays72/72 source views and48/48 frozen reader predictions with the
prefix-only execution path. Maximum vector error0, reader error1.31e-16;
forbidden suffix calls0. Paired median prefix/full extraction time ratio.6121.
This is a measured execution shortcut, not a new human prediction score.
See `reports/grain_vector_mechanisms/S13_PREFIX_READOUT_REPORT.md`.

S14 now fixes all213 N authors,3views and the existing KV/PCA C readers.
Replace native per-author queries with author-out fold references at alpha0/.5/1,
keeping KV fixed. Add first/second query directional cumulants from the same
cache,0 extra forwards. Protocol V2/code committed as d4b0948 before collection.
Private r1 was prepared but never collected; r2 is the live source.
Inspect `results/grain_vector_mechanisms/shared_query_s14_r2/run_status.json`.
Do not launch another neural collector or read interim questionnaire outcomes.

Theory V4, `GRAIN_RELATIONAL_READOUT_THEORY_V4.md`, connects paired KV evidence,
shared-query exponential tilting and conditional moment compression. Two new
finite tests verify that identical key/value marginals can yield opposite
paired readouts, and that the zero-query derivative is a within-comment joint
moment. The source experiment decides how much of this geometry is useful.

## Historical checkpoint: S14 complete; S15 conditional-moment pilot

S14 completed639 prefix calls in590.30seconds,0 suffix/backward calls. Common
query target-C r=.315627 versus native.320283; paired delta-.004656
[-.030492,.036014]. Source N r=.133706/.142869, all fixed doses reported.
Query references are already near native (median cosine.999191). Their small
personalized increment is reconstructed by the unfitted second-order cumulant
formula at r=.999416,MSE/zero=.001185. No new human prediction-accuracy gain is
claimed from that internal reconstruction. All source scores froze before Y.

S15 now tests four fixed operators at the common query: exact pairing, uniform
mass plus joint covariance, exact mass plus joint covariance, and Gaussian
mass approximation plus joint covariance. This separates within-comment higher
moments from normalization. The24-author/72-prefix-call resource pilot runs
without outcome access. Inspect progress in
`results/grain_vector_mechanisms/conditional_moments_s15_r1/run_status.json`.
Use `run_grain_conditional_moments.py inspect` after the pilot, and decide the
value of completing213 authors before resuming with `--max-authors 213`.
The full run cap is639 calls/25minutes. Do not launch another neural process.

S13/S14 reports, theory V4 and S15 protocol/code are committed locally.
The active research goal remains open. No new user input is required.

## Historical checkpoint: S13-S15 complete, S16 gain decomposition next

S15 completed all213 authors/639prefix calls in583.01seconds, including the
label-free24-author resource pilot. Exact-mass joint covariance reconstructs
the exact common-query reader at r=.847179,MSE/exact-author-variance=.701136.
Uniform weighting r=.415440,error ratio1034.28; Gaussian mass approximation
r=.594964,error ratio3.797855. Keep exact attention mass and exact reading as
the active source route; retain covariance as an explanatory approximation.
Source N r=.104370 for exact-mass covariance versus exact.133706, no advantage.
All four methods and paired source differences are reported; no recalibration.

No neural experiment remains running at this checkpoint. Next implement the
fixed24-author S16 gain/allocation/conditional-moment pilot in
`GRAIN_READOUT_GAIN_S16_PLAN.md`. It separates total body attention from
which comment is selected, then tests one fixed second-order conditional term.
72prefix calls maximum, no human outcomes. Do not expand it before seeing which
mechanism is decision-relevant. Primary sources are verified in the attention
mass note; no need to invoke Claude or ask the user for a new idea now.

Latest results: S13/S14/S15 reports in `reports/grain_vector_mechanisms/`.
Theory V4 preserves the exact relational, conditional and exponential-tilt
formulas, with explicit finite approximations.67 focused tests pass, Ruff and
diff whitespace checks pass. Active goal remains open; private outputs ignored,
no remote push, protected-label restrictions and source T/F control unchanged.

## Historical checkpoint: S16-S17 complete; S18 reader-pattern intervention

S16 completed24 authors/72prefixes/67.12seconds, no questionnaire join. Gain,
relative comment allocation and conditional curvature each contribute. Fixed
second-order conditional expansion improves exact-reader r=.773 -> .948 and
MSE/exact-author-variance=.742 -> .222 on this pilot.

S17 reused those24 authors and collected189 more/567prefixes/520.33seconds.
New189: second-order r=.920711,error=.318792; first-order .854969/.708774.
All213: second-order .925021/.301820. Target-C r=.307056; source N r=.125430
versus exact.133706, paired delta-.008276[-.062050,.046117]. Formula selection
preceded continuation; full scores froze before exposed source Y connection.
Keep exact reading as reference, second order as useful mechanism approximation.

S18 is now prepared and running as a separate168-forward/24-author/view0 pilot.
Read `results/grain_vector_mechanisms/reader_pattern_s18_r1/run_status.json` for
live state. It converts each frozen fold reader beta into a training-covariance
activation pattern and compares it with Euclidean projection. Remove the
reader-selected and reader-null pairing components separately at the actual
layer17 head output. Keep nonlinear interaction; no outcome fit or label access.
The question is whether the learned readout corresponds to an operative vector
pathway, not whether another hyperparameter improves a benchmark.

S16-S17 reports and theory V4 are updated. S17 code/protocol a306320; S18
code/protocol62d4da0.42 focused tests pass after the new geometry/continuation
modules. One neural collector only; no Claude, remote push, LoRA, private text
upload, protected data or old queue. Active goal remains open.

## Historical checkpoint: S18 complete; S19 functional-gradient recovery next

S18 completed24source authors/view0,168forwards,0backwards in242.76seconds.
No human outcomes. Covariance-pattern selected effect RMS=.0002687 versus
Euclidean.0000168 and full pairing-removal.001319. But reader-null components
retain.001245/.001314; correlations with full effects.98377/.99994. Their
linear-reader scores vanish below9e-18. Source native/vector/self replays are0.
This is the observed readout-versus-functional-response gap, not a failed source
configuration signal. See the S18 report for all magnitudes, CIs and norm caveat.

No neural collector remains running. Next implement the frozen scope in
`GRAIN_FUNCTIONAL_GRADIENT_S19_PLAN.md`: actual suffix gradients along the same
pairing path, integrated completeness, author-out mean-gradient prediction and
fixed-rank active-gradient diagnostics.24authors, view0, up to144forwards and
120backwards,15-minute cap; first8 numerical/resource checkpoint. This is a new
mechanism question, not another label-selected prediction grid.

S16/S17/S18 reports and theory V4 are current.42 focused tests pass. S17 finding
commit1e798b6; S18 code62d4da0. Keep T/F and exact N reference unchanged. Goal
active; no user decision is needed. Do not reopen protected labels, run Claude,
push private artifacts, restart old queues or alter frozen S12-S18 implementations.

## Historical checkpoint: S19-S20 complete; functional source link transported

S19 completed24authors,96forwards/72backwards,183.26seconds. Path completeness
r=.99999977, endpoint replay0. Leave-author-out mean-gradient reconstruction
r=.8865[.7149,.9675], first sensitivity direction95.117% energy. Preserve r1's
MPS export failure; r2 fixes CPU/Float64 ordering only, with an actual MPS test.

S20 freezes that24-author mean and all213x3 source scores, then measures24
new authors with48forwards/0backwards in67.24seconds. Functional response
transport r=.9431[.9009,.9809]. On the189 authors outside the gradient reference,
the frozen three-view score has source N r=.2000,p=.0058,CI[.0589,.3332].
All213 r=.1749; reference24 r=-.0079. The +.0425 primary difference versus the
S12 statistical reader has CI[-.1521,.2377]. Useful functional source association
is established on the exposed source, not a new benchmark superiority claim.

No neural collector remains running. Next implement
`GRAIN_CROSS_QUERY_FUNCTIONAL_S21_PLAN.md`: a matched numeric-format query
family and cross-reader matrix to locate shared versus query-specific pathways.
Use12reference and12transfer authors, reuse N caches, at most288newforwards and
180backwards with adaptive nodes,ten-minute cap. Finalize exact question texts
and hashes before preparation; no outcome-based selection or questionnaire join.
This is a mainline location-of-specificity question, not more defensive controls.

S19/S20 reports and theoryV4 are current;52 focused tests pass, two existing SWIG
warnings only. Code8b8d5f2/4bbbd7c. Active goal remains open; no user decision is
needed. Preserve T/F, release scorer, protected labels, old queues and unrelated
changes. Local commits only. Continue independently; do not request a new idea.

## Historical checkpoint: S21-S24 complete; S25 actual configuration source link next

S21 completes12reference/12transfer authors across four matched numeric queries.
216newforwards/108backwards,411.41seconds. Own-query reconstruction r is
.9307/.9486/.8015/.6374 (N/confirmation/Feeling/content). N with the content
reader gives-.0716, but with confirmation.8266. This finds shared and distinct
operative directions despite very similar author/query pairing vectors.

S22 defines the empirical response-function metric K=G Sigma_D G^T, invariant
under invertible coordinate changes, and retains both content-aligned and
orthogonal functions. Primary N/confirmation functional correlation.7150;
first two query-function combinations contain94.76% of reference Gram trace.
The N orthogonal function still reconstructs its transfer response at.9247.
This is reference-defined function geometry, not newly named personality factors.

S23 preserves S20's original24-author gradient and all213source predictions.
On189 reference-excluded authors, total/aligned/orthogonal source N r is
.2000/-.0344/.1946. Orthogonal p=.00730, bootstrapCI[.0529,.3298]; its paired
delta versus total-.0054[-.0472,.0346]. The exact covariance identity locates
the association primarily outside this content-reader function. Keep both
parts; do not declare a noise deletion or a prediction-accuracy breakthrough.
S22 andS23 need no new neural calls, no outcome-based fit or sign selection.

S24 code/protocol d51628c completes the actual S9 native/projected body contrast
on24authors/view0 with four hidden-state factorial cells. Total96forwards,
0backwards,139.51seconds. All source endpoint and additive replay errors0.
The original S20 gradient predicts the new symmetric head allocation at
r=.95433[.88975,.97925], error/allocation-energy=.022989. The allocation itself
correlates with C at.70647; other paths remain important. Head and other signed
means partly cancel, so no RMS ratio is called a human variance fraction.
No source questionnaire join or new fitting in this24-author mechanism packet.

No neural collector is running. Next implement `GRAIN_CONFIGURATION_SOURCE_S25_PLAN.md`:
one prefix plus local KV reprojection per view recovers the native/projected
head difference, then uses fixed S20 gradient on all213three-view authors.
Pilot8S24authors checks this shortcut's head replay, zero final-query change
and forbidden suffix calls before continuing. Preferred639prefix calls,20minute
cap; conservative fallback at most1278prefixes. Freeze scores before source N.
Do not add overlapping pairing/configuration scores or fit a blend on source Y.

Reports S21-S24, theoryV4 sections12-13 and the S25 plan are the current reasoning
chain.85 focused tests pass, Ruff passes, two existing SWIG warnings only.
Goal remains active; continue independently. The next step follows the measured
source configuration path rather than enlarging the prompt/gradient grid.
T/F and released scoring unchanged; old queues/protected data/cloud uploads/push
remain off. No new user decision is required at this checkpoint.

## Historical checkpoint: S25 source collection running; S26 retained-path accounting ready

S25 implementation b6d2a53 is prepared and its eight-author/three-view pilot
passes:24prefix calls plus24local KV reprojections in25.40seconds. Actual S24
head replay error3.02e-6, scalar replay5.14e-8. No final-query change or suffix
calls. The24-author functional gradient stays frozen.213-author collection
is now running, with a projected676second cost inside the1200second cap.
Inspect `configuration_source_s25_r1/run_status.json` and the live process handle
before any new collector. Do not inspect source outcomes before complete scores.

During collection S26 was specified and implemented without viewing S25 outcomes:
Delta_head=d_pairing+r_remaining, so the shared covector gives
z_config=z_pair+z_remaining. No part is deleted or declared independent.
The zero-neural-call follow-up checks exact source covariance and variance
accounting to locate reinforcement/cancellation. It will run only after S25
scores freeze. See `GRAIN_CONFIGURATION_ACCOUNTING_S26_PLAN.md`.

Focused S25 plus inherited tests:88passed. S26 adds3 passing finite algebra
tests. No new model fit, protected labels, remote transmission or old queue.

## Current checkpoint: S25-S26 complete; S27 query-carried configuration next

S25 completes213authors/639views with639prefix calls plus639local KV reprojections
in590.43seconds. No suffix/backward calls. All24S24/view0 actual heads replay
to3.78e-6 and frozen scores to5.14e-8; query differences0. The compute shortcut
is now verified, with no dual-prefix fallback. Code b6d2a53.

Primary189 source N r: new configuration head.1160,p=.1119; pairing.2000,p=.0058;
full S9 C.1989. Paired new-minus-pair delta-.0840[-.2131,.0315]. All213 new.1104
versus pairing.1749. These are exposed source observations, not a new benchmark.
The useful pairing route remains the reference; no source-selected blend.

S26, pre-outcome plan/code e5d51e2, needs no new neural calls and.81seconds
analysis. New configuration=pairing+retained remainder exactly. Primary source
covariance retention96.20%, variance ratio2.74935: this yields the observed
r ratio.580168. Remaining scalar N r=-.0060 but all three view-pair r are near.39.
This is criterion-relative variance dilution, not evidence that stable remaining
structure is globally noise. All213 remaining N r=.00039. Exact accounts and
all source cohorts are retained. Vector addition error5.55e-17.

Normalization-order prework e0fc5f8 proves a concrete mathematical candidate:
within-group dispersion can become a mean shift through R(PH)-P(RH). Its
Hessian formula agrees with autograd and has the predicted small-scale error
on finite examples. It has not been measured as the cause of the source
remainder. Defer a broad normalization experiment; prioritize the actual C path.

No neural process remains running. Next implement `GRAIN_QUERY_CARRIER_S27_PLAN.md`:
24authors/view0, native/projected residual-query states at blocks16/17/20/23/26/27;
cross-patch the four interior cuts and measure two-order query-carried allocation.
At most240forwards,0backwards,10minutes; first8resource checkpoint. No source Y
join. Endpoint27 recovers C by construction, so finding useful earlier transport
is the discriminating question, not claiming the mandatory endpoint as success.

Reports S25/S26 and theoryV4 section14 are current.94focused tests pass, Ruff
and whitespace checks pass, two existing SWIG warnings only. Goal active;
continue independently without Claude or a new user idea. T/F, released scores,
old queues, protected questionnaires/lockboxes and unrelated changes preserved.
