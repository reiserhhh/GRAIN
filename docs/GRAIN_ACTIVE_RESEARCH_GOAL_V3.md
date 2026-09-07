# GRAIN autonomous research: functional information left behind

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

## Current checkpoint: S8 complete, S9 source experiment running

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
