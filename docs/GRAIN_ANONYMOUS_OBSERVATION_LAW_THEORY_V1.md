# Unknown observation laws: what joint evidence can identify

2026-10-11, HM142. This extends the e/F/ell separation from HM134 and the
HM140/HM141 conditional-diagnostic construction. It is a finite constructive
identification result in this system, using established mixture mathematics.
It does not add three named traits or claim a new tensor theorem.

## Positive construction, not another reader accuracy target

Earlier packets supplied a choice law or an explicit population reference.
HM142 makes an unknown observation law constructible from anonymous source
observations. No native reader answers, Big5/MBTI labels, hidden component IDs,
or application outcomes enter the spectral estimator. Three source-only GRAIN15
vector alphabets provide a finite observation channel. They are9 existing public
action-contrast vectors, not9 newly found psychological items. Hypothetical
authors select categories from these alphabets in a declared simulation.

The empirical chain is:

    observed finite vector -> frozen alphabet category
    -> same-author three-event joint frequency
    -> source-only latent observation law
    -> qualified event likelihoods
    -> cold anonymous author configuration posterior.

The classifier count k=3 is imposed by the finite world, not inferred from data.
Nearest-alphabet decoding succeeds exactly because the simulator emits those
vectors. This does not establish arbitrary text embedding decoding or natural
psychological conditional independence. The alphabet origin is real frozen
GRAIN computation; its anonymous response law is simulated.

## Observable operator and normalization

Let x_v be a one-hot categorical observation for event channel v, and h an
anonymous component with weight pi_h. Assuming x1,x2,x3 independent given h,

    P12 = M1 diag(pi) M2^T,
    T_eta = M1 diag(M3^T eta) diag(pi) M2^T.

Use the top3 source singular vectors U,V and form

    B_eta = (U^T T_eta V)(U^T P12 V)^-1
          = (U^T M1) diag(M3^T eta)(U^T M1)^-1.

Distinct fixed-contraction eigenvalues recover M1 column directions. Their
categorical column sums identify scaling. The source marginal determines pi;
P12 gives M2, and coordinate contractions in the same eigenbasis give M3.
Components are ordered by the source eigenvalues, not their oracle class names.
The constructor's input is observed T, fixed k, fixed eta. The generator's
tables and latent labels remain evaluation-only. The simulation is not blinded
to the engineer; the numerical estimator is outcome/label-free.

Full column ranks here give Kruskal ranks3+3+3>=2*3+2, sufficient for uniqueness
up to permutation within the declared product-mixture model. Source recovery
does not prove that all alternative causal models are equivalent to that model.
Three distinct chunks or three LLM queries do not establish these assumptions.

Anandkumar/Hsu/Kakade2012 Sections2-3 provide the operator method:
https://proceedings.mlr.press/v23/anandkumar12/anandkumar12.pdf .
Allman/Matias/Rhodes2009 Theorem1/Corollary2 give the tensor identification
condition: https://www.cs.uaf.edu/~jrhodes/papers/Latent.pdf . This simplified
categorical implementation does not import a random-probe concentration bound
for its one fixed eta or solve nonparametric likelihoods from conditional means.

## A concrete second-order ambiguity, not more accurate second-order fitting

Let q=sqrt(pi), A_v=M_v diag(q), and Q be the frozen0.12-radian orthogonal
rotation. With q'=Q^T q positive, define

    pi'=q'^2,
    M'_v=A_v Q diag(1/q').

These matrices remain strictly positive and column-stochastic in the chosen
interior world. Each marginal and every pairwise distribution stay identical:

    M'_v pi' = M_v pi,
    M'_v diag(pi') M'_w^T = A_v Q Q^T A_w^T.

However, multiplication of three factors does not cancel Q the same way. The
two third-order laws have TV0.0290260458 despite maximum pairwise error
8.33e-17. Under the nearby component correspondence, conditional anonymous
class probabilities differ by up to0.213973479. This is not a personality-score
difference. Both laws are reconstructed from their own full tensor with
emission maximum errors4.44e-16/3.33e-16. More exact covariance fitting alone
cannot choose between them. The useful tool change is the observation order,
not a lower regression RMSE.

## Finite sampling and cold author inference

All3 fixed seeds succeeded,120000 simulated source authors each,6000 disjoint
application authors each. Total360000 source/18000 application instances, not
human participants. Each source has3 categorical events. Three application
bundles are the common first bundle and two alternative second-bundle worlds.
No hyperparameter or model selection was run.

| Seed | Emission RMSE | Joint-law reconstruction TV | Cold posterior RMSE |
| --- | ---: | ---: | ---: |
|142|.00413542|.00197681|.00708908|
|143|.00580606|.00394671|.00997155|
|144|.00742850|.00419369|.01353787|

These are estimator errors against the simulated law, not human personality
accuracy. Oracle permutation matching is evaluation only. Once constructed,
the numerical channel computes a posterior without an LLM:

    p(h|e_u) proportional to pi_h product_{distinct qualified events e} M_view(e)[category(e),h].

This is a likelihood channel for declared independent observations. Composition
keeps self/event qualification separate from category geometry. One event
repeated five times gives exactly the same score;12 repeated mentions add0
independent events in the3-event check. Unknown qualification is not given a
false0 value. Narrative repetition and context-choice can still be informative
through their separate GRAIN channels, not this independent-event product.

## Identified mixture is not an identified psychological cause

Stable-author h_u and per-occasion h_{u,r} can generate exactly the same source
single-bundle tensor and hence exactly the same fitted law. The second bundle
is another set of events, not a requirement to repeat the same verbal response
or prompt on another day. The two finite worlds retain the first bundle and
source data unchanged but differ in whether h persists.

Stable next-bundle predictive log gain over the population prior is
.29148/.29205/.27895 nats per3-event bundle(SE .00829-.00859). Per-occasion
world gain is negative: -.38318/-.39363/-.39630(SE .01281-.01301). A correct
single-bundle mixture can therefore be a bad stable-author predictor.

For view1 with mean mu=M1 pi, the population cross-occasion covariance is

    stable: M1 diag(pi) M1^T - mu mu^T,
    fresh independent occasion: 0.

The stable population Frobenius norm is.130284295. Empirical stable norms
.13317/.12519/.12116 versus occasion .00512/.00854/.00745. These are one-hot
covariance norms, not psychometric reliability r. More data identifies the
appropriate law within a model; it does not remove an alternative causal model
that produces the same data at the observed level.

The log-score sign has a mathematical explanation, not a threshold choice.
For the correct stable world, its population expected gain is I(X;Y)>=0.
Applying the same stable forecast to independent occasions gives
-E_X KL(p(Y)||p_stable(Y|X))<=0. A post-result read-only27-by27 enumeration of
the frozen population tables gives+.292188068 and-.387739562 nats, respectively.
This is an analytic follow-up, not another sample run or a new acceptance rule.
It explains why making the single-bundle model more accurate cannot repair an
incorrect assumption about the measurement object's persistence.

A copied-one-event diagonal tensor is an especially instructive case: spectral
reconstruction error0 and seemingly perfect three latent classes. They merely
name the category of one event copied three times. Declared event provenance
rejects this as three independent event views. Statistics alone cannot tell
whether the diagonal tensor represents a stable person or copied observation.

## Failures retained and next construction question

The rank-one population and uninformative third-view worlds are nonidentifying
for the fixed constructor. The third-view degeneration produced a non-real
numerical eigensystem around its repeated eigenvalues; no extra contraction
was tried and no scientific threshold was relaxed. Spectral estimates that
have substantive negative probabilities are retained failures. Passing this
small well-conditioned world is not a universal source-law learner.

HM142 changes the theoretical route: observer precision -> identifiable
multi-event source law -> author/condition/state decomposition -> psychological
anchoring. The next missing operation is a condition- and occasion-aware
source law that preserves author choices and personal responses without
silently treating a shared occasion as a stable author class. Inventory F02/F03/
F04, M04/M08/M11 and existing C1-C4 source laws first; do not rerun their finite
protocols or call standard random-effects/HMM mathematics a new invention.
Use a new selective shared-condition/independent-event design only if it
distinguishes an ambiguity those source laws have not already addressed.
Anonymous component naming or a real psychological likelihood requires
behavioral/psychological anchors, separate from source algebra.

The OpenAI/math pinned elasticity Introduction inspired the comparison of
physical observations rather than a noninjective auxiliary summary. Its PDE,
solutions and uniqueness theorem are not transferred here. This packet reads
the introduction as a design analogy, not as a validated personality theorem.

## Reproduction and state

Canonical results: results/grain_math_response_v1/multiview_law_r1.
Plan: docs/GRAIN_MULTIVIEW_LAW_PLAN_HM142.md.
Runner: scripts/run_grain_multiview_law_hm142.py prepare/run/verify (existing run
rejects prepare/run; verify is read-only).17 new/60 focused tests pass;3 new
Python files Ruff lint/format pass. First lint failed because the shared volume
marks files executable; shebangs added before freeze, no research failure.
CPU experiment wall .382054seconds. All54000 cold probability values, tensors,
models and metrics replay; direct-Bayes error3.89e-16, alternate metric5.55e-17.
Same-agent alternate arithmetic, not independent blind replication.
No neural/API/private/training calls, Git changes, old-queue replay or background
jobs. Goal remains ACTIVE; original18/100/44 remains OPEN.
