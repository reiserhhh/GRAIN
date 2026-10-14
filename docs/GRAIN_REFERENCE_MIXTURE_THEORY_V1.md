# Source-reference composition and author comparison

2026-10-13 JST. HM241 complete, verified; goal active, no live job.
Canonical `results/grain_math_response_v1/reference_mixture_r1`.
Original18 protocols/100cards/44questions remain OPEN.

## 1. What has been added

A cold comparison can now state whether author u is above author v throughout
the ENTIRE declared convex family of two fixed source instruments, or which
reference weights change that order. It returns a partial order, not a forced
total author ranking. No prediction accuracy, encoder, bandwidth or fitted
coefficient was optimized. No old score or failed result was overwritten.

On SAME24 historical authors/full histories/all5 targets/both inherited laws,
1425/2760 target-pair comparisons (51.6304%) have a constant order throughout
the family;1335/2760 (48.3696%) cross. All1425 endpoint-consensus comparisons
stay ordered inside the family. No hidden interior reversal was found, also
on all8 inherited edited frames. These are dependent comparisons of24 authors,
not2760 independent participants, psychological accuracy or a null test.

Paired-law invariant orders E/N/A/C/O:87/150/184/125/169 out of276 per target.
Product-law orders:90/143/188/119/170. All actual comparisons had resolved
numeric status; near ties remain explicit in the general interface.

## 2. Reference mixture is not posterior-score averaging

Let b index SOURCE0/1; g one of32 target profiles; x addressed SAME evidence.
Keep the common prior pi(g)=1/32 and inherited conditional likelihood L_b.
Define joint mass f_b(g,x)=pi(g)L_b(x|g), evidence Z_b(x)=sum_g f_b(g,x).
For a declared source-composition weight lambda:

    f_lambda=(1-lambda)f_0+lambda f_1
    p_lambda(g|x)=f_lambda(g,x)/sum_h f_lambda(h,x)
    alpha_x(lambda)=lambda Z_1(x)/[(1-lambda)Z_0(x)+lambda Z_1(x)]
    p_lambda=(1-alpha_x)p_0+alpha_x p_1.

The weights depend on the evidence. A common-weight average of normalized
posteriors is a different observation law. Complete normalized posteriors
are sufficient for target inference WITHIN their fixed experiment, but they
do not by themselves determine the relative source normalizers needed for
this experiment change. Retain both profile shape and source evidence mass.

Here both source banks have identical Gaussian variance and dimension, so
only the identical density-normalizing constant is omitted. Query-distance
terms are included in the copied log masses. Removing a factor common to
both banks for ONE query leaves all results unchanged; independently
normalizing banks destroys their relative evidence weights. Densities in
incompatible coordinates/units cannot be mixed using this contract.

The controlled positive toy has u above v by+.1 in both banks but-.586139
at lambda=.5. Source1 responsibilities are.990099 and.009901 respectively.
Its two crossings are.00166412 and.99833588. Ordinary posterior averaging
remains+.1. This is an illustration of a known aggregation mechanism, not
new psychological data. Actual full-history responsibilities at lambda=.5
span only.501531--.514919; the toy's extreme weighting is absent here.

## 3. Finite algebra gives a whole-family comparison

For a target statistic Y_q, retain

    A_ub=sum_g Y_q(g) f_b(g,x_u), B_ub=sum_g f_b(g,x_u)>0
    s_u(lambda)=(A_u0+lambda dA_u)/(B_u0+lambda dB_u).

With dA_u=A_u1-A_u0 and dB_u=B_u1-B_u0, the sign of s_u-s_v equals the sign
of a quadratic c0+c1 lambda+c2 lambda^2, since both denominators are positive:

    c0=A_u0 B_v0-A_v0 B_u0
    c1=dA_u B_v0+A_u0 dB_v-dA_v B_u0-A_v0 dB_u
    c2=dA_u dB_v-dA_v dB_u.

Its endpoints and an interior vertex, if any, determine min/max numerator.
Real interior roots partition orders. The derivative of score difference has
numerator

    D_u(B_v0+lambda dB_v)^2-D_v(B_u0+lambda dB_u)^2
    D_u=dA_u B_u0-A_u0 dB_u.

This is also quadratic; its roots plus endpoints give the complete score-
difference range. No lambda grid, optimization, Monte Carlo or accuracy
selection is needed. Floating-point certificates use the frozen1e-12
margin; unresolved boundary cases are not promoted to exact strict orders.
The retained coefficients, ranges and roots permit a literal readback.

Common per-author density rescaling multiplies a comparison numerator by a
positive factor and cannot change its sign. Exchanging banks maps lambda
to1-lambda; exchanging authors negates the difference. Same-author comparison
is a tie. Any strictly invariant pair relation is transitive as an intersection
of reference-specific scalar orders, but not every author pair is comparable.

## 4. Meaning for the mainline

The large actual crossing count is already visible in the two endpoint
orders. In this finite family the extra evidence-weighting mechanism does
NOT create otherwise hidden reversals. The constructive object is therefore
the qualified partial order plus the explicit reference-sensitive edges.
Do not choose lambda to force a preferred ranking or tune it to fit model
answers. No source-dependent component is deleted as noise.

The two banks are frozen partitions of historical SOURCE, not two newly
identified cultures or psychological populations. Sampling uncertainty and
source heterogeneity have not been separated by this computation. It is a
comparison-contract finding, not a causal account of personality change.
SOURCE target bits remain externally anchored labels of the inherited
instrument; an ordering certificate does not establish the target's human
validity. Existing E5 inputs remain neural, though this backend needs no LLM
decoder, application questionnaires or natural-language explanations.

HM229 already gave distributional comparisons under one source law; HM233
already joined source meaning to model action; HM240 retained joint updates.
The specific increment is their missing source-composition-aware comparison
interface. Bayesian averaging, quadratic extrema, and Simpson reversal are
not new mathematics. Source-order changes are not a newly discovered trait.

## 5. Prior art and OpenAI/math

[Kamary et al., 2018](https://arxiv.org/abs/1711.10016) studies Bayesian model
averaging through a mixture and posterior model probabilities; abstract read,
not a claim that its full derivations or estimator were reproduced.
[Kievit et al., 2013](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00513/full)
places aggregation reversals in psychological research. HM241's result is
not discovery of Simpson's paradox or a causal explanation of it.

OpenAI/math adc7f124 three-state foundations110--212 was actually reread.
It states posterior sufficiency inside an existing experiment and retains
a marginal density tilt in experiment composition. This inspired retention
of source normalizers; its tree theorem/independence hypotheses were not
transferred. Finite Bayes supplies the actual formula here.

## 6. Execution, reproduction and next decision

Prepare/compute/verify/check each ran once. Compute1.013561s CPU; zero new
neural/API/private/training/application-label calls. All24840 comparisons,
864 evidence-source records,90 summaries and partial-order edges remain.
3072 direct-distance common-mass values checked with max log error3.02e-14;
86954 analytic-point accounts max2.47e-13; root probability error3.85e-16;
old normalized endpoints6.66e-15. Separate cold process exact readback.
This is same-agent alternate arithmetic, not independent blind replication.
34focused tests and three new Python lint/format checks pass.

A pre-freeze synthetic-control failure and its repair are recorded in
`GRAIN_REFERENCE_MIXTURE_PREFLIGHT_HM241.md`; scientific calculations did not
fail or rerun. All previous scientific code/input/result seals stay unchanged.

Read-only checks from the implementation repository:

    PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
    "$PY" scripts/run_grain_reference_mixture_hm241.py check
    "$PY" scripts/run_grain_reference_mixture_hm241.py cold

Stop this reference-mixture question here: no more weights, source splits,
precision, robustness grids or sharpening. Next inventory existing SAME-event
source anchor/condition law and criterion-specific vector action before a
new finite plan. A new packet must add an author-observation/target-meaning
construction, not merely certify more rankings or imitate the neural reader.
Choice, state, dynamics and configuration remain in the original scope.
