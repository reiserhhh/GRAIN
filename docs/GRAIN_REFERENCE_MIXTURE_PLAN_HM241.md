# HM241: SOURCE-mixture author comparison, not precision optimization

## Question and competing mechanisms

HM240 retains complete32-profile endpoints but not their source evidence
normalizers. HM209 retains the fixed SOURCE banks and SAME application vectors.
Do two reference-specific author orderings agreeing at the endpoints also agree
under every convex mixture of those source observation laws?

Mechanism A: mix normalized posterior readings with one common weight. Agreement
then persists by convexity. Mechanism B: mix likelihoods BEFORE normalization;
source responsibilities depend on each author's evidence. The two constructions
coincide only under additional conditions, not by definition. They may differ
without any encoder, reader, noise removal, or prediction-accuracy change.

## Frozen scope before calculation

Reuse HM209 SOURCE203/183, variance, uniform32-profile prior, SAME24 authors,
all9 frames, both positive paired/product laws. Read only registered numerical
panel points, source prototypes/codes/variance, old normalized panel laws and
opaque CSV addresses. No application labels, raw text, native responses, model
forwards, APIs, private material, fitting, or Gaussian-width changes.

Primary: full-history276 unordered author pairs x2 laws x5 targets =2760
dependent target-pair accounts. Secondary: all8 inherited edited frames,
22080 accounts. These are24 historical authors, not24840 participants.
No select-best reference, post-outcome relabeling, or subset rescue.

Recover common-unit profile likelihoods from existing source vectors. Both
sources have identical variance, dimension and profile prior; Gaussian factors
depending only on the query are common to both and may cancel. Independent
normalization of sources must NOT erase their relative evidence masses.

## Mathematics and new construction ability

For author u and source b, retain A_ub=sum_g bit_q(g)*p(g)*L_b(x_u|g),
B_ub=sum_g p(g)*L_b(x_u|g). Mixture score is

    s_u(lambda)=(A_u0+lambda*(A_u1-A_u0))/(B_u0+lambda*(B_u1-B_u0)).

Comparison sign is a quadratic after multiplying by positive denominators.
Its endpoints and optional vertex give sign over the ENTIRE [0,1] family.
Roots partition reference-dependent orders; derivative roots give the full
score-difference range, without a precision/weight grid. Keep near ties as
unresolved, not zero. A cold numeric comparison returns all five certificates,
crossings, evidence-dependent source weights and both endpoint laws.

The useful increment is a source-qualified partial order with explicit
reference-dependent comparisons, not Bayesian averaging as new mathematics.
HM229 already implements single-law distributional comparison; HM233 already
joins source meaning and model action. Do not rediscover either as novelty.
HM241 isolates a missing reference-composition rule on their inherited evidence.

## Controls and acceptance

1. Positive toy law: both endpoint orders positive but interior negative.
   Compare it with common-weight posterior averaging, which stays positive.
2. Identical sources and equal evidence-ratio sources preserve endpoint order.
3. Author swap, source swap/lambda reversal, common per-author density rescaling,
   self comparisons and invertible profile permutation preserve proper accounts.
4. Independently normalizing the two banks can change the toy conclusion;
   this intentionally tests loss of evidence mass, not a scoring repair.
5. Replay ALL inherited normalized endpoints before conclusions. Audit actual
   common-unit likelihoods with direct-distance Gaussian component sums.
6. Compare all certificates at analytic extrema/roots with direct mixed-law
   calculations. Run a separate cold process and focused tests once.

Success is a functioning comparison law plus a mechanism answer, regardless of
whether real interior reversals occur. A numerical improvement is NOT success.
Preserve all data cases and failures. Existing upstream E5 remains neural.
Finite-family robustness is neither human psychological validity nor all-source
transport. Sources are fixed instruments, not newly identified human populations.

## Prior art and OpenAI inspiration

[Kamary, Mengersen and Robert, 2018](https://arxiv.org/abs/1711.10016) describes
Bayesian model averaging via a mixture and posterior model probabilities
(abstract consulted); no estimator or asymptotic theorem imported.
[Kievit et al., 2013](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00513/full)
explains aggregation reversals in psychology. This packet is not discovery of
Simpson's paradox, a causal explanation, or a new robust-Bayes theorem.

OpenAI/math adc7f124 three-state 02-foundations.tex110--212 was reread: posterior
sufficiency is stated inside an existing experiment, and combining experiments
retains a marginal density tilt. This motivates keeping evidence normalizers;
its tree recursion theorem and conditional-independence assumptions do NOT
transfer to human text. The quadratic statement here is elementary finite Bayes.

## Deliverables and next decision

Exclusive prepare/compute/verify/check stages; common-unit likelihoods, complete
24840 comparisons, all90 target/frame/law summaries, source-relative partial
orders, toy counterexample, cold input/output, verification and immutable seals.
If endpoint consensus survives the entire mixture family, retain that qualified
comparison as usable. If not, require the reference-composition contract before
assigning author order. Do not tune lambda until agreement or accuracy improves.
Original18/100/44 scope remains OPEN.
