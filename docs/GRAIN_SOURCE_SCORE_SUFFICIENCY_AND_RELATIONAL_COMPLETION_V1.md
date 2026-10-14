# GRAIN: source-score sufficiency and relational observations

2026-10-10 JST. HM78 is a result-after, CPU-only analysis of frozen HM68/HM69/HM71
observations. It changes the retained author observation schema, not reader
accuracy. Original 18 protocols /100 cards /44 computational questions remain
open. The mathematical tools below are established, not original theorems.

## 1. The measurement question

An author profile can summarize individual readings accurately while omitting
relations needed by a subsequent observation law. Improving the accuracy of
the retained means cannot distinguish profiles whose retained means are equal.
HM62 already demonstrated finite moment bounds; HM69--71 already supplied
equal-score profiles. HM78 quantifies the consequences for author comparisons
and tests fixed, source-only additions to the observation schema.

The measured support here has eight configurations c=(x,y,g): focal behavior,
companion behavior and companion ownership. The ownership manipulation also
swaps an anchor, so g is not a pure psychological variable. Correspondences
are declared by the material design, not inferred from unrestricted text.
Each context has six native readings F_q(c), q=1,...,6. The six targets are
extraversion, neuroticism, agreeableness, conscientiousness, openness and
anthropomorphism. Target names describe queries, not validated human traits.

## 2. Exact observation fibers and author comparisons

Let p be a probability distribution on the eight measured configurations and
put A=[1,F_1,...,F_6]^T. The retained summary is s=A p. All 60 source contexts
(30 directed pairs, two material families) have rank(A)=7. A unit vector h
spans ker(A), so the feasible profile fiber through p_0 is

    p(t)=p_0+t h,   A h=0,
    t_min=max_{h_i>0}(-p_0i/h_i),
    t_max=min_{h_i<0}(-p_0i/h_i).

Endpoints are nonnegative, normalized distributions; no clipping or
renormalization is used. For a different measured law v on the SAME support,

    theta_v(p(t))=v^T p_0+t v^T h.

Its extrema over [t_min,t_max] are attainable. For independent profile fibers
of u and w, their relative comparison ranges over

    [L_u-U_w, U_u-L_w].

These are identification intervals, not sampling confidence intervals. A
nonzero width is mathematical nonuniqueness, not necessarily a practically
large score difference. The means over all profiles are about .009--.014 on
the declared 0--1 native score scale.

On the full simplex, a linear target v is determined by A p for every p
exactly when v belongs to row(A). Sufficiency follows by writing v=A^T b;
necessity follows by perturbing an interior probability vector along a null
direction with nonzero v action. A boundary fiber can be smaller, including
a singleton; one boundary case does not imply global sufficiency.

## 3. What the omitted direction actually contains

Use the fixed orthonormal three-bit Walsh characters, not learned factor axes.
The average squared norm of h is distributed as follows across all 60 contexts:

| Character order | Fraction |
| --- | ---: |
| Single variable | .078392 |
| Two-variable relation | .634171 |
| Three-variable relation | .287437 |

Ownership-by-content characters xg and yg account for .462159 of the squared
norm; xy accounts for .172012 and xyg for .287437. These are fractions of the
UNOBSERVED PROFILE DIRECTION, not personality information or explained human
variance. Its direction also depends on material: mean squared alignment
between the two families' null vectors for the same pair is .379464, with
range .000026--.931998. No universal omitted psychological axis was found.

## 4. Positive finite construction: retain relations before aggregation

Six observation schemes were fixed before this analysis; none was selected
using the application score tables:

| Scheme | Additional observations | Source rank across 60 contexts |
| --- | --- | --- |
| scores_only | None | 7 in all 60 |
| owner_count | E[g] | 8 in all 60 |
| owner_response | E[yg] | 8 in all 60 |
| score_squares | Six E[F_q^2] | 8 in all 60 |
| score_cross_products | Fifteen E[F_q F_r], q<r | 8 in all 60 |
| score_all_products | All 21 second moments | 8 in all 60 |

The design-coded g/yg schemes require content/ownership observations. The
score-product schemes use the existing per-configuration reader; they do not
remove that dependency. Crucially,

    E[F_q F_r] != E[F_q] E[F_r]

in general. Multiplying already aggregated author means supplies no additional
observation. Retain paired readings of the same evidence unit first.

Full rank makes every linear law on the KNOWN eight-point support identifiable
and collapses the finite intervals. Because the original deficit is only one
dimension, any added function with nonzero action on h completes it. This is
generic finite interpolation, not an eight-factor psychological model or a
reason to treat ownership count as a universal solution. Application tables
are known cached measurements used for evaluation, not a new-text predictor.

## 5. Empirical consequences and verification

All 16 old profiles plus uniform were evaluated in each of 60 contexts:
1,020 hypothetical profiles, six schemes, three changed-law banks, six targets.
For DISTINCT source-score collision pairs (case same_scores_f{family}), both
future rankings are attainable in 360/360 comparisons in each bank. Their
mean relative-interval widths are .025432 (other family), .021936 (paraphrase),
and .033593 (reordered). This uses hypothetical compositions over measured
synthetic texts, not human authors changing personality.

The comparison summary also includes identical law_only profile pairs among
its source collisions. Do not present its 120-per-query count as 120 distinct
profile collisions; the distinct filter above has 60 pairs per bank.

The runner saved 110,160 target bounds, 51,840 comparison intervals and 2,160
linear programs. Analytic/LP difference is 5.551e-16; alternate cofactor and
endpoint/original-unit dual verification checks 1,136,169 scalars with maximum
error 1.024e-14. This is same-agent alternate arithmetic, not third-party
replication. Wall time is 3.036 seconds; no new model calls, fitting, training,
API or private data use. Eleven new tests plus 26 related tests pass.

Canonical results: results/grain_math_response_v1/score_fiber_relational_completion_r1.
Frozen plan: docs/GRAIN_SCORE_FIBER_RELATIONAL_COMPLETION_PLAN_HM78.md.

## 6. Literature and the next mainline decision

The [Bertsimas and Popescu working paper](https://flora.insead.edu/fichiersti_wp/inseadwp2004/2004-62.pdf)
provides an established moment-constraint/primal-dual framework. Read scope:
Definitions 1--2 and Section 2, PDF pages 2--9. Our finite linear programs do
not require their continuous semidefinite hierarchy.

The pinned [OpenAI/math metric/connection introduction](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/01-introduction.tex)
motivates an inverse-problem question: which objects does a specified family
of observations distinguish, and which ambiguities remain? Its manifold,
elliptic and unitary-connection hypotheses have not been established here;
its uniqueness theorem is not applied to LLM evidence. Read scope: the
155-line introduction, not the whole repository in this packet.

The useful GRAIN increment is a declared author-evidence schema whose retained
relations have explicit consequences for comparability under changed material.
Next develop a source-built evidence unit that retains joint GRAIN/query
observations and event/actor/condition identity before author aggregation.
Keep choice composition and state; do not remove common conditions by fiat.
Test an operational consequence on new units without an application eight-corner
table. Do not continue a moment-degree, head, layer, rank or fidelity grid.
