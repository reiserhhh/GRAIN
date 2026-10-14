# Same-Event, Target-Qualified Evidence Response

HM199, 2026-10-12. Finite exploratory observations on the historical public
PANDORA panel. This document follows the sealed experiment, not a preregistered
new mathematical theorem or a psychological accuracy benchmark.

## 1. New empirical contact

The missing HM184 interface now has five actual reader queries on the SAME
comment edits, rather than five SOURCE directions compared only with N.
The strongest new diagonal is A: SOURCE matched full-vector action versus
native A score action has author-mean r = .699066 / .532352 in the two SOURCE
text views. The same SOURCE A action has r = -.385899 / -.369631 with native N.
Thus the empirical contact changes with the target query; a single identical
readout for all five targets does not describe this interface.

These are correlations between two responses to text edits, not between
predicted and observed personality scores. SOURCE labels supply the historical
field meaning; no application psychological labels enter HM199.

## 2. Observation and action are separate objects

Let X_uv be an author's twelve-comment view. S and C are the unchanged four
selected and four length-matched control comments from the old N-guided design.
g_S and g_C delete those respective event sets. The target-indexed action is

    D_uvq = F_q(g_S X_uv) - F_q(g_C X_uv).

For the frozen SOURCE field d_br and the mean comment embedding x:

    V_uvbr = d_br^T [x(g_S X_uv) - x(g_C X_uv)]
           = d_br^T (sum_C x_j - sum_S x_j)/8.

Here b is a SOURCE text view, r a SOURCE target, and q a reader query. Keep
all profile components and all full/retained/complement spaces. A useful
descriptive observation matrix is

    M_brq = Cov_u(mean_v V_uvbr, mean_v D_uvq).

The matrix is indexed by the evidence operation, source/reference construction,
reader target and readout unit. It is not an intrinsic, context-free personality
matrix. Off-diagonal entries are retained, not defined away as noise.

This avoids an invalid identification: a SOURCE contrast, a semantic query,
and the neural answer are not automatically the same function.

## 3. Shared and query-specific accounts

For the final-normalized reader-state action delta_h_uvq, define

    c_uv = (1/5) sum_q delta_h_uvq,
    s_uvq = delta_h_uvq - c_uv.

With the frozen common digit head w = mean(W_5..9) - mean(W_0..4):

    delta_ell_uvq = w^T c_uv + w^T s_uvq.
    Cov(V_r, delta_ell_q) = Cov(V_r, w^T c) + Cov(V_r, w^T s_q).

This is an exact descriptive account up to native floating-point measurement.
It does not intervene on c/s, remove input context, or identify two independent
causal pathways. Correlations and psychological information fractions do not
add by this identity.

For A/full, specific covariance is positive in both SOURCE views and both
application text views. Conditional author-cluster bootstrap intervals are
[1.9473e-5, 6.8632e-5] and [3.0555e-6, 5.3545e-5]. For N/full the common
covariance is negative while specific covariance is positive in both SOURCE
views. N is the inherited observation, not a new HM199 discovery.

Subtracting the query mean distinguishes an identical common readout from
query-dependent responses. It does NOT exclude one shared scalar latent with
different query gains/polarities, nor common evidence processed by different
nonlinear functions. Do not promote this accounting into a unique circuit.

## 4. GRAIN's retained space is not the whole target observation

Use the fixed HM184 projection P onto the original GRAIN15 span:

    d_r = P d_r + (I-P)d_r,
    V_r = V_r,retained + V_r,complement.

For the new A query, score-action correlations are:

| SOURCE view | Full | Retained | Complement |
| --- | ---: | ---: | ---: |
| 0 | .699066 | .236489 | .479209 |
| 1 | .532352 | .123805 | .488085 |

Complement-specific geometric covariances are positive with conditional
bootstrap intervals excluding zero in both SOURCE views. Retained intervals
cross zero. The complement is outside this fixed linear score interface; it
is not a demonstrated noise component, new named factor, or psychological
information share. Other GRAIN dynamic/tensor/nonlinear interfaces are not
being tested or rejected here.

The exact source full = retained + complement account has maximum discrepancy
2.29038e-18. This arithmetic is ordinary linear projection, not new mathematics.

## 5. Context remains part of the observation

The additive SOURCE mean action on isolated S/C is exactly -2 times its
deletion action. The actual reader's two context regimes need not obey that
law. The geometric and probability accounts for both operations remain in the
canonical outputs. HM174 already established N context nonadditivity: it must
not be counted as a new discovery here.

Do not turn an event edit into a fabricated new text vector, equate endpoint
patching with a natural edit, or call the unedited remainder irrelevant noise.

## 6. Mathematical lineage and scope

[Cook and Li (2002), Sections 1 and 2](https://www.stat.umn.edu/RegGraph/RecentDev/CookLi.pdf)
distinguish conditional-mean dimension reduction from the full conditional
distribution. This motivates declaring the target functional; HM199 estimates
neither a central mean subspace nor its existence/uniqueness conditions.

[OpenAI/math, pinned three-state foundations](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex),
lines 217-232, use conditional expectation and Jensen to compare degraded
observations for a specified target. This is design inspiration, not transfer
of the broadcast-tree reconstruction theorem to personality. A SOURCE field
or a neural answer is not automatically a calibrated posterior likelihood.

HM60, M06 and the older shared/trait-specific path work already motivate
cross-target comparisons. The empirical advance is their missing SAME natural
event/SOURCE/GRAIN/native interface, particularly the new A contact. Shared
response decomposition, covariance and projection are established tools.

## 7. What this enables next

The next decision concerns target-qualified evidence use, not prediction
precision: distinguish a shared carrier with query-dependent gain from
target-dependent evidence use, on the SAME evidence addresses. Inventory the
existing cross-query carrier/action interfaces before designing a new finite
intervention. Preserve A/N contacts, all off-diagonals, event qualifications,
condition selection, state and reference population. No prompt/rank/layer/head/
weight grid or compound-answer rescue follows from this result.

Evidence limitations are concrete: n=24 historical authors, N-guided comment
selection, previously used representation with partial panel overlap, two
SOURCE text views of the same population, and one local model. Historical N
was single-sequence; new E/A/C/O were batched. Bootstrap conditions on the
frozen fields and panel, with no multiplicity correction or new population
claim. The source/vector contact is useful at this exploratory level.

## 8. Canonical implementation

- `configs/grain_same_event_multitarget_hm199.json`
- `grain_research/construction/math_same_event_multitarget.py`
- `scripts/run_grain_same_event_multitarget_hm199.py`
- `tests/test_grain_same_event_multitarget.py`
- `results/grain_math_response_v1/same_event_multitarget_r1/`

240 new batches / 960 enquiries / 911.079423 model seconds; old N240 endpoints
reused without replay. All stages succeeded once; no answer fit or retry.
31 focused tests and three Python lint/format checks pass. One alternate-formula
audit replays 1200 native and 6000 covariance accounts; same agent, not blind.
Goal active/nojob; original 18 protocols/100 cards/44 computational questions
remain OPEN. No Git, API, private material, training, Claude, automation,
lockbox or old queue operation was performed.
