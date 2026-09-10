# Complete conditional response: magnitude, direction, and source reading

2026-09-10. Twenty-first finite logic pass, NEXT33. Source construction `95f472b`,
held assessment `d8bb93f`, subsequent no-fit directional diagnostic `d3b0f7b`.

## 1. A useful baseline now belongs to an executable composed function

Four old N reference authors define the same geometry, heads and baseline;
twenty old evaluation authors are never used to fit these quantities. Both old
comment pairs and both fixed Fourier representations are retained. The original
random projection is regenerated exactly from its data-independent seed.
Geometry center/scale use only4 source authors x3 old views, not18-source assets.

For a conditional comment pair i,j:

    zhat_ij = bhat(M,R) + dhat_i(c,phi_i) + dhat_j(c,phi_j)
              + mhat_ij(phi_i,phi_j),
    scorehat_ij = S_original(zhat_ij).

bhat is the previous source-average Jacobian function, including its source
reader-position covariance. The singleton kernel is

    k_single = (1+k_context(c,c')) <phi_i,phi_i'>.

The mixed kernel is the inner product of symmetric tensor features:

    k_pair = [(a.c)(b.d)+(a.d)(b.c)]/2.

Five heads are fitted per construction: one matched kernel baseline, two
singleton heads and two mixed heads, all penalty .01. The affine baseline is
not refitted. Zero contrast gives zero increment; exchanging partner comments
exchanges pair outputs. The independent scoring process can open only fixed
numerical assets and current numerical inputs, not current teacher endpoints.

All of this operates on block16 hidden inputs. It is a complete conditional pair
reader, not full natural-text F and not human trait validation. An exact runtime
does not imply an accurate approximation; both are measured separately.

## 2. What the composition actually recovers

| Fixed family | Affine complete r | Matched kernel complete r | Affine complete RMSE | Baseline-only RMSE |
|---|---:|---:|---:|---:|
| Fourier bag | .610 | -.095 | .014470 | .015018 |
| Fourier position | .593 | .092 | .014322 | .015018 |

All40 pair observations belong to20 authors; paired bootstrap resamples whole
authors. Pooled MSE-difference intervals versus kernel and baseline-only cross0.
The finding is a usable local composition, not a new confirmed improvement in
human prediction. The original baseline r=.727 uses a different target B;
this table uses the native pair target, so the two r values are not a regression.

The native baseline itself correlates .860/.944 with adjacent/distant pair
scores. Baseline-only r=.602 is consequently substantial. The complete score
therefore cannot certify accurate configuration reading. Adjacent mixed-logit
MSE/zero=.783/.709 and norm r=.338/.364 are useful local interaction results,
but distant mixed reading remains weak. Previous18-source findings are not
erased by this separate4-source test.

## 3. The directional bottleneck is now quantified

The predicted *magnitude* of adjacent configuration action has r=.637/.721.
Its author-centered vector correlation is -.009/-.065, and native score-change
correlation is -.214/-.321 at the affine base. These are different functions
of the same vector: norm recovery does not determine signed query response.

For the known candidate score S and a logit displacement d, the fundamental
theorem of calculus gives

    Delta S = S(b+d)-S(b) = vbar(b,d)^T d,
    vbar(b,d) = integral_0^1 grad S(b+t d) dt,
    grad S(z)_k = p_k(z) [value_k-S(z)],
    Delta S = ||d|| ||vbar|| cos(theta).

This is the same straight-path completeness principle used in
[Integrated Gradients, Sundararajan et al. (2017), section3](https://proceedings.mlr.press/v70/sundararajan17a.html).
Our calculation integrates only the known numerical output link, not a newly
traced transformer circuit or a new attribution theorem. Fixed16-node Gaussian
quadrature reconstructs the score actions in these80 paired diagnostic records
within2.39e-16 (the native path is reused across input families).

At a common true baseline, adjacent signed-alignment r=-.315/-.312 despite the
positive norm association. The native score-link tangent is already accurate:
r=.9981 adjacent, .9998 distant. This locates the useful next question before
the final score link: how the configuration vector acquires its query direction.

## 4. An exact diagnostic error account

Write d=n u, d_hat=n_hat u_hat. Using native and predicted path readers v,v_hat
at the same true baseline, one ordered telescoping gives

    Delta S_hat - Delta S
      = (n_hat-n) v^T u
        + n_hat v^T (u_hat-u)
        + n_hat (v_hat-v)^T u_hat.

The terms correspond to magnitude, orientation, and output-link change. Their
sum reproduces the observed error within3.35e-16. Adjacent RMS values are:

| Family | Magnitude term | Orientation term | Output-link term |
|---|---:|---:|---:|
| bag | .004086 | .005255 | .000048 |
| position | .003954 | .005763 | .000159 |

Magnitude also remains imperfect; orientation is not the only error. Reordering
the telescoping changes allocations, and squared terms have signed cross terms.
These are not independent causal/variance shares. The native baseline/readers
are explicitly oracle diagnostics, never supplied to the deployed candidate.

## 5. Implication for the GRAIN connection

Group-referenced expression positions and dispersion can organize *where* and
*how much* configuration changes. A query-specific local reader determines the
signed projection of that change. This is a compatible connection, not automatic
equivalence between GRAIN factors and questionnaire factors. The earlier exact
same-material Cw/localized-reader identities remain the appropriate interface.

The next experiment should characterize position-specific native readers before
compressing them. Uniform token reading cannot recover all within-comment
mean-null contrasts: sum_t d_t=0 implies sum_t w^T d_t=0 for constant w. Existing
native GRAIN directions are not all of that special form, so this is a precise
input-family statement, not a claim that every shared reader must fail.

Follow [NEXT31 position-specific plan](GRAIN_POSITION_SPECIFIC_READER_PLAN_V1.md),
not score calibration, another symmetric-kernel grid, or human-label selection.
This pass improves the usable assembly and identifies which local mechanism to
develop next; it does not close the unrestricted personality-inference theory.
