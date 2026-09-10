# NEXT34: content-conditioned token coefficients on a frozen shared basis

2026-09-10. Planned after NEXT31. Goal: turn useful shared directions into a
source-only local reader; test the same-material GRAIN/path connection directly.
No claim that this finite experiment completes the global logic goal.

## Fixed inputs and roles

- Reuse NEXT31's4 old authors x3 queries, selected body H (width1536), four GRAIN
  fields, and the12 frozen leave-one-author-out rank8 bases V_q,-u.
- For each held author, all preprocessing and fitting uses the other3 only.
  Query target is the same scalar natural full-state gradient, NOT the
  mean-null candidate Jacobian of NEXT32 or a human questionnaire score.
- Held J is read only by the evaluator, never by preprocessing or prediction.
  Current H/G are permitted inputs. No author-ID feature, raw-text inspection,
  MEPS, human Y, new neural call, lockbox opening or release scorer change.
- Source-only code must export a model and predict a held packet that contains
  H/G/selected-position coordinates, with no gradient. Verify target independence
  by changing the evaluator gradient and retaining exactly identical predictions.

## Two models, no grid

For each source token, the8 targets are density coefficients a=T J V^T.
Fit source-weighted mean and rank32 PCA of H with per-token weight1/(3T_u),
without whitening. Standardize projected components and four GRAIN fields with
source weights; zero-variance columns use scale1 and are recorded. PCA is fitted
once per source split from shared body states and reused for all three queries.
Record exact preprocessing hashes and source IDs privately.

Let h be the32 standardized coordinates; g the4 standardized GRAIN values;
p the3 nonconstant Legendre polynomials of selected-body ordinal midpoint.

1. **Content:** phi=[1,p,h],36 columns.
2. **Content + GRAIN:** phi=[1,p,h,g,vec(h outer g)],168 columns.

Standardize nonconstant design columns using source weights after assembling
each arm, intercept unpenalized. Fit weighted multioutput Ridge with objective

    sum_u (1/(3T_u)) sum_t ||a_ut - phi_ut W||^2
      + .01 ||W_nonintercept||_F^2.

alpha=.01 and PCA32 are finite engineering choices, not held-effect tuned.
Do not choose a best rank from NEXT31; reuse rank8. Prediction returns
Jhat=(phihat W)V/T and its contractions with the old GRAIN directions. GRAIN's
presence in both input and declared effect direction is intentional, so report
generic field error as well as GRAIN action error; an effect-only gain is not
the reconstruction of every native model direction.

## Comparators and success meanings

Keep NEXT31 source-position degree0/3, shared-rank8 oracle coefficients and own
rank8 diagnostic separately. For every query and factor report signed action,
RMSE/RMS, SSE/zero, descriptive r, same-sign count and field energy error;
include source fit errors to reveal underfit vs source-to-held loss. No p-value
claim from4 reused authors. Preserve all12 query/held folds and both arms.

- Better field/action fidelity than position-only supports content-dependent
  coefficients. Closeness to oracle measures the remaining coefficient gap.
- Added GRAIN improvement over content supports a same-material modulation
  mechanism candidate; field-only/action-only gains have different meanings.
- If both arms fit source well but fail held, next inspect source support and
  nonlinear state dependence, not a new sweep of regularization constants.
- If GRAIN does not help, keep a useful content reader; don't force merger.

## Bound, tests, deliverables

At most4 input PCA fits +24 small Ridge fits, zero neural calls. Monitor first
query cost; stop at complete units if projected numerical cost exceeds240s.
Repeat once for exact numerical replay; no hidden extra tuning. Checks: source
weights, centering, shape/position alignment, coefficient-target separation,
known bilinear toy recovery and model export inference. Public aggregate tables
and report; private coefficients/predictions/hashes stay under results.

The calculation still requires cached block16 content and source native
gradients. This is a step toward a compiled reader, not already raw-text inference
without an encoder. Statistical predictor links and the bounded psychodynamic/TA
interpretation protocol remain separate downstream uses, not training labels.
