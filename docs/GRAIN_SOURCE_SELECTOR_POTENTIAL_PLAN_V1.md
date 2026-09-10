# NEXT74: source-trained query selector potential, with its own GRAIN derivative

2026-09-11. PLANNED after NEXT73. Return to numerical function construction,
not another selection/orientation/attention attribution study.

## Mechanism to develop

Source-query swap is primarily a selection-measure change in the current
source4/new8 test. Learn the *used scalar logits* rather than a generic hidden
state or unconditioned Gaussian summary. Replace current K/dK-based selection
with one differentiable source-trained potential; explicitly retain current
native V/g as a separate input for this local key-channel test.

## Data and comparisons

-Frozen source4LOO/new8all4, both N/confirmation queries, all4 GRAIN directions,
 both old readout modes. No current Q, human labels, MEPS, API or new neural calls.
-H16/state/GRAIN/positions/layout: existing `logic_source_state_transport_v1`
 pure inputs. Source original K/dK and query bank: `logic_source_query_selection_v1`.
 Current V/g are exported into separate value-only packets; current K/dK remain
 inaccessible during prediction. Original masks/GQA/RoPE preserved.
-For each source split, derive the exact source-only Qbar used for teacher logits.
 Source training records must use that split's same Qbar, not mix incompatible
 query anchors or use a held author's Q. Never fit on exposed new8.
-Use the existing source-only rank32 causal-context features and their derivatives
 from `source_state_transport.features`; freeze current source reference/scale.
 No rank, alpha, head or landmark search in this card.

## One compatible function, two teaching objectives

Let ell0_qh(H16) be identity H16 passed through the fixed norm/K/RoPE projection
and fixed source Qbar. This is a computable baseline, not current native K.

    ellhat_qh(H16)=ell0_qh(H16)+Phi_source(H16) B_qh.
    dellhat_qhf[D]=dell0_qhf[D]+D Phi_source(H16)[D] B_qh.
    ahat=softmax(ellhat+mask).
    Rhat_qmf=sum_heads Cov_ahat(native_value_m,dellhat_qf).

Fitting directly targets the source teacher's logits and optionally directions,
not its full hidden state. Remove only the per-sequence allowed-token constant
gauge from logit/dlogit fit residuals; center features and their derivatives
consistently. Softmax itself remains unchanged by this gauge operation.
Pin constant-gauge coefficients to zero (or omit those columns); do not leave
an unpenalized all-zero intercept column in the normal equations.

Two fixed objectives: normalized source logit loss only, and the same loss plus
source-unit-normalized GRAIN dlogit loss. Equal authors, equal query materials,
equal tokens within a material; alpha=.01 and fixed source-only scaling.
Five references x2queries x2objectives =20 bounded numeric solves; identity and
original source-Q/native-key readers remain unchanged comparators. Each query
learns12head logits jointly, no post-hoc beneficial-head selection.

## Acceptance / outputs

1. Freeze plan/code/tests before fitting. Verify exact token/mask/source addresses.
2. Fit with source-only logits/directions; independently apply on source-LOO/new8
   from H16/GRAIN/layout and the explicitly retained value-only channel.
3. Freeze complete logits/dlogits/weights/actions before reading current native
   K/dK or action targets. All counts and missing conditions must be reported.
4. Test finite differences of the same scalar path, common-logit-shift invariance,
   source exclusions and prediction target denial. No arbitrary gradient add-back.
5. Report attention KL/centered-logit error, GRAIN dlogit error and query/factor/
   contrast action fidelity and amplitude against both source-Q/native-key and
   old native-Q targets. Tokens are not independent authors. No new8-based tuning.
6. Distinguish selector fidelity from full personality prediction; native V still
   requires neural evidence, so this does not by itself save the whole forward pass.

Max20fits, cumulative120CPU seconds including one replay. Stop after fixed comparisons; retain
negative outcomes and name the next missing functional dependence if needed.
Publish only aggregate report/theory/CSV/hash manifests. Keep private sources,
models and predictions ignored; old scorers/releases/lockboxes unchanged.
