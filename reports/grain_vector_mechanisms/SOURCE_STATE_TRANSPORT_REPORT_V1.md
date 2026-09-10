# Source state transport: position similarity does not preserve response

2026-09-10, NEXT52 complete. Primary code frozen at `f326e39` before construction.
Same12 author materials, two queries, four GRAIN directions. No new neural calls,
human labels, external APIs or decoder training. Source4/new8 denote the existing
development cohorts, not a fresh confirmatory sample.

## 1. What was found

Source-only small maps reduced preAttention22 **state error by42–47%**, while
making the corresponding directional derivative error **39–50% larger**.
The discrepancy is observed on the same materials and the same map: derivatives
are analytically computed from that map, not fitted by an incompatible second head.

Simply feeding H16 to the retained selection program preserves substantial new8
component association (N r=.878, confirmation r=.893), but over-amplifies responses
by about4x. Thus there is useful early information, but copying positions is not
enough to reproduce how the native intervening computation transforms it.

The original three-arm replacement gate is not passed. Keep NEXT51's native-input
component and close this **primal-only** candidate family. Develop response-aware
transport next; do not erase the observed early association or tune on new8 outcomes.

## 2. Actual program and data separation

    same text -> frozen H16 and token-aligned GRAIN directions D
              -> source-only Psi(H16), analytic D Psi(H16)[D]
              -> actual attention22 RMSNorm/differential, K/V/dK, RoPE/mask
              -> fixed same-query Qbar selection -> fixed source rbar -> kappa_hat

Identity, pointwise residual and causal-cumulative residual maps were compared.
Rank32 weighted input PCA, alpha=.01, seed52; equal authors, queries within author,
tokens within material. Eight source materials/9040 token rows are **four people**.
Four leave-one-source-out references and all4 for every new8 author give ten Ridge
fits. The fit sees source **primal states only**, not native tangents or endpoint
actions. Current prediction reads H16/GRAIN/layout/fixed assets, not current pre22
states, tangents, queries, gradients or outcomes. Hash verification checks lineage
bytes; numerical target access is denied inside isolated fitting/prediction stages.

All24 candidate packets were frozen before evaluation. Complete candidate states
and tangents are stored losslessly as X16+ZB and D+DZB with exact full-array hashes;
they are not rounded or reconstructed from opened targets. Native input is an
explicitly ineligible reference. The target is NEXT51's **one K-channel contribution
to a fixed final logit derivative**, not a whole-model score or questionnaire r.
H16 still requires the original prefix; this is a blocks17–21 replacement test.

## 3. Endpoint results, new8

| Arm | N r | N SSE/zero | Confirmation r | Confirmation SSE/zero | Query-contrast r | Contrast SSE/zero |
|---|---:|---:|---:|---:|---:|---:|
| Identity | .877807 | 11.643160 | .893467 | 8.704698 | .826269 | 12.187527 |
| Pointwise | .662383 | 12.619267 | .828645 | 12.134430 | .500384 | 10.751751 |
| Causal context | .688443 | 12.204460 | .829507 | 13.236393 | .535973 | 10.134311 |
| Native-input reference, not eligible | .972986 | .156129 | .978394 | .071926 | .963235 | .208588 |

Native N/confirmation RMS=.223946/.093901. Identity=.952682/.357178;
pointwise=.924318/.400012; causal=.918831/.415040. Correlation alone misses this
large amplitude mismatch. Causal context reduces contrast SSE relative to identity,
but this is not the frozen per-query gate (SSE/zero<1 separately in both queries).

Source4 leave-one-out N/confirmation r: identity .306/.298, pointwise .278/.329,
causal .207/.224, reference .968/.875. Pointwise source SSE falls4.716->2.257 and
4.414->1.748, but does not pass either gate or transfer that advantage to new8.
All four factors are retained: new8 identity contrast r=.744/.802/.833/.973;
pointwise=.395/.375/.543/.690; causal=.435/.424/.544/.738. No factor was selected
for the headline, gain calibration or a new fitted rescue.

## 4. State versus directional response

Equal-author/material averages; ratios are not token-level significance estimates.
State relative MSE includes the large preserved residual-stream state, not R-squared.

| New8 arm | State error / identity, N / C | Tangent error / identity, N / C | Tangent cosine, N / C |
|---|---|---|---|
| Identity | 1 / 1 | 1 / 1 | .615 / .615 |
| Pointwise | .531 / .530 | 1.389 / 1.389 | .560 / .560 |
| Causal context | .578 / .577 | 1.499 / 1.499 | .543 / .543 |

Post-result cached geometry, with no refitting, adds two clues:

- The source PCA32 basis captures31.94% of new8 GRAIN direction energy (31.13%
  source4). A direction orthogonal to that basis has exactly identity transport
  in both residual-map families. This is a mathematical blind spot, not proof that
  the remaining68.06% necessarily needs a native update.
- Native full-tangent norm is about1.117 times input norm, whereas candidate/native
  norm is1.095 pointwise and1.129 causal. Endpoint amplification is therefore not
  simply a4x error in full-vector length: orientation and attention/readout matter.
- Final-query tangent is zero for identity/pointwise. Causal mean permits BODY to
  reach the query, but gives mean norm11.593 versus native6.974/7.587. Adding a
  communication channel is not the same as learning its right response.

## 5. Research decision and theory connection

GRAIN can specify the perturbation distribution a replacement should preserve:
same token material supplies both its group-referenced configuration and its native
query-response target. The next experiment changes **what is taught**, not hidden
rank/alpha selected from current results. Keep the same feature maps and train one
coherent function on source values plus the four source directional responses.

Intermediate representation hints motivate the primal arm ([FitNets](https://arxiv.org/abs/1412.6550)).
Value-plus-derivative fitting has precedent in [Sobolev Training](https://arxiv.org/abs/1706.04859)
and [Jacobian Matching](https://proceedings.mlr.press/v80/srinivas18a.html).
Our NEXT52 result is a local example of their relevant distinction, not a novel
theorem that value matching always fails. [Theory](../../docs/GRAIN_SOURCE_STATE_TRANSPORT_THEORY_V1.md)
derives the blind spot and the GRAIN-weighted response loss; [NEXT53](../../docs/GRAIN_RESPONSE_AWARE_STATE_PLAN_V1.md)
freezes the next finite comparison. Auxiliary psychodynamic/transactional coding
continues through the existing evidence-linked interpretation interface; no
diagnostic construct is assigned to these mathematical derivatives.

## 6. Reproducibility and resource accounting

Preparation .920284s, ten fits .588025s, prediction5.676324s: total7.184633s within
180s. One cached prediction replay5.635678s; no further fitting/model jobs.
24 NPZ packets and9 primary CSVs reproduce exactly. Source Qbar/rbar match NEXT50/51
exactly; original relative-RMS BODY units match exactly; independently calculated
metric max error6.218e-15. Four denied-path probes in fitting and prediction each,
no unexpected opens. Primary code/plan unchanged after freeze.

Private packet: `results/grain_vector_mechanisms/logic_source_state_transport_v1/`.
Public seven aggregate CSVs plus audit: [tables](source_state_transport_tables/).
Implementation: `grain_research/vector_mechanisms/source_state_transport.py`;
runner/audit: `scripts/run_grain_source_state_transport.py` and
`scripts/audit_grain_source_state_transport.py`. Stage order for a new local packet:
prepare -> fit -> predict -> optional cached replay -> evaluate -> replay evaluation
-> audit. Existing packet is complete; do not overwrite or restart it.

44 focused tests pass (18 existing torch/Python deprecation warnings); Ruff passes.
Tests cover same-function finite differences, source exclusions/weighting, causal
boundaries, fixed rank failure, original direction units, and post-result nullspace
and value/derivative witnesses. No release/scorer/lockbox/MEPS or remote push changes.
