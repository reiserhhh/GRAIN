# NEXT44: query specificity is mainly transported into a shared local reader

2026-09-10. **Finite experiment complete; whole-system goal ACTIVE.**

## Finding / next decision

The two query-specific block25 compiled readers can be exchanged without losing
their useful local reading. In new8, reciprocal action r remains .990935/.994290
(N/confirmation recipients). Under the exact symmetric split,91.4% of the own-mode
compiled contrast's signed projection comes from the incoming tangent difference,
not the reader difference. Keep a shared local reading interface and next develop
the upstream query-conditioned transport. Do not keep polishing these near-one r's.

These are local responses of a fixed model to GRAIN directions on previously
explored materials, not new personality prediction accuracy. No human labels,
fitting, new model execution or new participants were used.

## Fixed reciprocal experiment

Same4 source and8 source-disjoint reader authors, two queries, four GRAIN BODY
directions. All12 pairs have exactly equal BODY states, token positions and GRAIN
values at H16. Old4 use other3 source authors; new8 use all4. At the current
pre-MLP25 position only pure d and source w are supplied to the numeric predictor.

    prediction(recipient q, donor p) = w_p^T d_q

The native target remains the recipient's saved MLP25 increment read with its
original frozen t. Changing the donor does not change that target.

New8 own-t primary comparison; each row is8 people x4 directions:

| Recipient | Reader | Action r | SSE/zero | Predicted RMS | Native RMS |
|---|---|---:|---:|---:|---:|
| N | Own N | .996936 | .011988 | .332905 | .308761 |
| N | Borrow confirmation | .990935 | .016022 | .308569 | .308761 |
| Confirmation | Own confirmation | .998451 | .002932 | .221981 | .225690 |
| Confirmation | Borrow N | .994290 | .027966 | .254881 | .225690 |

Borrowing retains signs31/32 for N and32/32 for confirmation; factor-centered
r=.990276/.995228. The confirmation borrowing error increases from.003 to.028,
so query-specific amplitude is not perfectly exchangeable even though the
shared mechanism is useful. No rescaling or best-per-query choice was made.

Pooled-t borrowing retains r=.990741/.996167 and SSE/zero=.048785/.007518.
Old4 own-t borrowing r=.984738/.986519; all cohorts, directions and both modes
remain in the public aggregate tables. Repeated source reader assets for new8
are not eight independent estimates of reader geometry.

## Where the two queries differ

For each author/factor, define the complete compiled contrast C=tau_N-tau_C:

    R = (w_N-w_C)^T(d_N+d_C)/2
    T = (w_N+w_C)^T(d_N-d_C)/2
    C = R + T.

This is a declared symmetric bilinear identity, not unique causal mediation.
The native comparison is the difference of the two original recipient targets.

| New8 mode | Contrast vs native r | SSE/zero | R RMS | T RMS | R projected share | T projected share |
|---|---:|---:|---:|---:|---:|---:|
| Own t | .982463 | .055774 | .035711 | .162922 | .085728 | .914272 |
| Pooled t | .976153 | .067709 | .024992 | .163415 | -.048273 | 1.048273 |

Own-mode native contrast RMS=.155559 versus compiled.174634. T alone relates
to native contrast at r=.977335; R alone at.298830. R/T covariance=.000975,
with positive normalized cross-energy.087823. In pooled mode covariance is
-.001704 and cross-energy-.149067: the reader component partly cancels transport.
Negative shares and shares above one are valid signed accounting, not errors
or percentages of personality information. R/T have opposite signs on12/32
own and15/32 pooled observations.

Source4 own projected shares are.224/.776: the direction is consistent but the
precise91.4% figure is sample/interface-specific. New8 own-reader cosine=.972757,
while mean query-tangent cosines by factor are.769-.781. Angle and norm
differences alone do not identify complete Jacobians or human trait relations.

## Development: one shared reader plus a retained correction

After the primary comparison, derive the untrained equal average w*=(w_N+w_C)/2
and correction e=(w_N-w_C)/2. Original readers recover exactly as w*+e and w*-e.
The shared reader produces new8 own-mode r=.995074/.997422 and SSE/zero
.010366/.007822. This is a transparent post-outcome derivation from the existing
matrix, not an extra blind experiment or an optimized blend.

Private shared/correction vectors and192 derived scalar rows are exported.
Their dot products equal the corresponding two-donor average within7.772e-16.
The current upstream direction is still native. What becomes simpler is a
specific downstream reading operation, not all raw-text understanding.

## Execution / verification

- Primary implementation and evaluation code committed as `d4ede47` before
  packet preparation/prediction; plan already fixed in the preceding pass.
-384 scalar predictions on12 paired units,0.006599s numeric stage;0 neural calls,
  forwards/backwards/target scoring/fits/human labels/API calls.
- Four target-path denial probes pass with0 extra denials. The permitted inputs
  contain only w and d; no current x, checkpoint weights or native target arrays.
- Own diagonals replay NEXT43 within1.111e-15; contrast algebra error1.444e-15.
- Eight analysis CSVs replay byte-for-byte;24 source exclusions and12 material
  alignments checked. Independent dot-product replay1.777e-15; component replay
 9.993e-16; metric replay4.441e-16; covariance replay1.011e-16.
-83 focused tests pass,20 existing PyTorch/SWIG deprecation warnings; no full-suite
  claim. Ruff and whitespace checks pass. Unrelated work and old releases remain.

## Outputs / next link

Private root: `results/grain_vector_mechanisms/logic_compiled_query_transfer_v1/`.
It includes pure numerical inputs, frozen predictions, analysis/replay, alignment,
target hashes, post-outcome shared-reader conversion and `replay_audit.json`.
Do not publish author-level rows or vector assets.

Public aggregate tables: [primary comparison](compiled_query_transfer_tables_v1/comparison.csv),
[all factors](compiled_query_transfer_tables_v1/factor_actions.csv),
[contrast accounting](compiled_query_transfer_tables_v1/contrast_summary.csv),
[contrast factors](compiled_query_transfer_tables_v1/contrast_factors.csv),
[geometry](compiled_query_transfer_tables_v1/geometry_summary.csv),
[post-outcome shared reader](compiled_query_transfer_tables_v1/shared_reader_comparison.csv).

[Theory](../../docs/GRAIN_COMPILED_QUERY_TRANSFER_THEORY_V1.md) connects the product
identity to explicit path allocation, referencing
[Sundararajan et al.2017](https://proceedings.mlr.press/v70/sundararajan17a.html).
[NEXT45](../../docs/GRAIN_QUERY_TRANSPORT_FORMATION_PLAN_V1.md), planned only,
will trace the query difference through already saved upstream updates using
the same shared reader. It asks where the difference forms, not how to tune r.

```bash
PY="../project persona/.venv/bin/python"
# prepare requires a new directory; do not overwrite the completed packet.
"$PY" scripts/run_grain_compiled_query_transfer.py prepare
"$PY" scripts/run_grain_compiled_query_transfer.py predict
"$PY" scripts/run_grain_compiled_query_transfer.py evaluate
"$PY" scripts/run_grain_compiled_query_transfer.py evaluate --analysis-name analysis_replay
"$PY" scripts/audit_grain_compiled_query_transfer.py
"$PY" -m pytest -q tests/test_grain_compiled_query_transfer.py
```
