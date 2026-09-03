# GRAIN V8 Response-Set Incidence Geometry Plan

Status:
`PLANTED_AFFINE_V1_CODE_GATE_PASS__INDEPENDENT_METHOD_AUDIT_PARTIAL`

## 1. Motivation

A person is not represented only by a point. Under a shared condition domain
\(\mathcal Z\subset\mathbb R^q\), an author defines a response map

\[
f_u:\mathcal Z\rightarrow\mathcal Y,\qquad
f_u(z)=a_u+B_uz.
\]

The response image

\[
S_u=f_u(\mathcal Z)\subset\mathbb R^p
\]

may be a line segment, surface, or volume. The lifted graph

\[
\Gamma_u=\{(z,f_u(z)):z\in\mathcal Z\}
\subset\mathbb R^{q+p}
\]

retains which condition produced each response.

Kakeya geometry motivates one limited principle: zero ambient area or volume
does not imply absence of direction, dimension, or incidence structure. It
does not imply that a finite noisy author response set is a Kakeya set.

## 2. Three relations

### Same-condition incidence

\[
d_{uv}^{=}
=
\inf_{z\in\mathcal Z}
\|f_u(z)-f_v(z)\|.
\]

This is the primary technical relation. It asks whether two authors produce a
common response under the same identified condition.

### Free image incidence

\[
d_{uv}^{\mathrm{free}}
=
\inf_{z,z'\in\mathcal Z}
\|f_u(z)-f_v(z')\|.
\]

If \(d_{uv}^{\mathrm{free}}\) is small but \(d_{uv}^{=}\) is not, the two
paths cross only under different conditions. This is
`CROSS_CONDITION_ONLY`, not a shared-response edge.

### Uncertainty tube

With estimated pair uncertainty \(\sigma_{uv}\),

\[
D_{uv}^{=}=d_{uv}^{=}/\sigma_{uv},\qquad
D_{uv}^{\mathrm{free}}=d_{uv}^{\mathrm{free}}/\sigma_{uv}.
\]

Calibration defines intersection, refusal, and disjoint regions. Near
incidence inside the grey region must refuse rather than become an exact
intersection claim.

## 3. Geometry diagnostics

- principal angles between \(\operatorname{im}(B_u)\) and
  \(\operatorname{im}(B_v)\);
- transversality rank of \(B_u-B_v\);
- same-condition coverage inside a calibrated uncertainty tube;
- free-set overlap extent;
- cross-half and cross-observer persistence of the pair-specific excess graph;
- affine rank and finite-scale box-count dimension;
- recovered author-incidence graph versus its planted graph.

The dimensional objects are kept distinct:

- intrinsic dimension of one author set;
- dimension of the union across authors;
- ambient Lebesgue area or volume;
- graph connectivity created by intersections.

Direction richness, full finite-scale union dimension, and high graph
connectivity are not equivalent.

## 4. V1 planted worlds

The first executable battery uses affine maps only.

| Family | Positive worlds | Negative or boundary worlds |
| --- | --- | --- |
| 2D lines | same-condition transverse crossing, coincident overlap | cross-condition-only, parallel offset, endpoint near miss |
| 3D lines | same-condition transverse crossing, coincident overlap | skew lines, parallel offset, cross-condition-only |
| 3D surfaces | same-condition line overlap/intersection | parallel planes, cross-condition-only |
| 2D/3D full rank | positive-area/volume overlap | separated bodies |
| Population graph | common-center direction pencil | direction-rich tangent segments, shared-anchor null |
| Attacks | none | half shuffle, observer-specific displacement, condition-label permutation, private condition coordinates |

The primary run uses two halves, two observers, shared condition coordinates,
and disjoint discovery/calibration/confirmation seed blocks.

## 5. Frozen primary gates

- same-condition intersection AUC at least 0.90;
- free-image intersection AUC at least 0.90;
- same-condition versus cross-condition balanced accuracy at least 0.85;
- line/surface/volume dimension median absolute error at most 0.25;
- principal-angle median absolute error at most 10 degrees;
- recovered pair-specific excess-incidence graph edge F1 at least 0.85;
- cross-view excess-edge-set Jaccard at least 0.80;
- decisive false-positive one-sided 95% upper bound at most 0.03 for
  parallel, skew, condition-only, and shared-anchor controls;
- no-overlap/private-condition/observer attacks refuse at least 0.90;
- rigid rotation and translation invariance at least 0.95.

Failure of a primary gate returns `STOP` or a narrower partial-boundary
decision. Confirmation results are not used to change thresholds.

## 6. Claim boundary

A pass would show that GRAIN can technically estimate line/surface/volume
dimension and distinguish same-condition incidence, cross-condition-only
incidence, overlap, near miss, parallel offset, and 3D skew geometry in
registered planted affine worlds.

It would not show interpersonal compatibility, psychological similarity,
personality validity, clinical relationship, or real-text incidence. Those
interpretations remain closed.

## 7. Completion

The audited run is
`results/v8_response_set_incidence/v1_final_audited_20260726/`.

The executable gate passed:

- same-condition and free-image AUC: 1.000 / 1.000;
- relation balanced accuracy: 0.9657;
- median dimension error: 0.0863;
- median principal-angle error: 0.2147 degrees;
- paired map-equivalence graph F1: 1.000;
- cross-view map-equivalence Jaccard: 0.9979;
- false same-edge one-sided 95% upper bound: 0.000535;
- attack refusal: 1.000.

The independent methodological audit is `PARTIAL`. The graph gate currently
identifies near-equivalent affine response maps rather than a general
condition-matched excess-incidence relation. Persistence, transversality,
intersection dimension, nonlinear sets, and real-text incidence remain open.
See `reports/V8_RESPONSE_SET_INCIDENCE_REPORT.md` and
`reports/V8_RESPONSE_SET_INCIDENCE_INDEPENDENT_AUDIT.md`.
