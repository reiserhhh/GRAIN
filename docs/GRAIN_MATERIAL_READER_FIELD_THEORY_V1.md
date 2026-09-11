# Joint material reader: values, movement and functional geometry

2026-09-11. NEXT82 computed; source4 ALL and reused new8. This extends the
[fixed-last material result](GRAIN_READER_MATERIAL_DEPENDENCE_THEORY_V1.md).

## 1. The same-material interface is a function, not a factor-name match

Let H be the cached full hidden state, B its BODY addresses, and G the four
frozen GRAIN factor coordinates on those addresses. The two query states share
the BODY exactly. Define

```text
C = concat(H_N,last,H_confirmation,last)/sqrt(2)
M0 = mean_B H
Mf = mean_B [(G_f-mean_B G_f)(H-M0)], f=1..4
x = block-scaled concat(C,M0,M1,...,M4).
```

GRAIN supplies a material coordinate system; the query reader supplies a
question-dependent function on that material. No one-to-one semantic naming is
required to connect them. In this finite computation G is held fixed under H
interventions, so `DMf = mean_B [(G_f-mean_B G_f)DH]`. If a future system
recomputes G from perturbed text, its `DG` terms must also be differentiated.

The differential reader is `d=r_N-r_confirmation`, with fixed source common
reader m and `r_N/C=m +/- d/2`. NEXT81 showed a BODY dependence at fixed last;
NEXT82 now expresses it through a jointly learned d(x). The four cross-moments
are one finite representation, not an assertion that all material information
is contained in those moments.

## 2. Source value and directional teaching are mathematically compatible

For Gaussian `k(x,z)=exp(-||x-z||^2/2)`, value/derivative observations use

```text
Dx_v k = -(v dot (x-z)) k
Dz_w k = +(w dot (x-z)) k
Dx_v Dz_w k = [v dot w-(v dot (x-z))(w dot (x-z))] k.
```

With eight residual values and twenty derivative targets Y, a source-weighted
28x28 solve gives `dhat=d0+k_eval,obs beta`. Values subtract the original-source
mean d0; derivatives do not. Three observation groups use inverse total target
energy weights. The frozen .001 trace-scaled regularizer is not selected by new8.
Every predicted reader derivative is obtained by differentiating this same
function. Gaussian symmetry/PSD and both first/mixed derivative signs are tested.

The context-only function necessarily gives `D_BODY dhat=0`. BODY mean and GRAIN
moment inputs remove that mathematical restriction and fit the source BODY
derivative target with residual energy ratios .0019963 and .000038664. This is
an explicit construction of material dependence, not a personality-validity rate.

Combining values and derivatives follows the established principle in
[Solak et al. (2002)](https://proceedings.neurips.cc/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html).
The particular material map, source clamp and GRAIN directions are our present
experimental combination. We do not borrow a GP uncertainty interpretation for
the current deterministic regularized solve.

## 3. An endpoint gradient cannot silently become a global potential

The native reader is an endpoint derivative. A learned reader vector alone
does not determine a globally integrable scalar potential or its missing offset.
Use the explicit anchored functional

```text
Fhat(H;A)=rhat(A)^T phat(H)
D_H Fhat(H;A)[D] = rhat(A)^T Dphat(H)[D]
D Fhat(H;H)[D] = rhat(H)^T Dphat(H)[D] + Drhat(H)[D]^T phat(H).
```

The first derivative at H=A has the earlier native local K/V component as its
matched target. The second is a different, well-defined scalar derivative. The
reader-motion correction is not another first-order attribution of the original
native model. Matching finite differences verifies both objects without merging
them. Current H16 and the complete model score L remain separate dependencies.

## 4. A functional geometry, not just Euclidean reader distance

For a fixed evidence vector p and reader error e,

```text
delta F = e^T p
(delta F)^2 = e^T (p p^T) e.
```

For several GRAIN/evidence rulers R, the appropriate squared projected error
is `||R e||^2=e^T R^T R e`. The directions relevant to the output are therefore
determined jointly by reader and material. A source vector norm and an applied
functional norm need not order candidate readers the same way.

The actual new8 result instantiates this distinction: BODY mean has lower vector
error but scalar-projection r=.1812; context field has r=.5818 there and the
highest assembled scalar ordering r=.6840; GRAIN field has the lowest assembled
raw scalar error .179712. Covariance/cancellation with evidence transport can
also influence the assembled result. Do not infer a unique causal explanation
from ranking disagreement alone.

This motivates NEXT83 source-observable teaching: teach which vector directions
matter to the scalar/actions, while still defining one differentiable reader
field. General vector-valued kernel learning has an established mathematical
basis ([Micchelli & Pontil, 2005](https://pubmed.ncbi.nlm.nih.gov/15563752/));
the present matched coefficient experiment and source rulers must be specified
independently rather than claimed as a theorem of that paper.

## 5. Place in the whole theory

The current usable chain is:

```text
same permitted text -> shared H16/BODY and frozen GRAIN coordinates
 -> material summary x -> source-learned reader and its derivatives
 -> anchored local evidence reading / explicitly separate diagonal response
 -> later external criterion and sourced interpretation.
```

This develops the first user-proposed junction (same sentence/group, multiple
views, GRAIN-weighted vector reading). Psychoanalytic/transactional-analysis
descriptions remain interpretation hypotheses with separate evidence and privacy
requirements, not labels inferred from these numerical results. The present
query contrast has no new clinical or questionnaire targets. The result is a
working local mathematical interface with a bounded component gain; full model
replacement and psychological interpretation are not silently identified with it.
