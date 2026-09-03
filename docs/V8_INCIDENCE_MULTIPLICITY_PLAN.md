# GRAIN V8 Incidence Multiplicity and Latent-Group Plan

Status: `V2_REGISTERED_BEFORE_CONFIRMATION`

## 1. Hypothesis

V2 does not test whether a high-multiplicity intersection is automatically a
personality signal. It tests the narrower planted-world proposition:

\[
\boxed{
\text{condition-matched, scale-persistent, cross-view-stable incidence}
\Longrightarrow
\text{recoverable latent co-membership information}
}
\]

The latent groups are anonymous simulation labels. They are not personality
types or psychological constructs.

## 2. Hyperedge object

For author \(u\), condition \(z\), define the fitted scalar uncertainty
\(\widehat\sigma_u(z)\) from cross-view dispersion and residual scale, and
the frozen population scale

\[
\widehat r(z)=\operatorname{median}_u\widehat\sigma_u(z).
\]

The implemented isotropic uncertainty ball is

\[
T_u(z,\epsilon)
=
\left\{
y:
\|y-\widehat f_u(z)\|_2
\le\epsilon\widehat r(z)
\right\}.
\]

This is not a fitted author-specific Mahalanobis tube. Full covariance tubes
remain an untested extension.

An author set \(H\) forms a same-condition hyperedge only when

\[
\bigcap_{u\in H}T_u(z,\epsilon)\ne\varnothing.
\]

Pairwise graph connectivity is insufficient because chained overlaps need
not have a common intersection. The implementation therefore verifies a
common enclosing-ball feasibility condition for every proposed cluster.

For a verified hyperedge, V2 records:

- multiplicity \(|H|\);
- condition support;
- persistence across the frozen epsilon grid;
- tangent compatibility;
- localization in the condition domain;
- half/observer co-membership stability.

## 3. Pair-specific persistence

Raw multiplicity is downweighted when the same component contains most of
the population. For a verified component \(H(z,\epsilon)\), define

\[
s(H)=\frac{n-|H|}{n-2}.
\]

The pair persistence score is

\[
P_{uv}
=
\mathbb E_{z,\epsilon}
\left[
\mathbf 1\{u,v\in H(z,\epsilon)\}\,s(H)
\right].
\]

This makes a population-wide common anchor contribute zero pair-specific
weight. After aggregation across the frozen condition-radius grid, scores
are centered and scaled across all author pairs in the population. A
separate tangent-weighted score is recorded diagnostically; it does not enter
the frozen group decision.

A discrete group claim additionally requires at least two verified
hyperedges of multiplicity at least three whose union covers at least 75% of
the population. Isolated local neighborhoods in a continuous population are
retained as local geometry and do not become a type claim.

## 4. Planted worlds

| Class | Worlds |
| --- | --- |
| Clear group | affine regional convergence in 2D/3D, nonlinear tangent convergence, common anchor plus true groups |
| Boundary | group-specific isolated transverse intersections |
| Null | global common anchor, random crossings, pair-only map equivalence, continuous population, 3D projection-only crossings |
| Attack | observer-only group, half-only group, condition permutation, inadequate condition support |
| Fresh challenge | held-out RBF groups, heteroskedastic 3D groups |

Every repetition generates fresh authors and fresh geometric parameters.
Discovery, calibration, confirmation, challenge, and sensitivity use
disjoint seed blocks.

## 5. Frozen gates

Clear latent-group worlds:

- co-membership AUC at least 0.85;
- co-membership F1 at least 0.80;
- adjusted Rand index at least 0.70;
- multiplicity median absolute error at most 1;
- condition localization error at most 0.10;
- cross-view pair-score Spearman at least 0.80.

Controls:

- false group-claim one-sided 95% upper bound at most 0.03 in global-anchor,
  random-crossing, pair-only, continuous, and projection controls;
- private/incorrect/half/observer/support attacks refuse at least 0.90;
- fresh nonlinear challenge co-membership AUC at least 0.75;
- fresh nonlinear challenge non-refusal rate at least 0.80;
- equivalent condition reparameterization consistency at least 0.95.

Any critical null failure returns `STOP`. Confirmation metrics do not change
thresholds.

## 6. V2 result and registered V2.1 correction

The first full V2 confirmation returned
`V8_INCIDENCE_MULTIPLICITY_PARTIAL_BOUNDARY`. Pair-score AUC was 1.0 in
every clear world and all null/attack controls passed, but the tangent
quadratic world had median F1 and ARI of 0.0. The cause was identifiable:
V2 calibrated its hyperedge threshold against the largest individual local
hyperedge in each null population. Random local triplets around a
population-wide anchor therefore raised the threshold above the persistence
of the planted tangent groups.

This is an estimand mismatch. A local intersection is not the same object as
a discrete population partition, which already requires at least two
components of multiplicity at least three covering at least 75% of authors.
V2.1 therefore freezes a new seed and selects the smallest threshold that,
on calibration data only:

- controls the population-level group-claim one-sided 95% upper bound at
  0.03;
- gives every clear calibration world median F1 at least 0.80 and median ARI
  at least 0.70;
- gives every clear calibration world a group-claim rate at least 0.80.

The confirmation seeds remain untouched. Smoke tests may use zero observed
null claims as a small-sample preflight proxy, but only the full run can pass
the registered binomial upper-bound gate. This correction was registered
before the V2.1 confirmation run and does not alter the psychological claim
boundary.

V2.1 recovered all four clear worlds at median F1/ARI 1.0, including the
tangent world, but its nominal PASS was rejected during acceptance review.
The global-common-anchor control made group claims in 8/160 populations
(5.0%, one-sided upper bound 0.088), while pooling it with four easier null
worlds produced an apparently acceptable upper bound of 0.018. Pooling null
classes therefore hid a world-specific failure.

V2.2 freezes another seed and changes only this aggregation rule:
calibration and confirmation must control the one-sided false-claim upper
bound separately in every registered null world, and the reported null gate
is the maximum world-specific upper bound. This correction was registered
before V2.2 confirmation.

V2.2 exposed a design-power error before it could support a substantive
conclusion. With 80 calibration repetitions per null world, even zero false
claims have a one-sided 95% upper bound of 0.0368, so the registered 0.03
gate was mathematically unreachable. The calibration data nevertheless
contained 31 thresholds with zero null claims and passing clear-world
recovery. V2.3 therefore changes no estimator rule: it raises calibration
repetitions to 120, freezes a fresh seed, and adds a pre-run power check that
rejects any full configuration whose best possible zero-event upper bound
cannot meet its own gate.

## 7. V2.3 confirmation result

The untouched V2.3 confirmation panel returned
`V8_INCIDENCE_MULTIPLICITY_PLANTED_PASS`.

- all four clear worlds had median co-membership AUC, F1, and ARI of 1.0;
- the minimum clear-world group-claim rate was 0.95;
- recovered multiplicity error was 0;
- all five null worlds separately had 0/160 group claims, with worst
  one-sided 95% upper bound 0.0185;
- isolated transverse intersections had 0/160 group claims;
- attack refusal was 1.0;
- both fresh nonlinear challenges had median AUC 1.0 and non-refusal 1.0;
- the global common anchor had raw multiplicity 24 but group-claim rate 0.

The canonical output is
`results/v8_incidence_multiplicity/v23_final_20260726/`. Noise sensitivity
remains a boundary: the common-anchor-plus-groups world refused at noise
0.06 and both clear worlds in the sensitivity panel refused at 0.12. See
`reports/V8_INCIDENCE_MULTIPLICITY_REPORT.md`.

Independent methodology audit is `PARTIAL`: the implemented gates reproduce,
but a fitted whole-map distance baseline also reaches AUC 1.0 in every clear
and challenge world. Complete hyperedge enumeration, cross-condition chain
attacks, and incremental incidence value remain open. See
`reports/V8_INCIDENCE_MULTIPLICITY_INDEPENDENT_AUDIT.md`.

## 8. Claim boundary

The pass shows only that persistent same-condition incidence contains
recoverable anonymous latent co-membership in the registered simulated
worlds, beyond common anchors, chance crossings, and pair-only map
equivalence.

It would not show that intersections are personality similarity,
compatibility, clinical relation, social connection, or a real-text
construct. Kakeya geometry remains a direction-and-incidence motivation,
not an empirical or psychological conclusion.
