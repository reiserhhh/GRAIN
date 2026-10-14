# HM70: Common Observables and Joint Profile Transport

2026-10-09. New bounded CPU-only mechanism packet. Freeze before acquisition.
Question: what can two declared material banks measure in common, and does
event-level joint structure retain profiles that common means cannot distinguish?
This is not a fidelity, personality-prediction, rank or polynomial-degree search.

## Objects and Original Scope

Reuse complete frozen HM68 tables: 30 ordered behavior pairs, two banks, eight
coded cells (x,y,g) in lexicographic {-1,+1} order, all six observation stages.
Reuse all 240 HM69 paired probability profiles, including all 60 score fibers.
No new model calls, labels, raw texts, API, generation or training. Original
18 protocols/100 cards/44 questions remain open; HM45 stays deferred.

The material banks share designed cell identities, not certified semantic
equivalence of arbitrary language. F0/F1 differ in wording and behavior content.
C=(y,g) includes companion behavior and ownership, not just external situation.
This packet does not discover event identity, estimate causal Bu, or validate
human traits. It asks what is measurable once the declared unit identity is held.

## Fixed Construction

For each pair and stage form Af=[1,Ff], and Wf=col(Af). A common event function
must belong to W0 intersect W1. Remove the constant and obtain a basis Z with
1^T Z=0 and Z^T Z=I. The common projector is the object; basis axes are anonymous.
Use left-nullspace intersection and deterministic ordered projections of Walsh
characters for eight-cell charts. Column equilibration and machine-relative
rank tolerance 64*eps*max(shape) are numerical rules, not searched parameters.
Calibration Cf satisfies Af Cf=Z on the complete declared support.

Keep two author observations separately:

    m_u = P_u^T Z
    H = [1, Z, Z_i Z_j for i<=j]
    h_u = P_u^T H

Products are within the same coded event. E[Zi Zj] is not E[Zi]E[Zj]. Do not
assume independence. Degree is fixed at two regardless of results. If H has
row rank eight, its moment map identifies the complete finite profile:

    P_u^T = h_u pinv(H).

Otherwise retain the observable quotient and lost profile component; do not
raise degree or change the basis to make recovery pass. Full rank on eight
known cells is finite-support identification, not unseen-text generalization.
Products of calibrated event observations cannot be obtained from already
aggregated six-score means alone. This is an additional measurement object,
not new information conjured from an old summary.

For each of the 60 HM69 origin-score fibers report its origin and other-bank
mean-score gaps. Nonzero other-bank gap with zero origin gap is a witness that
no function of the origin six means can exactly transport all these profiles,
including nonlinear functions. Half the other gap is an unavoidable worst-case
Euclidean error on that witness pair. This is standard quotient/nullspace
reasoning, not a newly invented impossibility theorem.

Also form one global LINEAR chart per stage by stacking the 30 pairs as 240 DISTINCT
states (pair_id,x,y,g). One global affine calibration uses the same coefficients
for all pairs. Never identify different behavior pairs merely because x/y/g
match. Global common-space size need not equal local chart sizes. Preserve
failure of a context-free calibration without declaring all nonlinear transport
or psychological universality impossible. The global audit tests a common
affine coefficient law only; do not form a new author population by pooling
distinct behavior-pair supports. Quadratic profile identification is local.

## Complete Outputs and Checks

- All 180 local quadratic charts, six global linear charts, their full bases,
  projectors/calibrations, spectra, H matrices, and column/rank metadata.
- All 1440 local stage/profile pairs: common means, event-product moments,
  recovered or observable-projected probabilities, raw material observations.
- All 60 score fibers, same-event versus product-of-means controls, exact
  origin/other-bank readout witnesses, all stages and failures retained.
- Unit tests for affine gauge invariance, bank exchange, support alignment,
  constant-only intersection, products distinguishing equal means, incomplete
  moment rank, probability validation, and nonfinite/empty input rejection.
- Independent direct scalar sums, dimension via union rank, projection equations,
  product construction, probability reconstruction, tables and input/code seals.

prepare/run/verify CLI refuses overwriting r1. Save input hashes, frozen code,
output hashes, checkpoint and actual timing. Bound collector to 120 CPU wall
seconds with one-thread BLAS. All invariance/reconstruction tolerances are
numerical checks; no psychological thresholds or optimized score claims.

## Prior Art and Reading

Common subspaces, finite function algebras, and moment injectivity are established
mathematics. GRAIN already has kernel-distribution and full-rank finite-observation
work; this packet's candidate contribution is the material-common event/joint
measurement interface and its actual consequences, not rebranding those tools.

[Sriperumbudur/Fukumizu/Lanckriet 2011](https://www.jmlr.org/papers/v12/sriperumbudur11a.html):
actually read definition (2), characteristic definition, and section3.1 equations
8--11. Distinguish injectivity on declared finite support from global universality.
[Diehl's authored lecture](https://diehlj.github.io/stone-weierstrass/stone-weierstrass.html):
read theorem and proof discussion; the displayed proof has notation/union-index
typos, so use only the standard theorem statement, not that text as verified
proof. We give a direct finite rank argument here, not an asymptotic application.

Re-read all155 lines of the pinned OpenAI/math metric/connection introduction:
commit adc7f1241b42e322a6451854ab7e4b4c146bf78a, preprints/Determination-of-a-metric-
and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/
01-introduction.tex. Common observation objects and coordinate/gauge ambiguity
inspire the design. No elliptic/PDE, smooth-manifold, full-boundary-data or
uniqueness theorem is transferred to language. No complete repository-reading
or full-proof claim.

## Scientific Decision

If material-common joint moments recover score-equivalent profiles, retain
event-level relations before author aggregation rather than tune the predictor.
If only constants transport globally, develop condition-indexed observation
charts, not one universal affine score remapping. If quadratic joint moments
fail, record exactly what quotient remains. Next mainline work must determine
how the declared event/condition units are obtained and transported to new
materials; no fidelity, rank, threshold, or degree sweep follows this packet.
