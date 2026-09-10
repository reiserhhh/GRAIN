# NEXT29: spread, alignment and signed same-material coupling

2026-09-10. Continuation of the ACTIVE whole-system goal. Previous turn is
progress: NEXT27-28 produced reproducible numerical evidence and a new interface.
This card develops that object, not an accuracy search or clinical branch.
NEXT24-26 remain pending and are not replaced.

## Fixed source and costs

Use only `logic_material_coupling_v1_r2`, its frozen45 OOF coefficient maps,
the original cached E5 comments and four-slice GRAIN scores. Same276 source
authors,3 views,9 queries and4 factors. No fits, no neural/API calls, no new
participants or MEPS/lockbox/release changes. Bind input, code and plan hashes.
Replay old S and G/psi variances on every row before any new criterion analysis.

## Numerical definitions, before new results

For each author/view/query/factor, let sg and sp be within-comment population
standard deviations and S=Cov(g,psi). Define A=sg*sp, rho=S/A and S=A*rho.
At zero spread, S=0, rho is recorded as0 by a marked accounting convention;
the frequency of these cases must be reported. This is not an observed correlation.

One fixed comparison rotates centered psi by+1 comment, keeping its marginal
values and both spreads unchanged: shifted_S, shifted_rho, matched_excess=S-shifted_S.
Do not average all nonzero rotations as an independent null: their average is
algebraically -S/(n-1). Verify this with a synthetic witness, not a new real-data
permutation grid. One fixed rotation is a paired diagnostic, not a calibrated test
of an unrestricted conditional independence hypothesis.

For each query/factor use label-free full-source means a0=mean(A), r0=mean(rho).
They are descriptive references, not fitted deployment norms. At each view:

    S = a0*r0 + r0*(A-a0) + a0*(rho-r0) + (A-a0)*(rho-r0).

Aggregate each term over the3 views for each author. Keep the interaction term;
mean(A*rho) is not generally mean(A)*mean(rho). The exact source-criterion
correlation is the sum of the three variable terms' covariances with Y divided
by sd(S)*sd(Y). Components are signed reference-dependent accounting, not
independent variance shares or a causal mediation decomposition.

## Predefined output families

- Eight source-criterion associations: sg,sp,A,rho,S,shifted_S,shifted_rho,
  matched_excess across all36 query/factor combinations (288 links, BH family).
- Paired r contrasts S-minus-shifted_S and rho-minus-shifted_rho (72), with
  no direction reorientation or ranking as an automatic prediction gain.
- Three normalized covariance contributions, their bootstrap intervals, and
  reconstruction errors for every combination (108 components).
- Geometry summary: mean spreads, alignment, shifted alignment, effective
  comment coverage, count of degenerate cases, source reference means.
- Whole-author bootstrap999, seed20260910; maps and descriptive reference fixed
  for conditional uncertainty. Unknown/constant correlations stay undefined.
  Preserve all tables, not just the previously noted Openness/F03 row.

Freeze numerical rows before joining already-exposed source Y. These data are
exploratory reused evidence, not fresh confirmation. Do not predict new Y or train
another pooling head. Reference normalization is not a new release score.

## Decision and deliverables

Choose the next mechanism target from observed spread versus alignment geometry:
amplitude conditioning, same-material signed coupling, or a joint operator.
Write a source-backed theory note connecting multilevel within/between-person
distinctions and cross-covariance operators. A kernel generalization may be
derived, but do not add a bandwidth or feature search in this card.

New helper/runner/tests, frozen private rows, public aggregate report and one
reconstruction replay. Keep original local-coupling artifacts unchanged. Local
commit, parent handoff sync, goal ACTIVE. Stop this card after it changes the
next mechanism decision; no attempt to exhaust all covariates.
