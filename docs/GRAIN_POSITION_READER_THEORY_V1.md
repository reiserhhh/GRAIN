# Shared directions, local coefficients, and the measurement metric

2026-09-10. NEXT31 / twenty-second logic pass. The empirical object is a local
derivative of a fixed model output on the same material measured by GRAIN. It is
neither a unique attention circuit nor a psychological factor definition.

## 1. Same-material interface

For an author material u, query q and fixed token layout, let H_u be the selected
body hidden states and J_uq=dF_q/dH_u be its T_u by d scalar-output gradient.
The four historical GRAIN fields g_uk(t) define directions

    D_uk(t,:) = s_uk [g_uk(t)-mean(g_uk)] [H_u(t,:)-mean_t(H_u)],
    tau_uqk = <J_uq,D_uk>_F.

s_uk is the existing RMS-normalizing scale, not a fitted personality weight.
This retains the previous mean-plus-position-specific identity. Here we develop
the position-specific part rather than drop it. A derivative describes the
infinitesimal intervention in this specified hidden coordinate. Finite changes
would require integration or endpoint measurements, with other state held as
declared. No causal interpretation of a naturally chosen topic follows solely
from this differential.

## 2. Hidden vocabulary and local reading

With r orthonormal hidden rows V_q, define the density K_uq=T_u J_uq and write

    a_uq(t) = K_uq(t,:) V_q^T,
    Jhat_uq(t,:) = a_uq(t) V_q / T_u,
    tauhat_uqk = (1/T_u) sum_t <a_uq(t), D_uk(t,:) V_q^T>.

Thus a shared direction vocabulary does not imply identical token weights or
identical people. In the oracle diagnostic a uses the held J. A new predictor
must estimate a from available content/context without that held gradient.
Author differences can appear in H, GRAIN, a, or their signed alignment. The
factorization alone does not identify a personality-specific component.

The source basis minimizes an approximate energy objective with equal author
mass, not equal gradient norm per person:

    (1/U) sum_u (1/T_u) ||K_uq - K_uq V_q^T V_q||_F^2.

Stacking sqrt(T_u/U) J_uq gives this objective. We use randomized rank8 SVD,
oversamples10 and five power iterations, fixed seeds. It is numerical low-rank
approximation, not psychometric common-factor analysis. See
[Halko, Martinsson & Tropp, 2011](https://arxiv.org/abs/0909.4061).

For any invertible change of hidden coordinates, derivatives and directions
must transform dually for tau to remain invariant. Euclidean SVD and its energy
ranking need not be invariant under a nonorthogonal change. This experiment
keeps the model's hidden coordinates and metric fixed.

## 3. Why keeping more energy is not the same as reading the effect

Let E=Jhat-J and collect the declared GRAIN directions D_k. The relevant
measurement error is the semimetric

    ||E||_D^2 = sum_k <E,D_k>_F^2
               = vec(E)^T [sum_k vec(D_k)vec(D_k)^T] vec(E).

Its matrix has rank at most four for this material. The nullspace is invisible
to these four measurements, not psychologically irrelevant to every task.
The usual bound still holds:

    |<E,D_k>| <= ||E||_F ||D_k||_F.

For nested projections the Frobenius error decreases; individual contractions
can improve or worsen because omitted and added terms can cancel. With
e_old=tauhat_old-tau and delta=tauhat_new-tauhat_old,

    ||e_new||^2 = ||e_old||^2 + ||delta||^2 + 2<e_old,delta>.

An exact three-dimensional example: J=diag(3,2,1), D=diag(1/3,-1/2,1).
The full action is1, rank1 action1, rank2 action0. The better rank2 field is a
worse reading of this D. This is ordinary inner-product algebra, not a novel
theorem or a claim that rank1 should always be selected.

## 4. What the actual data add

Four old authors, three queries,48 old tangent actions. Shared-source rank8
with oracle coefficients gives action r=.862/.987/.977 (T/F,N,confirmation),
but relative squared errors .314/.319/.100. Source-only cubic-position readers
give .498/.047/.033 with errors .922/.998/1.001. Position alone is insufficient
in this bounded family; a content-conditioned coefficient map is the next
specific missing mechanism. Pooled correlations combine four GRAIN directions
and four authors; the per-factor four-author tables are retained, not treated
as independent population confirmation.

Four of18 nested-rank additions worsen action despite improving field energy.
N shared4->8: gradient error .840->.738, action error .073->.319. Keeping all
fixed ranks exposes this signed geometry without selecting a held-effect winner.
No human scale labels or new model inference were used. See the complete
[report and tables](../reports/grain_vector_mechanisms/POSITION_SPECIFIC_READER_REPORT_V1.md).

## 5. Development decision

Do not replace GRAIN or rename singular vectors as personality constructs.
Develop a_q=f_q(H,position,GRAIN) using only source authors' gradients as targets,
then check both field and measured-action errors on a held author. Holding V
fixed separates a missing direction vocabulary from a missing coefficient map.
The two fixed arms in [NEXT34](GRAIN_CONTENT_COEFFICIENT_READER_PLAN_V1.md) ask
whether GRAIN-conditioned interactions help this map beyond content alone.
Only after a useful coefficient program exists should we transport its input
from block16 to an earlier encoder or connect a new human criterion.
