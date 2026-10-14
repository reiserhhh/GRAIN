# Conditional anonymous coordinates: shared authors, not shared class names

2026-10-11, HM143. This follows HM142's unknown local observation-law
constructor. The new system interface is a source-constructed relationship
between conditional anonymous configurations. The group, inverse and marginal
polytope mathematics are established; no new personality or group theorem is
claimed.

## Why this is a mechanism question

The earlier F04/M04/M08/M11 mechanisms already infer author levels, conditional
responses and state in declared common observation units. C3/C4 supply anonymous
continuous coordinates and calibrated transport. Repeating those mixed-effects
fits would not repair the distinct issue here: separate local mixture laws
identify component addresses only up to a permutation. Accurate local scoring
does not say which addresses belong together across conditions.

HM143 constructs this missing interface without fitting a reader answer:

    qualified vector/event in condition c
    -> observed local frequencies -> unknown local likelihood M_c
    -> same-author source pair frequencies P_cd
    -> anonymous coupling J_cd
    -> optional common coordinate correspondence, if supported
    -> individual CONDITIONAL profiles plus relations, not a forced average.

Author/condition/event identities are supplied by the finite design. The
constructor does not infer them from text. The nine observation vectors are
existing frozen GRAIN15 action contrasts. The simulator emits these exact
finite alphabet vectors; nearest decoding is not natural-text segmentation.
Three components are imposed, not discovered or psychologically named.

## Observable coupling and its assumptions

Let M_c have conditional category probabilities in columns. If the observed
event channels at conditions c,d are independent given their local latent
states, their paired law is

    P_cd = M_c J_cd M_d^T,
    J_cd = M_c^+ P_cd (M_d^+)^T.

Full column rank identifies J in the local anonymous frames. The construction
uses only source-learned M and observed paired P, never source hidden labels,
application outcomes, native reader answers or Big5/MBTI targets. If channels
have rank loss, the inverse need not identify a unique coupling. Common event
errors, latent state within a slice or changing emission laws can invalidate
the factorization; supplied event IDs do not eliminate them.

All local prevalences are1/3 in this experiment. Thus matching class sizes
cannot supply the correspondence. Source calibration, each paired bridge
cohort and application cohort are disjoint. Within each bridge, the two
observations belong to the same simulated author. The three paired cohorts
sample the same DECLARED population for the compatible worlds.

With square invertible M, any P is reconstructed by the raw inverse. Therefore
its tiny reconstruction error is NOT evidence that the inferred coupling is
psychologically correct. Sampling inverses can have negative entries; they
are retained as signed estimates, not silently clipped into probability laws.
The exact population compatibility tests below are not finite-sample tests of
those signed estimates.

## Label gauge, bijective correspondence and invariants

Let Q_c be a permutation matrix mapping old addresses to new addresses:

    M'_c = M_c Q_c^T,
    J'_cd = Q_c J_cd Q_d^T,
    p'_uc = Q_c p_uc.

The observed law P_cd is unchanged. If R_dc maps class c to class d, its
coordinate form changes as R'_dc=Q_d R_dc Q_c^T. The triangle cycle

    H_c = R_cd R_db R_bc

changes by conjugation Q_c H_c Q_c^T. Its number of fixed points and cycle type
do not depend on arbitrary class names. This is a finite correspondence cycle,
not a temporal psychological trajectory, quantum measurement effect or curl
of the earlier scalar response function.

For three k=3 charts, fix S_0=I and enumerate the36 exact reference-fixed frames:

    maximize sum_(c,d) <J_cd, (S_d S_c^T)^T>.

This is the finite weighted permutation synchronization objective. Return all
optimal frames. A synchronized OUTPUT is necessarily flat; it does not prove
that the observed preferred pairwise relations were flat. The difference
between unconstrained edge maxima and the best consistent objective measures
the correspondence mass sacrificed by that restriction, summed over edges.
It is not a proportion of personality information removed.

The implemented profile operation is p_uc -> S_c^T p_uc. It readdresses each
condition separately. It never averages different conditions or asserts their
profiles must be identical. All216 independent local relabelings were checked
in each of the four exact bridge worlds; objective error <=4.44e-16. The unique
shared-class frame's conditional profile covariance error was0; for ambiguous
worlds the complete optimal frame SET was checked, not an arbitrarily chosen
profile. Their initialized zero profile-error fields are not separate profile
covariance tests.

## Three different meanings of an apparently failed common frame

All four worlds have EXACTLY the same local three-event observation laws.
Local accuracy, covariance or class prevalence therefore cannot distinguish
them. Joint paired evidence can.

| Population | Pair match mass | Relative frame | Preferred cycle | Common joint law |
| --- | ---: | --- | --- | --- |
| Shared author class |1 on each edge|1 unique frame|identity|exists|
| Independent local classes |1/3 on each edge|36 tied frames|undefined; edges tie|exists|
| Soft conditional response |2/3 on each edge|3 best compromise frames|moves all3 classes|exists|
| Incompatible pair cohorts |1 on each edge|3 best compromise frames|moves all3 classes|does not exist|

The soft response law has nine configurations, each with mass1/9. For a in
{0,1,2}, its supported triples are

    (a,a,a), (a+1,a,a), (a+1,a+1,a), modulo3.

Its modal AB and BC matches are identity; the CA match advances one class.
Every author fails exactly one of these preferred relations. It is a valid
joint conditional configuration, even though no bijective global frame makes
all modal links consistent. Forcing consistency sacrifices1/3 expected match
mass summed over the three edges. Different routes move the same cold profile
to different class addresses; maximum discrepancy.977732 on the exact probe
set. That large discrepancy is an effect of hard coordinate reassignment, not
a change in an observed person's personality.

The incompatible case sets AB and BC to exact equality but CA to an exact
cyclic shift. Each pair is locally normalized with uniform marginals. However,
no common three-variable distribution can realize all three. They could arise
from different source populations or histories; calling them one homogeneous
joint population is the error.

If all three selected relations hold, the reference class must be fixed by H.
Taking expectations of the corresponding indicator inequality gives

    sum_edges Pr(relation fails) >= Pr(reference class is moved by H).

The valid soft world has1=1. The incompatible world requires0>=1, a direct
contradiction. This is a classical probability/union-bound consequence, not
evidence for quantum contextuality or a novel physical law.

An LP with27 nonnegative joint cells and the three9-cell marginal constraints
returns feasible witnesses for the first three worlds and infeasibility for
the fourth. Feasible-table replay errors <=6.00e-16. A witness is ONE possible
joint table; its existence does not uniquely identify higher-order relations.

## Preserve soft relations: they are not automatically nuisance

A stochastic conditional operator, if J is a valid probability coupling, is

    K_(d<-c) = J_cd^T diag(pi_c)^-1.

Unlike a permutation, this operator can preserve several possible conditional
responses to the same local state. In general K_(e<-d) K_(d<-c) need not equal
K_(e<-c). Equality needs the appropriate Markov/conditional-independence law;
its failure does not by itself contradict a valid common joint distribution.
HM143 preserves J but does not claim to estimate a causal or temporal response
operator from these cross-sectional pair frequencies.

The independent pair case also has an important limit. A uniform independent
triple and the mod3 parity law (a+b+c=0, nine cells of mass1/9) have exactly
equal marginals AND all pair laws. Their latent joint TV is2/3; after the fixed
observation channels their observed joint TV is.101952667. Therefore absence
of identifying pairwise bridges is not absence of all shared author structure.
Appropriately qualified same-author three-condition observations can reveal
the higher-order interaction. More precise PAIR fitting cannot do so.

## Empirical constructor and what it enables

Three fixed seeds ran with1,080,000 local source calibration instances,
1,620,000 paired source bridge instances and54,000 cold application instances.
These are simulated authors, not human sample sizes. CPU2.581565500s, zero new
encoder/reader/API/private-data/training calls. No K, probe, EM, rank or
hyperparameter search.

All stable-world candidate cycles were flat; all soft-response candidate
cycles moved every class. Empirical forced costs were0 versus.31710-.32853,
consistent descriptively with the exact0 and1/3 mechanisms. Signed inverse
minima reached-.01186 and remain unprojected.

Even the INDEPENDENT finite samples returned unique argmax frames, with small
gaps .000505-.001465. This is random symmetry breaking, not a significance
result. Exact independence has36 tied frames. A production score must not use
numerical uniqueness as proof of a shared factor; uncertainty for estimated
observation laws and couplings is a separate construction requirement.

All486000 cold conditional probabilities and486000 reference-candidate values
were replayed. Alternative direct Bayes error3.89e-16, alternative linear
solve error1.01e-15;1,620,000 pair rows independently counted by alternate
arithmetic.75 focused tests pass, including15 new tests. This is same-agent
verification, not blind replication.

The useful addition is an anonymous-coordinate bridge that does not require
psychological names, preserves conditional differences and tells apart three
objects: common class correspondence, valid conditional configuration, and
incompatible population merging. It connects GRAIN's condition-specific
measurement premise to source-constructible observation laws. It does not
solve natural evidence decoding or name the latent classes as traits.

## Mathematical ancestry and next mechanism

- Anandkumar/Hsu/Kakade2012 Sections2-3 supplies the local observable operator:
  https://proceedings.mlr.press/v23/anandkumar12/anandkumar12.pdf .
- Pachauri/Kondor/Singh2013 Section2, Eqs1-5 supplies permutation synchronization:
  https://papers.nips.cc/paper/2013/file/3df1d4b96d8976ff5986393e8767f5b2-Paper.pdf .
  The implementation enumerates exactly; its count noise is not their Gaussian
  Wigner model, and no spectral-noise guarantee is transplanted.
- Wainwright/Jordan2008 Section4.1.1/Proposition4/Example14 supplies the
  distinction between locally valid pairs and globally realizable marginals:
  https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf .
- The pinned OpenAI/math symmetric-group manuscript's introduction and
  preliminaries motivate inspecting group actions and coordinate-invariant
  properties. Its universal tensor-square/Kronecker result does NOT imply a
  personality identification theorem. This is methodological inspiration,
  not a direct application of that claimed result.

Next: inventory the already qualified public/permissioned same-author condition
and event units. Construct a SOFT joint conditional response law only where
there is source support, including higher-order evidence when pairs alias.
Do not start another correspondence-margin or estimator-accuracy sweep. Keep
state/choice composition, all previous failures and original18/100/44 scope
open. Natural author/event identity and psychological meaning need their own
evidence, not labels inherited from this finite algebra.
