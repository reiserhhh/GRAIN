# GRAIN M3 Micro-to-Meso Mechanism Atlas

Status: discovery theory after a synthetic `PASS`; not sealed confirmation and
not a human-text or personality claim.

Overall decision:
`M3_INTERMEDIATE_MECHANISM_ATLAS_DISCOVERY_SUPPORTED`.

## 1. Why an intermediate layer is necessary

The earlier M3 notation allowed the microscopic event law to be projected
directly into an author response field. That is mathematically valid as a
projection, but it leaves the substantive coarse-graining mechanism
unspecified. The new working object is an atlas rather than a single bridge:

\[
\mathcal A_u
=
\left(
\mu_u,\,
\mathcal R_u,\,
\mathcal K_u,\,
\mathcal C_u,\,
\mathcal P_u
\right).
\]

- \(\mu_u\): distribution of expressed states under a declared opportunity
  measure;
- \(\mathcal R_u\): condition-to-response operator;
- \(\mathcal K_u\): temporal evolution operator and its slow modes;
- \(\mathcal C_u\): response to another actor or interactional input;
- \(\mathcal P_u\): higher-order path information not contained in a
  first-order transition.

These are competing or coexisting coarse-grainings. They are not five
mandatory modules to concatenate, and none is licensed as personality without
human-text persistence and external psychological or behavioral anchoring.

## 2. Interdisciplinary foundations

### Psychology

Fleeson's density-distribution account treats a trait-relevant behavioral
profile as a distribution of states, including central tendency and
variability rather than one observation
([Fleeson, 2001](https://doi.org/10.1037/0022-3514.80.6.1011)).
Whole Trait Theory subsequently separates descriptive state distributions
from explanatory mechanisms
([Fleeson & Jayawickreme, 2015](https://doi.org/10.1016/j.jrp.2014.10.009)).
This supports \(\mu_u\), but does not imply that any text-vector distribution
is a personality trait.

CAPS describes stable person information as situation-behavior `if-then`
patterns rather than context-free behavior
([Mischel & Shoda, 1995](https://doi.org/10.1037/0033-295X.102.2.246)).
This is the closest psychological precedent for \(\mathcal R_u\). It also
implies that opportunity must be fixed or explicitly modeled; otherwise
condition selection and conditional response are not identifiable.

Idiographic psychology warns that between-person structure and within-person
dynamics are not interchangeable unless restrictive ergodic conditions hold
([Molenaar, 2004](https://doi.org/10.1207/s15366359mea0204_1)).
Dynamic structural equation modeling similarly separates within-person lagged
effects, between-person differences, and random effects
([Asparouhov, Hamaker, & Muthen, 2018](https://www.statmodel.com/download/DSEM.pdf)).

### Linguistics and sociology

Interactive alignment and communication accommodation show that an utterance
is partly a response to the interlocutor, at multiple representational levels
([Pickering & Garrod, 2004](https://doi.org/10.1017/S0140525X04000056);
[Danescu-Niculescu-Mizil et al., 2011](https://doi.org/10.1145/1963405.1963509)).
Consequently, stable coupling or resistance to partner input belongs to
\(\mathcal C_u\), while the partner's style itself is a condition.

Relational-event models treat social action as ordered events whose hazard
depends on recency, persistence, participation shifts, and network history
([Butts, 2008](https://doi.org/10.1111/j.1467-9531.2008.00203.x)).
This supports a separate event-history mechanism rather than reducing
interaction to author means.

### Mathematics and statistics

Kernel mean embeddings represent probability distributions as elements of an
RKHS and permit nonparametric comparison of marginal and conditional laws
([Muandet et al., 2017](https://doi.org/10.1561/2200000060)).
They motivate the current distribution summary and a future conditional-law
operator beyond linear regression.

Koopman theory studies nonlinear dynamics through a linear operator acting on
observables. Extended dynamic mode decomposition estimates finite
approximations to that operator
([Williams, Kevrekidis, & Rowley, 2015](https://doi.org/10.1007/s00332-015-9258-5)).
The present experiment estimates only a low-order linear lag operator; the
name `koopman_spectrum` is therefore an engineering label, not evidence that a
nonlinear Koopman-invariant subspace has been recovered.

Path signatures encode ordered path geometry through iterated integrals and
can retain information absent from unordered summaries
([Chevyrev & Kormilitzin, 2016](https://arxiv.org/abs/1603.03788)).
The current `higher_order_path` estimator is only a lag-two memory residual,
not a full path signature.

## 3. Discovery experiment

The atlas generates independent train/test event panels for the same authors.
Each positive world isolates one mechanism while suppressing simpler author
means where possible:

| World | Planted object | Registered estimator |
| --- | --- | --- |
| density | symmetric mixture weight | residual-response kernel mean |
| conditional | author rotation of condition input | condition-response slope |
| metastable | author persistence | lag-one operator spectrum |
| interaction | author rotation of partner input | partner-response slope |
| higher order | lag-two persistence with null lag-one dependence | lag-two memory spectrum |
| opportunity | author condition exposure only | condition profile |
| null | no author law | all methods should be chance |

The primary score is cross-panel same-author AUC. Pairwise-distance Spearman
correlation checks whether the recovered author geometry matches the planted
parameter without requiring coordinate alignment. A mechanism is selective
only when its registered estimator wins the matching world and the null and
opportunity controls remain calibrated.

## 4. Discovery result

At 96 events per occasion and 10 independent seeds:

| Mechanism | AUC | Truth-geometry rho | Win fraction |
| --- | ---: | ---: | ---: |
| state distribution | 0.759 | 0.789 | 1.00 |
| conditional response | 0.940 | 0.830 | 1.00 |
| lag-one persistence | 0.632 | 0.461 | 0.90 |
| interactional coupling | 0.938 | 0.831 | 1.00 |
| lag-two memory | 0.679 | 0.616 | 1.00 |

The null maximum mean AUC is 0.506. Opportunity profile AUC is 0.839. After
removing regression intercepts from operator features, the largest
opportunity-only response-residual AUC is 0.516. This correction matters:
including the intercept had allowed condition exposure to masquerade as an
interaction coefficient.

## 5. Licensed conclusion

The synthetic result establishes an existence proof for a mechanism-selection
layer between event vectors and author geometry:

\[
\mathcal N_u
\xrightarrow{\text{candidate coarse-graining}}
\mathcal A_u^{(m)}
\xrightarrow{\text{independent-view audit}}
\widehat{\mathcal A}_u^{(m)}.
\]

It shows that an author can be stable for different mathematical reasons:
distribution shape, conditional response, temporal persistence,
interactional coupling, or higher-order memory. It also shows that opportunity
exposure is a separately measurable author-associated object and must not be
silently absorbed into a response operator.

It does not establish that these mechanisms exist in human text, are
psychological constructs, form a complete basis, or are identifiable from
uncontrolled corpora.

## 6. Remaining attacks

1. **Pure distribution shape**: hold both mean and covariance fixed while
   varying modality or tails. The current density world can be partly solved
   by covariance.
2. **Nonlinear slow modes**: keep simple lag-one statistics matched and vary
   hidden metastable dynamics. Use EDMD/VAMP dictionaries selected on train
   data and test implied timescales.
3. **Exact path aliases**: match occupancy and the complete first-order edge
   multiset, then vary triadic motifs, loops, or path signature.
4. **Dyadic identifiability**: add multiple partners and separate actor,
   partner, dyad, and occasion effects. A single partner-input stream is not a
   Social Relations Model.
5. **Composite decomposition**: a high union AUC only proves useful combined
   representation. It does not prove that the planted mechanisms can be
   independently recovered when superposed.
6. **Actual data attacks**: inject MNAR, technical dependence, reference
   drift, mixed coarsening, and support failure as value-generating processes,
   not metadata flags.

## 7. Follow-up discoveries

The low-order-matched V2A attack retained only three of five mechanisms:
nonlinear condition response, AR(2) dynamics beyond matched lag one, and
nonlinear partner response. The original single-bandwidth KME failed the
equal-covariance shape world, and the generic lag-three spectrum failed pure
lag-three memory.

V2B replaced those two summaries with an author-whitened multiscale empirical
characteristic function and a lag-three partial operator. All five
low-order-matched attacks then passed. V3 independently permuted all five
author parameters, ran every pairwise superposition, and removed each
mechanism counterfactually. Own-truth partial geometry remained high,
cross-talk stayed low, and knockout geometry returned near zero.

The complete numerical record and the stricter claim boundary are in
`reports/SUICA_M3_MECHANISM_ATLAS_SYNTHESIS.md`. These follow-ups remain
matched-family discovery: they establish recoverability and pairwise
decomposability under known mechanism classes, not universal mechanism
discovery.

## 8. Formal discovery object

The current finite discovery object is written in the 2026-07-30 foundation
notation as an anonymous L4 object:

\[
\mathfrak A_u^{\mathrm{disc}}
=
\Delta_{P_0}\left[
D_u^\Omega
\oplus R_u^{\mathcal B}
\oplus K_u^{(2)}
\oplus C_u^{\mathcal B}
\oplus P_{u,3\mid1:2}
\right].
\]

The components are deliberately narrow:

\[
D_u^\Omega
=
\left\{
\Re\varphi_u^\circ(\omega),
\Im\varphi_u^\circ(\omega)
\right\}_{\omega\in\Omega},
\qquad
\varphi_u^\circ(\omega)
=
\mathbb E[
\exp(i\omega^\top W_u(X-\mu_u))
],
\]

\[
R_u^{\mathcal B}
=
\Pi_{\mathcal B_{\mathrm{NL}}}
\mathbb E[X\mid Z,u],
\qquad
C_u^{\mathcal B}
=
\Pi_{\mathcal B_{\mathrm{NL}}}
\mathbb E[X\mid P,u],
\]

\[
K_u^{(2)}
=
\operatorname{spec}
\begin{pmatrix}
A_{u,1} & A_{u,2}\\
I & 0
\end{pmatrix},
\]

\[
P_{u,3\mid1:2}
=
\arg\min_{A_3}
\mathbb E\left\|
X_t-A_1X_{t-1}-A_2X_{t-2}-A_3X_{t-3}
\right\|^2.
\]

These are finite projections chosen for discovery. They must not be renamed
as a complete distribution, general Koopman operator, full interaction
mechanism, or general path signature until cross-family confirmation passes.

## 9. M4-A composition upgrade

The atlas is no longer the terminal mesoscopic object. The 2026-07-31 M4-A
development adds an author mechanism-composition signature:

\[
\mathcal A_u
\longrightarrow
\Lambda_u
=
\left(
\{d_u^\otimes(A)\},
\{d_u^{\mathrm{obs}}(A)-d_u^\otimes(A)\},
\{G_{g\rightarrow i}\},
\{\Omega_{ij}\}
\right).
\]

Finite synthetic discovery separated genuine structural synergy from
dependence-induced redundancy and suppression, recovered pair/triple
hyperedges, detected directional gating, and refused perfect aliases. This
closes the original atlas attack “composite decomposition” only for the
registered order-three multilinear world family. General dependent-input
HOFD, human-text mechanism discovery, and completeness remain open. A
designed dynamic extension additionally recovered noncommuting
condition/history transition order, commutator geometry, and temporal gate
strength, but its path-score gain combines marginal-kernel, order, and gate
improvements. These results license M4-B development, not a causal or
psychological interpretation. See
`docs/SUICA_M4_MECHANISM_COMPOSITION_THEORY.md`.

M4-B subsequently converted opportunity from a fixed precursor into an
endogenous ecology. In finite matched-marginal worlds, the system separated
conditional selection from expression-driven opportunity creation, recovered
feedback-loop geometry, return/recovery dynamics, and observable history
gates, and refused hidden source aliases. The full loop matrix survived; its
spectral-radius author ranking did not. The validated extension is therefore
an ecology operator family rather than a scalar stability trait. See
`docs/SUICA_M4_ENDOGENOUS_OPPORTUNITY_ECOLOGY_THEORY.md`.

M4-C then removed planted condition coordinates for a bounded response edge.
It learned anonymous condition geometry from pre-response reference panels,
transported relation geometry across two numeric sources, and preserved a
frozen conditional-response surface while refusing author/response leakage,
no-manifold inputs, and noninjective response aliases. The licensed object is
a response-safe condition relation atlas, not a topic ontology or named
factor. M4-C.2 then recovered atomic choice, creation, return, and recovery
actions under the discovered chart but failed its composite loop and
repetition gates. Thus the complete M4-B ecology is not yet transport-closed.
M4-C.3 audits physical-edge composition; M4-D remains blocked. See
`docs/SUICA_M4_CONDITION_MANIFOLD_THEORY.md`,
`docs/SUICA_M4_CHART_COVARIANT_ECOLOGY_THEORY.md`, and
`docs/SUICA_M4_PHYSICAL_EDGE_COMPOSITION_THEORY.md`.

M4-C.3.1 subsequently repaired the statistical-object mismatch: a centered
author-relation Möbius budget predicted loop loss with pooled Spearman `.9868`
and allocated about 80% of symmetric loss to creation. A creation-only
intervention then rejected calibration-only isotropic RBF remapping: the best
arm reduced loop geometry from `.6702` to `.5316` despite `.2497` oracle
creation headroom. This isolates the next question to author-specific
derivative estimation, not further unsupervised chart smoothing. See
`docs/SUICA_M4_RELATIONAL_ERROR_BUDGET_THEORY.md` and
`docs/SUICA_M4_CREATION_ESTIMATOR_INTERVENTION_THEORY.md`.

The subsequent cross-fitted Fisher-Wiener intervention improved loop geometry
from `.7198` to `.7699` while satisfying predictive-risk, permutation, null,
and gauge controls, but recovered only 25.4% of oracle creation headroom and
passed 5/8 repetitions. This supports a partial author-derivative estimation
noise mechanism while refusing closure at the current opportunity budget.
The subsequent opportunity-excitation frontier increased minimum Fisher
information `13.04x`, produced a positive geometry dose slope (LCB `.0222`),
and raised the endpoint by `.0802` (LCB `.0611`). Yet it recovered only 44.0%
of oracle headroom and again qualified in 5/8 repetitions. Passive opportunity
contributed `.0611` of the endpoint and orthogonal excitation another `.0192`;
geometry saturated by `K=4`, while world-specific headroom recovery ranged
from negative to `.810`. Information is therefore a real mechanism variable
but not a sufficient explanation. Additional `K`, chart-kernel replacement,
and embedding replacement are stopped; the next M4-C object must model
declared creation/history heterogeneity without oracle access.

M4-C.3.4 tested that heterogeneous-observation explanation without changing
the gate-zero direct creation estimand. A complete `C/S/P` attribution cube
separated truth-open condition chart, target-aligned history/source
likelihood, and cross-author pooling. The full cube recovered only `.3405` of
oracle headroom. The proposed observation-law factor was negative in both
target worlds on every repetition (mean `-.0844`, clustered LCB `-.1581`);
pooling contributed only `.0090` by Shapley allocation. In contrast, the
truth-open chart factor contributed `.0727`, led the next factor by a
clustered LCB of `.0396`, and alone recovered `.7854` of headroom. This
rejects omitted history/source likelihood as the dominant residual account
and localizes the remaining recoverable error to the condition chart.

That localization is diagnostic, not a chart solution. The positive chart
cell reads truth, and alias information loss was detected in only `7/8`
repetitions. The next admissible object is therefore a response-safe,
truth-blind chart estimator evaluated against the frozen oracle attribution
ceiling. It must not be confused with the previously rejected shared
isotropic RBF smoothing route. M4-C.2 remains open and M4-D remains blocked.

M4-C.3.5-R1 identified that response-safe chart with a replicated cross-source
CCA quotient. R2B then tested downstream replacement under a physical
Phase-A/Phase-B seal. The registered conditional effect was strong:
eligible gain `.1227` (LCB `.0934`), oracle-headroom recovery `.8146` (LCB
`.6798`), and accessible efficiency `.9874` (LCB `.9801`). The effect beat
variance- and repeatability-matched equal-rank controls, while history,
compensation, null, alias, source, gauge, CKA-permutation, and hazard controls
passed.

R2B nevertheless remains partial because the native zero-refusal gate failed:
two of 160 cells fell outside the frozen calibration support, including one
eligible main cell. This changes the mathematical object. The chart is not a
total replacement map; it requires an explicit applicability description.
R2C therefore evaluated the routed operator

\[
\Pi = A R + (1-A)B0
\]

on every planned cell, where \(A\) is determined before response opening.
Deleting refused cells or reporting only accepted-stratum gains would change
the target population and is not admissible.

The first fresh R2C development had `16/16` accepted cells and thus identified
implementation but not routing utility. A paired boundary-ecology experiment
then held response truth fixed while moving evaluation points across the
frozen scalar support boundary. It falsified the implied monotonicity

\[
C_i\downarrow\quad\Longrightarrow\quad
d_i=\Gamma_{R,i}-\Gamma_{B0,i}\downarrow.
\]

Fallback lost `.0528` relative to always using `R`, refused cells retained
mean forced-`R` gain `.0880`, and scalar harm AUC was only `.554`. The same
coverage value represented beneficial source-partition extrapolation and
harmful source-rotation mismatch. Hence scalar coverage is a support audit,
not a sufficient statistic for relative chart utility.

The admissible next object is direction-sensitive:

\[
d_i=f_w(q_i)+\varepsilon_i,\qquad
q_i=\Phi_{\mathrm{pre}}(X_i),
\]

\[
\mathcal D_R=
\left\{
q:
E[d\mid q]\ge-\delta,\;
P(d<-\delta\mid q)\le\alpha
\right\}.
\]

The initial 24-component \(q\) pilot improved cross-world harm AUC only to
`.588`, with negative OOF \(R^2\) and negative routing value. It is therefore
unresolved. Before a larger study, departure direction must be independently
varied at fixed coverage; otherwise support amount and support geometry remain
confounded.

That fixed-coverage experiment is now complete. It held support at `12/16` or
`13/16` and crossed radial-dispersed, radial-concentrated, tangential, and
source-asymmetric departures across eight repetitions. All 72 null gains and
all replay errors were zero, but `Q` failed both transport views: LOWO harm
AUC `.533` (LCB `.464`), gain \(R^2=-1.616\), routing value `-.0243` (LCB
`-.0417`); LORO AUC `.512` (LCB `.477`), \(R^2=-.195\), routing value
`-.0134` (LCB `-.0314`). The historical threshold lost `.0503` relative to
always using `R`, while the scalar learner routed every cell to `R`.

This closes the current applicability search negatively. The observables
describe support geometry but do not identify relative chart utility. The
admissible object is therefore a design-level domain license, not a learned
cell gate: prospectively support `R` inside a declared world/study class and
otherwise expose `B0/R` sensitivity or refuse. A new gate requires a genuinely
new observable mechanism and fresh data, not threshold or feature retuning on
this panel. M4-C.2 remains open and M4-D remains blocked.
