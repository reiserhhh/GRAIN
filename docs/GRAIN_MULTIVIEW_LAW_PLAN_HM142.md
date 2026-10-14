# HM142: learn an anonymous observation law, rather than copy a teacher

2026-10-11. CPU-only finite identification experiment. Freeze this plan, config,
module, runner and tests before the first research simulation. Old18/100/44 stay
OPEN. No API/private text/new neural training/Git/Claude/automation/lockbox.

## Mainline gap and existing work

HM134 distinguishes selected evidence e, observer F_q(e), and diagnostic law
ell(e|theta,design). HM140 supplied a logit law; HM141 manipulated an explicit
population reference inside a reader. Neither learned an unknown diagnostic law
from anonymous source authors. HM139 teacher interpolation is not resumed.
HM81 joint moments did not identify a latent mixture. HM100 discussed a finite
latent class abstract but did not implement this source-law constructor. The
20260908 seminar correctly warned that three queries do not automatically meet
Kruskal's conditions. This packet tests a declared three-view model, not three
LLM questions mistaken for independent observations.

## Frozen experiment

Use the first9 source-only GRAIN15 menu contrast vectors from HM140 as three
finite vector alphabets, three categories each. They were calculated from public
fictional action records by the frozen encoder. The simulated observations are
these vectors, not newly generated text or real author responses. Exact alphabet
decoding is explicitly a finite dictionary operation, not natural-language
generalization. View groups are designed independent event channels; their
arbitrary grouping does not discover topics or psychological factors.

Three anonymous latent components are a fixed design parameter, not an estimated
number of personality types. Source has no component labels or teacher outputs.
The generator uses the frozen probability tables; the estimator receives only
observed category frequencies, fixed k=3 and eta. Oracle tables/labels can enter
evaluation only after the source function and cold scores are frozen.

One fixed spectral contraction, no eta/rank/K/EM/ridge grid. Three seeds, 120000
source authors and6000 disjoint application authors each. Each application has
two bundles of three events: stable-component and fresh-occasion worlds share
the exact first bundle/source law but differ in their second occasion.

Population battery: ordinary law, positive rotated pairwise-alias law, rank-one
uninformative law, nonseparating third-view law, and copied-one-event diagonal
law. The diagonal law can fit perfectly as an apparent latent mixture; its
event ledger must reject three copies as three independent events. Different
event IDs alone are not a proof of conditional independence.

## Mathematical constructor

For observed categorical x_v, latent h and positive pi:

    P12 = M1 diag(pi) M2^T
    T = sum_h pi_h M1[:,h] tensor M2[:,h] tensor M3[:,h]
    B_eta = (U1^T T(eta) U2)(U1^T P12 U2)^-1
          = (U1^T M1) diag(M3^T eta)(U1^T M1)^-1.

Recover M1 from eigenvectors normalized to categorical column sum1; recover pi
from the first marginal, M2 from P12, M3 from coordinate contractions in the
same eigenbasis. Components remain anonymous, ordered by source eigenvalue.
Negative probability estimates, deficient rank or repeated eigenvalues are
retained failures, not clipped into an apparent success. Only <=1e-10 numerical
roundoff can be set to0. Reconstruction and off-diagonal residuals are reported.

Cold inference uses pi_h product_v Mv[x_v,h] and normalization, with source
parameters fixed. Qualified repeated mentions of an event contribute once.
Unresolved qualification or inconsistent same-event re-encoding is rejected.
This is one conditional evidence channel, not a replacement for GRAIN condition
selection, state, narrative, response or vector-operation channels.

## Pairwise-equivalence witness and occasion distinction

Let A_v=M_v diag(sqrt(pi)), Q be a fixed orthogonal rotation, q'=Q^T sqrt(pi).
For positive q' and emissions, set pi'=q'^2 and M'_v=A_v Q diag(1/q').
All means and every pairwise law are exactly unchanged, but the third-order
tensor generally changes. Record that difference and conditional posterior
differences; do not reduce this question to greater prediction accuracy.

A stable author component and a shared occasion component can give exactly the
same single-bundle tensor. Class identifiability is not cause identifiability.
Use the second independent bundle to measure cross-occasion covariance and
predictive log score under both worlds. Anonymous class posterior is not Big5,
MBTI, emotion or a human psychological construct.

## Acceptance and artifacts

Exact law recovery up to component permutation and pair invariances <1e-10;
nonzero triple-law difference. Rank/eigen failures retained. All three fixed
sample runs reported even if any fails. Descriptive sample RMSE/TV target .05
does not license extra runs. Cold probabilities nonconstant and normalized;
evaluate distance to oracle posterior, not teacher r. No latent label enters
the scoring function. Exact event-repeat invariance and same-event copy check.
Cross-occasion law is assessed rather than assumed stable. Full read-only
replay of model, tensors, all application scores, metric rows and input hashes.

Artifacts: manifest, alphabet, observed categories (no truth fields), separate
oracle evaluation archive, source models, frozen cold scores, population witness,
metrics, verification and result. 120seconds CPU wall cap, no background work.
Initial implementation/unit-test fixtures are not research sample runs.

## Primary sources actually consulted

Anandkumar/Hsu/Kakade2012 Sections2.2-2.4,3.1-3.3 Lemmas4-6/AlgorithmsA-B:
https://proceedings.mlr.press/v23/anandkumar12/anandkumar12.pdf . Standard
observable-operator identification, not our invention. Categorical normalization
permits this simplified M1-first implementation. We do not inherit the paper's
random-contraction concentration guarantee with our one fixed eta.

Allman/Matias/Rhodes2009 Section4 Theorem1/Corollary2, Section5 Theorem4:
https://www.cs.uaf.edu/~jrhodes/papers/Latent.pdf . Column convention transposed:
full3-column ranks give3+3+3>=2*3+2. Conditions are structural sufficient
conditions; not obtained automatically by slicing natural text.

OpenAI/math adc7f124 elasticity Introduction: noninjective auxiliary reduction
does not identify the physical inverse map. Reread the actual displacement/
divergence comparison argument's introduction, not the whole PDE proof. Useful
design principle: preserve the observable law that distinguishes candidates,
not merely a compressed pairwise summary. No elasticity theorem is transplanted.
https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Global-Uniqueness-for-the-Smooth-Isotropic-Elasticity-Inverse-Problem-September-24-2026/build/sections/introduction.tex
