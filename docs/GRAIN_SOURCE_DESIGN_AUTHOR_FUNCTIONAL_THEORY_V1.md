# GRAIN: source-design-supported author functionals

2026-10-11, HM159. A constructive connection of existing probability/moment
theory to GRAIN author measurement, not a new binomial-mixture theorem.

## 1. What was added

HM158 separates event identity, effective vector carrier and target operation.
HM159 adds a separate source-design contract: a well-read event is not enough
to determine the reference law needed for an author's conditional score.

Previously HM140 specified a choice model, HM142 imposed three latent classes,
and HM143 coupled anonymous conditional classes. F04 already established that
conditional profiles and straight-line means differ. This packet does not repeat
those conclusions. It constructs conditional author functionals with an unknown
population distribution G, without fixing a latent-class count or estimating
all of G, and states the source observations each functional requires.

The empirical chain in the finite world is:

    qualified vector events (C,Y) -> source observable frequencies
    -> registered observation kernel -> target-functional coefficients
    -> independent application history -> conditional author profile.

The observation kernel is stipulated and checked in a controlled world, not
discovered from arbitrary natural text. The behavioral target is an expression
propensity, not Big5, emotion intensity, or a latent class with a psychological name.

## 2. Population size cannot substitute for within-author observation support

For p in [0,1], t conditionally independent binary observations have kernel

\[
B_{t,k}(p)=\binom tk p^k(1-p)^{t-k},\qquad
f_k=\int B_{t,k}(p)\,dG(p).
\]

Every history probability is a polynomial of degree at most t. More source
authors estimate these frequencies more precisely, but add no new polynomial
degree. This is the established finite-mixture moment limitation, not a new
claim that large population samples are useless.

An exact witness uses p_i=i/5 and masses binom(5,i)/16 on even versus odd i.
The fifth finite difference annihilates all polynomials of degree <=4, so every
ordered four-observation source history has exactly the same probability in
the two populations. Both have population mean p=.5 and variance .05.

Yet the fraction p>.5 is .3125 versus .6875. More importantly, for a NEW author
with exactly two successes in four observations:

\[
E_{G_{even}}[p\mid K=2]=.47272727,\qquad
E_{G_{odd}}[p\mid K=2]=.52727273.
\]

The same source observations and same new history place this author on opposite
sides of the common reference mean. Six-atom witness-grid bounds are exactly
[.47272727,.52727273]; these are identification bounds, not 95% confidence
intervals, and are not claimed sharp over unrestricted continuous G.

This identifies a specific theory gap that perfect reader accuracy cannot fix.
It does not assert that all rank orders are reversed or that this size of
ambiguity occurs in PANDORA.

## 3. Construct the required functional, not the entire population law

For application history h=(m,k), the conditional mean is

\[
\mu_h=\frac{\int p B_{m,k}(p)\,dG}{\int B_{m,k}(p)\,dG}.
\]

The denominator has degree m and numerator m+1. The conditional second moment
requires m+2. In a Bernstein source basis of degree n, compile directly:

\[
p^r B_{m,k}(p)=\sum_{j=k+r}^{n-m+k}
\frac{\binom mk\binom{n-m-r}{j-k-r}}{\binom nj} B_{n,j}(p),
\quad n\ge m+r.
\]

Then the corresponding integral is a fixed dot product with source frequencies.
No latent individual labels, application answers, teacher scores, fitted mixture
components or selected class count enter the cold scorer. The polynomial identity
holds throughout [0,1]. A continuous Beta(2,3) source independently verifies
all five means and variances against its analytic posterior.

Fixed-five source observations identify the four-history mean but not its
variance in general. Fixed-six observations identify both. Identification of the
functional does not identify this author's true p exactly: after K=2, the
matched populations both retain conditional variance .02380165. Population
identification, individual uncertainty and finite-source estimation are separate.

## 4. Preserving choice is useful, but not automatically twice the information

The primary free-phase law is C|p and R|p independent Bernoulli(p), Y=C xor R.
Registering (C,Y) preserves both C and R, giving six independent conditional
trials from three events IN THIS LAW. Fixed-four source histories have between-
population total variation0; the joint-three histories have TV .072. Both cold
mean and variance become constructible. Dropping C leaves Y probability
2p(1-p), invariant to p -> 1-p, and cannot distinguish the reflected populations
at any number of events.

The separately sealed, explicitly post-outcome supplement tested alternatives:

| Observation law | Three-event source TV between matched populations |
| --- | ---: |
| Same p, independent C/R given p | .072 |
| Choice parameter .5, response parameter p | 0 |
| R=C, copied within-event information | 0 |

From source histories with independent episodes, observable necessary checks are

\[
D=E[(C_1-R_1)(C_2-R_2)]=E[(a-b)^2],
\]
\[
V=E[C_1R_1-C_1R_2]=E[\operatorname{Cov}(C,R\mid u)],
\]
\[
V_a=E[C_1C_2R_2-C_1C_2R_3]
=E[a\operatorname{Cov}(C,R\mid u)].
\]

The distinct-parameter witness gives D=.05; the copy gives V=.2. A balanced
dependent witness has D=V=0 but V_a=.005 and three-event TV .0507 versus the
independent law. Therefore zeros of a few pooled checks do not certify conditional
independence. These are standard expectation identities; the construction rule
is to retain the source joint law and declare assumptions, not certify a law
from channel count or a zero pooled correlation.

Repeated mentions of one event are deduplicated. Conversely, two observations
in one event are not automatically independent: provenance and observation law
are both necessary. The followup does not revise the primary law after seeing
results, and does not open a new dependence/precision grid.

## 5. Finite-source and actual execution

Primary new data comprise 18 simulated-source runs, three worlds, three seeds,
N=2000/20000; 198000 source-author instances and1584000 exact-alphabet vector
events. These counts describe execution, not real participants or theory
completion. All four prototypes are existing public-source GRAIN15 vectors.
The simulator emits them exactly; decoding is not a natural-text generalization.

One of90 sampled joint mean/variance profiles has empirical variance -.00826821
(odd/seed159/N2000/history3). It is retained as INADMISSIBLE_EMPIRICAL_MOMENTS,
not clipped or turned into negative psychological uncertainty. Finite observed
frequencies need not lie in the exact mixture cone. No mixture refit or larger-N
search was used to rescue it. The two frozen N values distinguish sampling error
from the exact identification limit; no winner is selected.

A separate cold CLI reads only source frequencies and history counts. The
even/seed161/N20000 source produces history2 mean .47048460 and variance
.02742233; it neither reads true p nor recovers an exact point for this author.

## 6. Consequence for the mainline

The required chain now distinguishes:

1. Qualified event observation F and its vector carriers.
2. Conditional behavior/diagnostic law ell, including opportunity and dependencies.
3. Population/reference law G and the observation family that identifies it.
4. The intended author functional and its source-supported numerator/denominator.
5. Individual uncertainty, source sampling error and transport assumptions.

This is an executable addition to author-measurement construction, not another
model-answer fidelity task. It does not identify a natural psychological ell or
transport the registered law to a new task automatically. Next study that kernel
and target anchoring using qualified source contrasts; do not extend N/degree/
class-count grids or treat an arbitrary mathematical propensity as Big5.
Choice, state, dynamic response and prior GRAIN geometry remain valid separate
channels rather than being replaced by a single binomial channel.

## 7. Sources and artifacts

- Tian, Kong, Valiant2017, *Learning Populations of Parameters*, Proposition1
  and Sections2.1-2.2: finite observation limits and moments. This is prior art.
  https://papers.nips.cc/paper_files/paper/2017/file/bc4e356fee1972242c8f7eabf4dff517-Paper.pdf
- Vinayak etal2019, *Maximum Likelihood Estimation for Learning Populations of
  Parameters*, Sections1-3/Remark3.1: observed fingerprints versus mixing laws.
  https://proceedings.mlr.press/v97/vinayak19a/vinayak19a.pdf
- OpenAI/math adc7f124 joint metric/connection Introduction1-130: observation
  families and indistinguishable internal objects, design inspiration only.
  No elliptic/manifold/unitary theorem is transferred to authors or LLMs.

Canonical `results/grain_math_response_v1/history_resolution_r1/`:
`exact.json`, `exact_profiles.csv`, `source_design_aliases.csv`,
`finite_author_bounds.csv`, all observed source bundles, `sampled_profiles.csv`,
`alternate_audit.json`, `cold_cli_score.json`, `design_gate_followup_r1/`.
Primary and supplementary scientific dependencies have separate manifests.
Rational ordered-event audit and factorial-moment versus Bernstein calculations
agree; maximum sampled-profile arithmetic discrepancy4.61e-15. Same-agent
alternate mathematics, not blind third-party reproduction.

13 new/30 focused tests and seven Python lint/format checks pass. No API/private
MEPS/neural calls/training/Git/Claude/automation/lockbox/old queue. Numerical
scoring here is non-LLM; historical vector prototypes still originate in neural
encoding, so raw-text-to-measurement is not claimed wholly LLM-independent.
Goal remains ACTIVE; original18/100/44 is not promoted or declared complete.
