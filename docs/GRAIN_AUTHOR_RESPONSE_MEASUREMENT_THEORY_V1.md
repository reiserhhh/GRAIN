# Author evidence levels and observer-response measurements

2026-10-10. HM109 finite construction; existing research goal ACTIVE. This is
an observation-law result, not a new personality scale or a general inverse
of an LLM. Canonical evidence: `author_response_observability_r1` and its
separately sealed `units_companion`.

## 1. Object and qualification

Let E be registered evidence atoms and p an explicitly normalized positive
author evidence distribution. In HM109 each atom has a bridge, current carrier,
historical action and polarity. Only the two corners whose historical action
belongs to Person A enter the target-A author distribution. Person-B corners
remain provenance for the paired computational decomposition, not A evidence.

N(e) is the native reading: all six queries, each in score and log-odds units.
G(e) is the existing pre-answer local GRAIN reading. L(e) and R(e) are the
HM108 additive-readout and joint-field-lift components; N(e)=L(e)+R(e).
C(e) records the condition/action group. Nothing here subtracts condition or
choice means. D(e)=N(e)-N_joint_removed(e) is an actual model-intervention
contrast, and U(e)=N(e)-N_random(e) is its registered random comparator.

An author level and an observer-response profile are different objects:

\[
 S_N(p)=p^\top N,\qquad S_G(p)=p^\top G,\qquad
 T_D(p)=p^\top D.
\]

T_D measures how this instrument responds when reading the author's evidence.
It is not the author's psychological reaction to an intervention. The same
distinction applies to LLM outputs and any future behavioral interpretation.

## 2. A finite identifiability criterion

For B=[C,N], two evidence laws are observationally equivalent when

\[
 B^\top(p-p')=0.
\]

For a scalar response column d, its expectation is determined by B moments
on this finite support iff d lies in col(B). The forward implication is direct:
if d=B beta, then p^T d=(B^T p)^T beta. For the converse set
h=(I-Pi_B)d. Since C includes the constant function, h has zero sum. If h is
nonzero, for 0<c<1 the following laws are strictly positive and normalized:

\[
 p_\pm=\frac{\mathbf 1}{n}
       \pm\frac{c}{n\|h\|_\infty}h.
\]

They have identical B moments, yet

\[
 (p_+-p_-)^\top d
   =\frac{2c\|h\|_2^2}{n\|h\|_\infty}>0.
\]

HM109 sets c=.4, constructs every registered response column, and retains
algebraic nulls rather than selecting a winning witness. This proof is mature
linear algebra. A witness selected from source D is a construction, not an
independent significance test. The criterion concerns linear moments of p;
it does not rule out nonlinear statistics of richer observations.

## 3. What the actual model observations add

On 48 qualified source atoms, [C,N] has rank36. Adding G raises rank to42.
The mean hidden-response norm ratio across the 12 fixed channels decreases
from .250676 to .134478. These are dimensionless per-channel norm ratios,
not fractions of personality information or gains in prediction accuracy.

Under the source-equivalent positive laws, native log-odds difference RMS is
1.948e-16, while actual directed-response difference RMS is .000174203.
The paired L and R differences each have RMS .001526143 and cancel in N.
Thus perfectly estimating the current N mean would not recover this response
distinction. G adds observed directions without fitting it to reproduce N.

Adding L alongside N, or keeping group-resolved N, gives full rank on this
small registered vocabulary. That is finite-support interpolation. It is not
a universal sufficient statistic, a minimal psychological representation or
an argument for unlimited feature expansion.

## 4. Composition, transport and cold computation

The source laws and reference are sealed before exporting numeric bank1/2
observations. The same event histories are scored in three banks, with only
addresses, registered baseline observations and condition metadata passed
to a separate cold process. The process receives no p, truth or intervention
responses. Upstream numeric observations still depend on the original LLM.

Source score equivalence does not survive a changed expression bank: on the
complete cohort, native log-odds contrast RMS becomes .002292 and .002930.
Therefore the equivalence class is instrument/material indexed, not an
absolute identity of two authors. On the 19-group matched cohort, source-to-
known directed-response pattern cosine is .943016 in score and .982288 in
log-odds units. These are existing-material, source-selected comparisons,
not personality correlations or a blind replication. Fresh-bank interventions
are absent and remain missing, not zero.

For measured histories use explicit event counts and preserve group masses:

\[
 \widehat S_u=\sum_e\widehat p_u(e)\,[N(e),G(e),L(e),R(e)],\qquad
 \widehat p_u(e)=n_{ue}/\sum_{e'}n_{ue'}.
\]

Retain conditional means with observation counts; an unobserved condition
is missing. Repeated sampled events here are distinct designed events, not
permission to count duplicate reports as independent evidence in real data.

## 5. Ancestry and contribution boundary

[Littman, Sutton and Singh (2001), Predictive Representations of State](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)
motivates distinguishing current outputs from action-conditioned observations.
Read here: abstract/introduction, Section1 definitions and equations1-3. Its
POMDP representation theorem, Markov update and future-human-behavior claims
are not transferred to this static evidence-law construction.

[OpenAI/math](https://github.com/openai/math), pinned
`adc7f1241b42e322a6451854ab7e4b4c146bf78a`, finite-monoid manuscript
`build/sections/algebra.tex` lines1-135, motivates checking whether outputs
distinguish updates and whether assembled parts have compatible meaning.
This is design inspiration, not a monoid/star-height theorem for neural models.

HM78/82 already established score fibers; HM70/71/81 already studied joint
observations and cold author composition; HM102 separated author evidence
mass from attention mass. The HM109 addition is their concrete connection to
actual HM107 interventions, HM108 paired native accounts and HM106 local
observations on the same qualified support. It is not claimed as a new
nullspace theorem or as established psychological validity.

## 6. Next research decision

Do not optimize total-score fidelity to repair this lost object. Preserve
joint evidence and distinguish author levels from instrument-response laws.
The next question is whether a source-constructible joint response record can
operate on a new qualified evidence unit without memorizing the small support.
Inventory existing cold/action-operator experiments before proposing any run.
This question is not executed by HM109 and does not reopen HM107's budget.
