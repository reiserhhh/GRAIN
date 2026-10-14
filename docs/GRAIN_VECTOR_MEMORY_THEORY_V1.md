# Present readings and observation memory are different objects

2026-10-12 JST. HM211 COMPLETE once; Goal ACTIVE/nojob. Original18/100/44 OPEN.
Canonical `results/grain_math_response_v1/vector_memory_r1`.

## 1. What the actual construction establishes

Actual comments, not constructed positive weight pairs, formed the first two
blocks of each observation. A software reset preserves the ENTIRE current
32-profile posterior but replaces its within-profile component weights by
their original uniform reference. A third disjoint actual comment block then
changes every five-target reading in both held-source directions and the24
panel histories. Current profile discrepancies stay below9.021e-17;
held target-change RMS spans .000478078-.001605491 probability units.
The residual identity matches within6.50e-16; direct-distance replay max1.33e-15.

This gives a working observation-memory interface, not another set of scores.
It does NOT show two naturally occurring prefixes with exactly equal scores:
one prefix state is naturally induced under the declared source law, while
the other is the explicit reset intervention. HM191's stronger abstract
counterfactual sufficiency witnesses remain separate and unchanged.

## 2. Source law and state

Each frozen source component i has inherited profile g_i and three actual
8-comment means z_it. Let nu^2=.08287667382055752. The working observation law
is a product of three isotropic Gaussian channels conditional on component i:

\[
L_{it}(x)=\exp\!\left[-\frac{\|x-z_{it}\|^2}{2\nu^2}\right],\quad
\alpha_i^{(0)}=\frac{1}{32n_{g_i}},\quad
\alpha_i^{(t+1)}=\frac{\alpha_i^{(t)}L_{it}(x_t)}{\sum_j\alpha_j^{(t)}L_{jt}(x_t)}.
\]

The common Gaussian density normalization cancels in this posterior.
The code's dot kernel also omits the query-only exponential, which cancels
in fine/reset evidence ratios. Neither omission defines a normalized density
by itself. Three-channel conditional independence and the12/8 variance
conversion are DECLARED assumptions, not discoveries of natural noise.

Current profile probabilities and target readouts are

\[
p_g=\sum_{i:g_i=g}\alpha_i,\qquad \mu_q=\sum_g p_g Y_{gq}.
\]

These read all functions of the inherited profile at the PRESENT step.
They are not automatically sufficient for further observations. The recursive
state remains alpha, the block index, fixed source-bank identity, and the
qualified event ledger. Component i is a source reference address, not the
claimed identity of the current person or a discovered personality class.

## 3. Score-preserving intervention and its exact action

After two actual blocks define

\[
\widetilde\alpha_i=\frac{p_{g_i}}{n_{g_i}},\quad
r_i=\alpha_i-\widetilde\alpha_i,\quad
\sum_{i:g_i=g}r_i=0\quad\text{for every }g.
\]

Thus r lies in the null space of the CURRENT full-profile readout, not in a
psychological-noise space. For the SAME third-block likelihood L_i and reset
next target reading mu_q^R, direct subtraction of the normalized numerators
gives

\[
\mu_q^F-\mu_q^R
=\frac{\sum_i r_i L_i(Y_{iq}-\mu_q^R)}{\sum_i\alpha_i L_i}.
\]

The future multiplication by L can make an invisible direction visible to
the target readout. No state term is silently discarded as topic/noise.
This is finite Bayes/operator algebra, not a new theorem. In particular,
if L_i is identical for every i within each profile, the two next FULL profile
readouts agree; a source-independent L is a stronger null. Both null worlds
pass synthetic tests. Different invisible memory alone is insufficient: it
must be associated with the subsequent observation's target-valued component.

For strictly positive L and all profile masses positive, closure of the full
profile update for EVERY component state with a given p would require that
within-profile constancy. To see the necessity, shift mass between two
components of one profile while leaving all other profiles unchanged. If their
likelihoods differ, that profile's unnormalized next mass changes while other
profiles' masses do not, so its normalized share changes. A five-marginal-only
readout has weaker constraints and can hide additional profile interactions.
HM211 tests the all32-profile interface, not only those five marginals.

## 4. Results, including negative prediction accounts

Five targets are E/N/A/C/O; the labels are inherited source high/low codes.
For each source fold, evaluation uses the opposite authors, never the same
source components. Present equality is algebraic; real third-block action is
descriptive under the fixed working law. Both signs/target directions stay.

| Target | Held RMS, bank0 | Held RMS, bank1 | Log gain bank0 | Log gain bank1 |
|---|---:|---:|---:|---:|
| E | .001083660 | .000534746 | -152.068 | -22.224 |
| N | .001605491 | .001079745 | +420.273 | -10.829 |
| A | .000699754 | .000820666 | -70.636 | +48.414 |
| C | .000750760 | .000478078 | -91.502 | -10.501 |
| O | .001487341 | .001006152 | +147.650 | +65.347 |

RMS is fine-minus-reset TARGET PROBABILITY, not r/AUC. Log gain is observed
held-author mean binary log score in1e-6 nats/author, not target probability.
O stays positive in both profile-balanced references (+72.358/+118.431);
N bank1 becomes -102.654. Joint32 observed gains +19.857/+989.644, but bank0's
balanced gain is -22.452 (bank1 +241.608). Better retention is not a universal
prediction improvement. All results are saved; there was no choice of scale,
target, reset strength, source weighting, or source-bank winner by these gains.

On the24 panel people, cross-bank same-sign counts are12/19/7/10/23 of24
for E/N/A/C/O, with A/C signed products negative. This is descriptive source
transport variation, not24 independent validations of each bank or a learned
universal response axis. No panel personality labels/native answers read.

## 5. Executable use and limit

The cold CLI serializes16 actual event hashes/addresses/conditions and source
weights, WITHOUT those past vectors, then resumes on the next8 vectors in a
fresh process. Two source banks ran once; cold discrepancy6.66e-16. Event IDs,
view/slot addresses, frame, step, and bank identity are checked. Aliases or
labels/model answers are rejected, not treated as extra evidence. Distinct IDs
alone do not establish statistical independence; that belongs to the working law.

All24 comments are used in three balanced4+4 blocks, not chronological sessions.
TOP20 venue bits remain provenance, not a complete condition response model.
The historical E5 encoder still supplies vectors. This is a numerical decoder
AFTER encoding, not raw-text LLM independence or recovered native KV memory.
Source-component memory may retain topic/community/author fingerprint. Current
data establish its effect under this family, not that all such memory has
psychological meaning. No new factors or human state changes are named here.

## 6. Prior art and next scientific decision

State sufficiency is prior art: [Littman/Sutton/Singh2001](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)
uses future tests; [Subramanian et al.](https://arxiv.org/html/2010.08843)
section2.3 P1/P2/P2a/P2b distinguishes present performance from recursively
predictable observations. HSE-PSR2013 is adjacent continuous-observation work
(abstract read), not an implemented kernel-Bayes inverse or consistency bound.
The local new construction is the actual-prefix, profile-preserving reset and
resumable addressed-vector API. It is DEVELOPMENT, not first discovery of
memory insufficiency, finite Bayes, or predictive state.

OpenAI/math pinned adc7f124 Posterior-replicas... section08-streaming1-200
was reread: condition on the current memory, THEN study a fresh block. Its
Gaussian side-matrix, bounded-prior, angular-success and memory lower-bound
hypotheses are not satisfied/claimed here. It guides experiment order, not a
transferred personality theorem or reason to chase numerical precision.

The useful next question is which invisible response memory participates in
SAME-event target-qualified reading, rather than generic source fingerprint.
First inventory HM191/195/199-203 actual event/target actions against this
new3-block family; their12/17-event inputs must not be called the same update.
Declare a new discriminating intervention and construction before new compute.
Do not search memory dimension, Gaussian scale, rank, block count or log-score
weights merely to increase a metric. All old scopes/results/seals remain open
or preserved; a new numerical interface is not full theory completion.
