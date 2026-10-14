# Local evidence and one compatible author-observation law

2026-10-13 JST, HM238. Constructed and tested on the inherited numeric PANDORA
events. Original18/100/44 remain OPEN. The scorer is not promoted for use.

## 1. The missing operation now implemented

HM235-237 assigned target readings to local overlapping triples and averaged
them. HM174/191 composed events under a different empirical-author iid law.
Neither operation defined a consistent joint law for this positioned sequence.
HM238 implements that distinct construction with no LLM readout optimization.

For SOURCE profile G and record states X_0..X_17 let

\[
F_t(g,a,b,c)=P_{emp}(G=g,X_t=a,X_{t+1}=b,X_{t+2}=c),
\quad B_t(g,a,b)=\sum_cF_t(g,a,b,c).
\]

Each of386 source authors contributes1/386 and each of their two18-record
windows half of that mass. Every t is a POSITION in this sampling protocol,
not clock time or a true conversational response. The shared G is the full
five-bit measured profile, not an inferred personality class count.
The consistent separator is B_(t+1)(g,b,c)=sum_aF_t(g,a,b,c).

One compatible working joint law is

\[
Q(g,x_{0:17})=B_0(g,x_0,x_1)
 \prod_{t=0}^{15}\frac{F_t(g,x_t,x_{t+1},x_{t+2})}
 {B_t(g,x_t,x_{t+1})}.
\]

On zero separator support the transition is unadmitted; a path reaching that
row carries zero mass, not an invented neutral transition. Consistent cliques
ensure these rows have zero incoming mass in the unconditioned law.
Q has every original clique margin and the same profile prior. Each overlapping
record has ONE position; its shared pair appears in a separator rather than
being counted repeatedly as independent evidence.

The graph has cliques {G,X_t,X_(t+1),X_(t+2)} and satisfies running intersection.
Evans Chapter7 Proposition7.2 supplies exactly this mature factorization:
https://www.stats.ox.ac.uk/~evans/gms/_book/jt.html
This is not a new graph theorem or an identified natural order-two human law.

## 2. What maximum entropy does and does not supply

For any compatible P with these margins,

\[
H(P)\le H(G,X_0,X_1)+\sum_tH(X_{t+2}|G,X_t,X_{t+1})=H(Q).
\]

The inequality is the entropy chain rule plus conditioning reducing entropy;
equality holds for the declared Markov completion. Also
KL(P||Q)=H(Q)-H(P) on compatible support. These are standard finite facts.
They justify a specified minimally dependent completion, not the claim that
the author's real expression uses only two-step memory.

For the complete empirical SOURCE catalogue, H(P)=6.648985nats, H(Q)=11.786882,
KL=5.137897. The old label permutation gives11.654831 and5.005847. This cost
includes finite-catalogue structure; it is NOT a psychological information
fraction, a discarded noise quantity, or evidence of a new factor.

## 3. Masking is a marginal, not a shortened text

For observed positions A with their identified values e_A,

\[
P_Q(G|e_A)\propto\sum_{x:x_A=e_A}Q(G,x),\qquad
\mu_q(e_A)=\sum_g g_qP_Q(g|e_A).
\]

The implementation contracts pair-state messages over every original position.
A removed record is unknown, not zero, and does not create a new adjacency.
Native twelve-event records occupy6..17 in an18-record observation contract;
their first6 positions remain unobserved. Cold records do not contain removed
values or application labels. Exact identity aliases do not add evidence.

This is an observation-model marginal, NOT proof that missing publication is
noninformative. A different selection law would require a different model.
The neural E5-derived eight states and historical source reference remain.

## 4. An assembly error distinguishable without prediction accuracy

The registered comparison multiplies conditional clique likelihoods without
separator divisions. Its raw product changes the empty-evidence profile prior:
TV=.959750(actual),.990969(old permutation). Actual target shifts E/N/A/C/O are
+.508821/+.527027/+.429686/-.427085/-.487122 despite ZERO observed evidence.
The comparator is not a claim that previous GRAIN packets used this product.

Normalizing its full joint law within each profile restores the prior, but its
mean positioned-clique TV discrepancy from SOURCE is.530320/.549411. Thus
probability normalization alone does not preserve the observation law used to
give a text fragment target meaning. Corrected-chain margin error2.60e-18 is
BY CONSTRUCTION, not an empirical personality validation.

## 5. The decisive transport boundary

The real whole-history scorer has nearly no supported application histories:

| Law/arm | Held18-record windows | Native12-record histories |
|---|---:|---:|
| Actual/full |1/240|0/24|
| Actual/support removal |7/240|5/24|
| Actual/counter removal |2/240|2/24|
| Old permutation/full |0/240|0/24|
| Old permutation/support removal |3/240|2/24|
| Old permutation/counter removal |4/240|2/24|

Both proper candidates have the SAME hard support; their unsupported cases
remain missing. Agreement on the one supported held full case is not useful
transport evidence. No native correlations, AUC, r or application truth used.

A separately sealed POST-OUTCOME descriptor removes G solely to examine path
support. The pooled unlabelled chain supports60/240 full held and8/24 native.
Therefore BOTH bottlenecks exist: many positioned paths are unseen even pooled,
and59 held+8 native paths exist pooled but cannot be qualified by one fixed
profile through the complete sequence. Pooling restores no target meaning and
is not promoted as a personality scorer.

For a fully observed path the support distinction can be written exactly:

\[
\exists g\;\forall t:F_t(g,x_t,x_{t+1},x_{t+2})>0
\quad\hbox{versus}\quad
\forall t\;\exists g:F_t(g,x_t,x_{t+1},x_{t+2})>0.
\]

These quantifiers do not commute. The latter permits different SOURCE profiles
to support different fragments; it does not identify an author profile that
supports their conjunction. This is standard logic with an observed GRAIN
consequence, NOT a new theorem or evidence that unseen real paths are impossible.
SOURCE sampling zeros were hard zeros of this explicit empirical completion.
They must not be equated with human impossibility.

Among240 actual held windows,164 first lose support by original position5:
3 at2,40 at3,70 at4,51 at5. Failure is not exclusively at the final long suffix.
The old permutation has similar failures, consistent with a generic sparse
observation-law issue, not a specifically psychological effect.

## 6. Mainline decision

Local target associations plus exact arithmetic do NOT supply a transportable
whole-author observation law automatically. For this candidate, stop the
whole-profile empirical path-density branch. Do not smooth, shorten windows,
change profile count/order/bins or fit native answers just to make it return
scores. Keep existing useful local/relational/choice/state objects.

Next mechanism inventory: HM184/185/192/209/211 source-supported target
functionals and relational updates that do not require reconstructing a sparse
global density. The requirement is to keep target assignment AND a common
observation support in the SAME event interface. A pooled law that erases G
does not satisfy it. State a reusable construction and competing mechanisms
before a new finite experiment; no prediction/fidelity precision grid.

OpenAI/math adc7f124 Gaussian-fiber06 lines1-110 was read for the observation-
qualified likelihood lesson; sphere/Gaussian bounds do not transfer here.
No new higher-mathematics theorem is claimed.

## 7. Reproduction and evidence

Canonical results/grain_math_response_v1/overlap_chain_r1, with separately
sealed support_description_r1.386SOURCE;240held+24native histories overlap
4 authors/48publications,140distinct application authors.793cold cases include
792 history/arm records andONE empty control; not793 participants.
Separate cold process consumes only aggregate bundle and numeric observations.
Upstream encoding stays neural; no new forward/API/private/train/labels.

Prepare/compile/cold/analyze/verify/check each once exit0. Source compilation
.027208s,cold CLI.807642s. Literal source cliques8.67e-19;60supported case-law-
mode accounts and3112missing retained;posterior error3.33e-16,logmass1.42e-14.
Pooled descriptor alternate error1.07e-14. Same-agent audit, not blind replication.
8new/46focused tests and3primary Python lint/format checks PASS. Descriptor
retains a sealed F401 unused-import warning; no scientific edit or rerun to
hide it. These checks establish computation fidelity, not theory completion.

Read-only primary check:
`python scripts/run_grain_overlap_chain_hm238.py check`
Do not overwrite or rerun the immutable scientific stages.
