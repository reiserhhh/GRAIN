# Observation policy and author evidence laws

2026-10-10. HM117 companion theory, separate from its frozen specification.
The finite collection and replay are complete; section7 records its actual
finding. The original18 protocols/100cards/44 questions remain open.

## 1. The distinction needed by a measurement foundation

An author supplies evidence; an instrument decides how to observe that evidence.
The distribution of an author's conditions, choices, events and expressions is
not the same object as the instrument's choice of query or observation focus.
Improving agreement with a teacher cannot separate those objects by itself.

Let e be a qualified evidence object. Qualification includes the attributed
person, episode, condition and status, not just the words or a vector mean. Let
P_u be author u's evidence law and N_q(e,r) the instrument observation for target
q at reading focus r. A declared observation policy nu gives

\[
 S_{u,q}^{\nu}=\sum_e P_u(e)\sum_r\nu(r\mid e,q)N_q(e,r).
\]

This is a standard conditional expectation, not new mathematics. Its purpose
is to expose three different dependencies: author evidence, target observation,
and instrument policy. Neither a Qwen score nor a designed evidence law is a
true human trait label.

The provenance of r matters. In HM117 the experimenter assigns the reading cue,
so it indexes the instrument. If a natural author chooses which person to focus
on, that choice instead belongs in e and P_u(e); it cannot be removed by calling
it a reading-policy artifact. The same linguistic cue can occupy different
measurement roles depending on who supplied it.

For a fixed policy, author differences are

\[
 S_{u,q}^{\nu}-S_{v,q}^{\nu}
 =\sum_e[P_u(e)-P_v(e)]\,\overline N_q^{\nu}(e).
\]

If the policy differs between authors, the comparison also contains

\[
 \sum_eP_v(e)[\overline N_q^{\nu_u}(e)-\overline N_q^{\nu_v}(e)].
\]

The second term can be nonzero when P_u=P_v. It is not automatically a
personality difference, nor automatically meaningless noise: the policy may
belong to a deliberately different measurement purpose. Comparability requires
declaring it rather than silently mixing it into the author score.

## 2. The finite question in HM117

HM116's current carrier always described Person A. HM117 changes only its focus
token to B, keeping behavioral evidence, source operation, token positions and
all A/B target questions fixed. Source A actions are transported unchanged;
there is no independently learned B action or assumed neural symmetry.

Let c,b,o,p,r in {-1,+1} denote condition, behavior polarity, acting person,
queried person and carrier focus. A=+1 and B=-1 for the last three signs. Let E
be the actual baseline-minus-edit native observation, in a separately specified
unit. After averaging b/o with their product, its complete finite expression is

\[
 L(c,p,r)=G_0(c)+pT_0(c)+rG_1(c)+prT_1(c).
\]

Here G is person-common response, T is queried-person contrast; the subscript
1 records focus dependence. All other response coefficients also remain saved.
This identity is a full factorial account on declared binary support, not a
low-rank approximation or proof of a unique neural circuit.

In the frozen character notation:

\[
 G_0(c)=\beta_{bo}+c\beta_{cbo},\quad
 T_0(c)=\beta_{bop}+c\beta_{cbop},
\]
\[
 G_1(c)=\beta_{bor}+c\beta_{cbor},\quad
 T_1(c)=\beta_{bopr}+c\beta_{cbopr}.
\]

## 3. Why another observation, not another precision target, is needed

With only r=+1, the observable queried-person contrast is T_A=T_0+T_1.
For any lambda, replacing T_0 by T_0+lambda and T_1 by T_1-lambda leaves that
observation unchanged. Perfect measurement of T_A cannot identify both terms.
The same ambiguity holds for G_0/G_1.

Adding the r=-1 observation gives T_B=T_0-T_1 and hence

\[
 T_0=(T_A+T_B)/2,\qquad T_1=(T_A-T_B)/2.
\]

Thus the experiment changes what is identifiable, not how accurately one old
sum is estimated. This is classical linear identification. The empirical
question is the magnitude, sign, target selectivity and condition dependence
of the separately identifiable terms in the actual fixed reader.

An even focus average removes G_1/T_1 from that particular instrument output by
definition. It does not show these terms are psychological noise or irrelevant
to interaction. They remain a second output, not discarded data. A canonical
single-focus instrument can also be well-defined without being focus invariant.

There are two observation objects to compare: the unedited native law and the
effect of the fixed source-A physical action. If the native person distinction
survives focus change but the edit does not, this is an action-transport problem,
not evidence that natural person judgements have the same focus dependence.
The source action may have a focus-specific coordinate orientation. HM117 does
not learn an alignment or a new B direction, so it cannot distinguish every
possible alignment explanation. If both objects change, they are still recorded
separately; a local physical intervention is not a natural behavioral change.

## 4. Connection to GRAIN's author composition and 4W account

For the positive designed laws P_+/-=(1+/-a*bo)/4, a=.4, with equal condition
mass and family mass1/6, the contribution of family f to the author contrast is

\[
 \Delta S_{p,q}^{f,r}
 =\frac{2a}{6}\mathbb E_c\widehat N_{bo}^{f}(c,p,r).
\]

This integral keeps the actual focus law. It gives a finite bridge from local
vector operations to author evidence composition, not a new human correlation.
Opportunity choice P_u(c) and conditional response P_u(b,o|c) stay separate;
neither is removed because an instrument has a common shift.

In the project's 4W interpretation: where/what/when belong to qualified evidence
and its availability; who includes both attribution and differences in author
composition. The observation purpose is an instrument index across those four
accounts, not a replacement personality factor or a new psychological label.

## 5. Literature alignment and mathematical borrowing

This extends rather than replaces the existing purpose-dependent qualification
in docs/GRAIN_EVENT_LOCAL_OBSERVATION_V1.md and joint relation laws in
docs/GRAIN_RELATION_INDEXED_OBSERVATION_THEORY_V1.md. Those already distinguish
report modes, event identity, target purpose and author composition. HM117's
new finite variable is a one-token externally assigned carrier focus, separated
from the queried person while reusing the same fixed source operation.

Park, Choe & Veitch (ICML2024), sections2--3, distinguish concept representation,
measurement and intervention and specify a causal inner product. HM117 neither
estimates their causal metric nor treats Euclidean similarity as identification.
https://arxiv.org/html/2311.03658v2

Feng, Russell & Steinhardt (2024/ICLR2025), sections3--4.2, separately represent
content and binding and identify a subspace using mixed Hessians and interchange
interventions. HM117's source difference is not that identified subspace; a
focus-dependent effect does not establish a uniquely named neural mechanism.
https://arxiv.org/html/2406.19501v1

OpenAI/math, fixed adc7f1241b42e322a6451854ab7e4b4c146bf78a, signed-tensor
03b-isotypic72--140 and206--244, retains isotypic multiplicities and extracts
specified coefficients by a group integral. This motivates retaining separately
named response sectors rather than optimizing one norm. The text experiment
does not meet or transplant its tensor-density, compact representation or
operator-angle hypotheses. Binary character sums are mature mathematics.
https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Signed-tensor-densities-and-diagram-budgets-for-coordinate-sweeps-September-26-2026/build/sections/03b-isotypic.tex

The contribution candidate is an executable evidence/person/focus observation
law and its consequences for constructing author comparisons. Generic binding,
factorial decomposition or a more accurate reader is not claimed as novel.

## 6. Scientific decision, not a rescue search

If T_0 dominates and the same matched-target contrast survives both focuses,
retain the tested person-qualified source operation. If G_1 or T_1 is large,
the reading policy must remain explicit. Mixed effects yield a conditional
operation, not a universal personality vector. Report all six families, both
conditions, all controls and all four units; do not select a winning target.

No matter the outcome, do not tune layer, strength, rank or query wording to
increase it. First use the finding to decide which observation object enters
GRAIN. A local single-reader result does not require abandoning earlier choice,
style, state or dynamic findings, which address different measurement objects.

Implementation: scripts/run_grain_observer_focus_hm117.py.
Canonical evidence: results/grain_math_response_v1/observer_focus_r1.
No API, private/MEPS reading, training, Claude, Git, automation or lockbox.

## 7. Executed result and its construction consequence

All244 new batches/2928 endpoints completed in635.276671 model seconds,
with0 retries and four early no-op cells passing. The240 inherited A-focus
batches remained unchanged. Full32-character/native-distribution replay and
direct corner sums passed; no neural model is loaded by the numeric readback.

The natural score-unit queried-person contrast has A/B cosine+.968295,
RMS .033749/.031587, with all12 matched family/condition corners positive in
each focus. Native log-odds shows+.976959 and the same matched signs.
The unchanged source-A directed action instead has score cosine-.860517,
RMS .000610/.000489; A is positive in12/12 matched corners, B in3/12.
Its log-odds cosine is-.548849 and B is positive in2/12 corners.

The native score T_0/T_1 RMS is .032408/.004251; for the directed effect it
is .000156/.000530. The action's common response remains highly aligned
(score cosine+.994098). These are separately specified finite quantities,
not human Pearson correlations, causal information fractions or trait units.
The wrong-family source action also reverses its person effect (cos-.900174).
Score selectivity summaries alone would obscure the sign change and do not
consistently agree with log-odds; retain both signed, conditional tables.

Consequently, natural person-qualified inference and fixed physical action
transport must remain separate. Do not enlarge the edit or refit a metric to
save the universal direction claim. The next constructive object is an indexed
response operation connected to qualified author evidence, not one raw vector.
The existing HM91/HM102/HM109/HM112 relation/composition/response/cold interfaces
must be checked before proposing that next finite package. No new collection is
started by this theoretical closeout. Parent HM116 remains a partial budget stop.
