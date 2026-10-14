# Native evidence access and a target-conditioned observation operator

HM131, 2026-10-11. Mathematical definitions and architectural consequences;
empirical findings come from completed `native_access_r2`, not these equations.
The preserved r1 false whole-mask guard failure and scoped recovery are documented
in `GRAIN_NATIVE_ACCESS_HM131_RECOVERY_NOTE.md`; the input/target design is unchanged.
This is a mechanism-study extension, not a completed personality instrument.

## 1. Separate evidence encoding from target-conditioned reading

For body X, define its layered contextual carrier bank

\[
\mathcal E(X)=\{K_g^{\ell}(X),V_g^{\ell}(X):
g\in\{S,H,B,C\},\ \ell=0,\ldots,L-1\}.
\]

This is a collection of actual body K/V states across decoder blocks, not a
single sentence embedding, an inferred latent personality or a fitted proxy.
S precedes historical evidence, H contains that evidence and its ownership
declaration, B is a later relay/header, C is the current action phrase. These
are addressed intervention groups; they are not psychological constructs.

Let q be a fixed complete question suffix and T the accessible subset of H/B/C.
The suffix computation obeys a recursive operator

\[
p_{\ell+1}=\Psi_{\ell,T}(p_\ell,q;\mathcal E(X)),\qquad
N_{q,T}(X)=R(p_L).
\]

Psi contains native Q/K/V attention, residual connections, normalization and
MLP; R is the unchanged final normalization/unembedding and declared numeric
readout. Different targets change the suffix and therefore its intermediate
queries, not merely a final scalar coefficient. The operator remains pretrained
neural arithmetic: writing it mathematically does not compile it away.

## 2. Dependency proof and what a restricted route can establish

Assume the actual inspected decoder has no cross-token operation except attention,
and attention is causal. If only suffix attention masks change, all body states
remain unchanged by induction: embeddings/positions are identical; each body
attention row has identical accessible prior body states; tokenwise operations
then preserve equality at the next block. No future suffix state can return to X.
The experiment checks current K/V for every question slot and final body states
for slot0; structural induction covers the other unchanged body rows.

If T is empty at EVERY attention block and EVERY suffix position, only S and
causally earlier suffix states can influence p. Fixed S/q/positions imply

\[
N_{q,\varnothing}(X)=N_{q,\varnothing}(X')
\]

for the aligned support. This is a history-independence control, not evidence
that the question-only score is a person's true baseline or that S is noise.
An endpoint-only cut does not establish it: H can write into an early suffix
token, which the endpoint reads at a later block.

A nonzero behavior/owner contrast under B-only or C-only access proves a
history-dependent intermediate carrier affects the native question program,
because those carrier input IDs are fixed and direct H access is forbidden.
H-only measures another actual access program. Neither alone identifies a
unique semantic circuit, an information percentage or a minimal sufficient
statistic for all natural unmodified inputs.

## 3. Preserve qualified responses, common responses and interactions

For content sign b, acting person o and queried person p in {-1,+1}, retain

\[
\beta_{J,q,T}=2^{-3}\sum_{b,o,p}\prod_{j\in J}j\;N_{q,T}(b,o,p).
\]

The bop sector distinguishes evidence ownership relative to the queried person;
bo is shared across queried persons. A large overall score change can occur with
little bop action. This separation avoids equating generic judgment movement
with a person-qualified effect. All eight role sectors and all target/family
combinations are retained, including common, off-target and reversed effects.

Access combinations have the exact finite expansion

\[
M_A=\sum_{T\subseteq A}(-1)^{|A|-|T|}N_T,
\qquad N_T=\sum_{A\subseteq T}M_A.
\]

Signed components may cancel. Attention renormalization and changed suffix
states belong to these access interventions: they are not held fixed causal
mediation shares. Mobius inversion is established mathematics, not a discovery.

## 4. Connect the observation to an author evidence account

For a positive specified evidence law P_u on complete body observations e,

\[
S_{u,q,T}=\int N_{q,T}(e)\,dP_u(e).
\]

In this controlled battery, P+/-=(1+/-a bo)/4 with a=.4 gives identical content
and ownership marginals, but different joint organization. At fixed queried
person p, the exact account contrast is

\[
S_{+,q,T,p}-S_{-,q,T,p}
=2a(\beta_{bo,q,T}+p\beta_{bop,q,T}).
\]

These are designed numerical accounts, not new human participants. e includes
its contextual history; the equation does not assume natural fragments are
independent, nor that reading a concatenated history equals averaging isolated
fragment scores. GRAIN choice/condition/state accounts must retain those
observation-unit distinctions when this operator is integrated.

## 5. Construction consequence, not a performance target

If a current-state operation F and native N differ, the candidate explanation
is not only a poor fitted coefficient: their evidence access and recursive
target reading may differ. HM131 measures that explanation without changing F,
fitting N or deleting body information. The actual results decide whether the
next reusable operator needs H, B/C or their joint access.

The finite access experiment cannot conclude that C lacks all relevant
information merely because the native C-only program changes its output.
Its narrower causal conclusion concerns how this fixed model uses the specified
carrier access. An information-sufficiency claim would require an additional
identification argument, not a higher prediction score.

## 6. Complete finite result and the next construction question

All eight carrier-access fields were collected for both wording banks. In the
person-qualified bop sector, H-only has score-field norm ratios 1.302748/1.615675
to full HBC, and log-odds ratios 1.205864/1.412113. B-only and C-only are nonzero,
despite their fixed input IDs and the absence of suffix-to-H access. BC-only
score ratios are .396059/.413319, with cosines .663248/.295119 to full access;
their log-odds ratios are .442504/.353666. Thus native history reading is strong,
but intermediate contextual carriers also have actual qualified action.

The finite access interaction HB projects negatively onto the full qualified
score field (-.189855/-.423373); HC does too (-.287585/-.349624). Both negative
signs remain in log odds. BC projects positively in both units/banks, whereas
the HBC component is not sign-invariant across banks/units. Complete role and
6x6 target/family fields, including reversed and off-target cells, are retained.
These are signed contrasts of different recursive access programs, not mutually
exclusive causal shares or percentages of personality information.

The resulting construction question is how to retain a qualified contextual
evidence bank and a target-conditioned reading operation with their interactions.
It is not how to tune one current-state scalar until it imitates full native N.
The body-state/current-KV/no-op/all-off checks are exactly zero; independent
readout error is 3.77e-15 and composition error 8.89e-16. Those checks validate
the finite intervention, not psychological validity or a new mathematical theorem.

## Sources

Heimersheim & Nanda2024 section2.2 and backup/cancellation discussion:
https://arxiv.org/html/2404.15255v1 . Known path-mediation distinctions motivate
the competing explanations; access masking is not the same intervention.
Elhage et al.2021 Q/K/V composition, primary indexed excerpts:
https://transformer-circuits.pub/2021/framework/index.html . No exact
attention-only path expansion is asserted for this attention/MLP decoder.
OpenAI/math adc7f124 signed-tensor 03b-isotypic72--133: named sectors and complete
multiplicity accounting are a design inspiration only; none of its density
domination assumptions/conclusions are imported into personality measurement.
