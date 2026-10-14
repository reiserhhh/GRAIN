# Addressed Query Carriers and Conditional Action

This is the mathematical object used in HM33, written during its source stage.
It does not report unfinished results or replace frozen protocols.

## From Text Positions to an Action Law

At one fixed layer and target query, partition the actual decoder state into:

    H: four addressed property-token states,
    P: query-suffix states before the endpoint,
    Q: the endpoint state,
    B: all remaining text/system states together with receiver layout.

P consists of context-conditioned hidden states at token-identical query
positions. It is not a pure query-word vector: preceding text has already
affected it. B is not pure topic, and a donor swap does not produce a naturally
occurring independent human report. These are controlled decoder interventions.

The quantities of interest are distinct:

    F_q(H,B,P,Q): native conditional digit expectation,
    alpha_q(H,B,P,Q)=d_H F_q(H,B,P,Q): property sensitivity,
    A_q(H,B,P,Q;delta)
      =[F_q(H+delta,B,P,Q)-F_q(H-delta,B,P,Q)]/2.

The exact rounded float32 half-delta is shared across B/P/Q changes. A large
native-level difference need not imply a large change in property sensitivity.

## Experimental Index Algebra

The16 addresses have indices s=(i,j,l,k) in {0,1}^4 and characters

    chi_m(s)=(-1)^(m dot s),
    c_m=16^-1 sum_s chi_m(s) A_s,
    A_s=sum_m c_m chi_m(s).

Orthogonality and Parseval follow by a product of four two-point sums. They
apply to the experimental design measure, not an assumed independence law of
natural text or a physical group action on its embeddings. All16 terms survive
in the analysis; high-order interactions are not automatically noise.

The old P=B restriction makes chi_m equal to chi_(m XOR0110). For each old
three-bit mask n, embed its body bit as B and obtain

    d_n=c_embed(n)+c_(embed(n) XOR0110).

An old effect can therefore be amplified or cancelled. Repeating the old
diagonal cannot distinguish the two terms. The new addresses identify them;
the inherited diagonal and every old coefficient must still replay exactly.

## An Information Obstruction, Not a Fitted Accuracy Score

Let Z be the retained experimental indices. By completing the square,

    E||A-f(Z)||^2
      =E||A-E[A|Z]||^2+E||E[A|Z]-f(Z)||^2.

Consequently the first term is a lower bound for any law using only Z.
Retaining H/P/Q leaves all B-bearing character energies; retaining H/B/Q
leaves all P-bearing energies; retaining only H/Q leaves their union. This is
an optimistic per-cell oracle: actual hidden-state collisions or new-context
transport can only make a learned law harder, not easier. No application
oracle mean is supplied as a deployable predictor.

Use the same full16-cell measure when comparing these three bounds. Bounds
on the old diagonal have a different measure and contain cross terms; changing
experimental support is not an accuracy improvement.

## A Local Differential-Geometric Interpretation

Within a fixed token layout, let z=(B,P,Q) denote the remaining continuous
states. In the smooth real-valued decoder idealization, the partial one-form

    alpha=sum_a (partial F / partial H_a) dH_a

is a gradient when z is fixed. On the full (H,z) space its exterior derivative
also contains mixed terms:

    d alpha=sum_(a,b) (partial^2 F / partial z_b partial H_a)
                        dz_b wedge dH_a.

The H-H terms cancel by symmetry of mixed derivatives. Thus a partial
observation can fail to be closed on the enlarged space even though the full
dF is exact. This locates a specific mechanism: omitted conditioning variables
modulate the evidence covector. It is not evidence for a nonconservative
human personality force or quantum measurement.

HM33 measures finite changes of this partial gradient, not Hessian values.
It does not estimate a continuous B derivative across receivers with different
token layouts; those have separately addressed charts. Exact finite actions
remain valid without a Taylor approximation, and float32 implementations
retain their recorded rounding errors. A later local derivative compilation
must keep these distinctions.

## What OpenAI/math Contributes

Fixed manuscript source:
https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Polynomial-Hitting-Lists-for-Noncommutative-Rational-Formulas-September-24-2026/build/sections/encoding.tex

The entire240-line encoding section was read for this packet. Its affine-pencil
recursions preserve ordered multiplication and every inverse-node domain
requirement, even in a branch later multiplied by zero. The two nested-inverse
example has a simplified value defined at a point where the original program
is not. The lesson used here is to retain the addressed computation and its
conditions rather than erase a carrier because its observed diagonal effect
cancels. This is methodological inspiration, not a claim that a softmax
decoder is a rational formula or that the manuscript proves psychological
validity. The manuscript's later generator/rank/division proofs are not claimed
as read or as applicable.

The finite character and conditional-mean identities above have direct,
independently implemented proofs; they do not depend on a new unpublished
theorem. Discovery concerns which actual decoder carriers participate.

Another fixed manuscript was read in full for its introduction and WL/game
definitions,160 and219 lines respectively:
https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-complexity-of-identifying-a-graph-by-Weisfeiler-Leman-refinement-September-25-2026/build/sections/02-wl.tex

Its joint replacement-color multiset is not a collection of independently
counted coordinate marginals. This is a useful warning for our compiler:
separate H/P/Q summaries can destroy which states belonged together. Its
distinction between identifying one specified pair and all possible comparison
graphs also cautions against a universal claim from this finite bank. We have
not implemented WL or verified the later hardness reduction, and do not assume
a typed directed text graph satisfies that manuscript's simple-undirected
graph conditions. Only these explicit design lessons enter the continuation.

## Connection to the Measurement Mainline

A condition changes the observation law, not just its mean. GRAIN should
therefore keep evidence identity, expression content and condition-dependent
reading separate, then aggregate eligible author evidence without counting
several internal views as independent events. P and Q are reader-side carriers
of the same text, not new author observations or automatically named traits.

Once the carrier is localized, the next useful question is whether a
source-only mathematical action law can preserve it on cold contexts, with
the same target units. Unbounded subdivision of the state or a larger accuracy
grid would not answer that question. Original natural-author, response-operator
and psychological-meaning requirements remain open.
