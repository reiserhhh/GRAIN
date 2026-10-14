# HM75: ownership information routes versus output interactions

2026-10-09. Finite mechanism follow-up to HM74, fixed before acquisitions.
No accuracy optimization, training, API, private material, or parameter search.
Original 18 protocols / 100 cards / 44 questions remain open.

## Scientific decision

HM74 removed explicit query yg/xyg sectors, yet output xyg survived substantially.
Does that surviving ownership interaction depend on ownership variation in the
pre-query states, or can the retained contextualized suffix support it without
that variation? This determines whether an author-observation interface should
retain route-conditioned relational operations rather than identify an output
interaction with one same-named vector coefficient.

The question is about ownership variation, not all body information. A body
with ownership averaged still carries x/y content. Persistence cannot prove
that upper layers never reread body content, or locate a unique head/circuit.

## Complete finite scope

Reuse all HM74 288 public synthetic bodies / 66 query cubes / 528 captures,
including 30 directed behavior pairs with two predeclared queries each and six
neutral controls. These are known materials, not a fresh independent replication.
Reuse the 528 native scores and 528 query-remove-both scores without rerunning.
Six new arms on every corner plus one full-state noop on corner 0 of each cube:
3168 + 66 = 3234 actual forwards, at most 1800 collector seconds. Fixed Qwen
observer, float32 MPS eager, block16 output, unchanged text and queries.

At fixed x,y, the g-flipped input has exactly the same length/positions and
differs in two body name tokens. Pre-query means all positions before the
identical complete suffix. It includes the system prefix, which must already
be g-invariant by causal masking. No token interpolation or cross-x/y body
alignment is used. Natural x/y length variation is retained.

## Operations and exact mathematics

Let B(z), S(z) be pre-query and suffix states at z=(x,y,g). Define

    B_even(x,y,g) = [B(x,y,g) + B(x,y,-g)]/2
    S_even(x,y,g) = [S(x,y,g) + S(x,y,-g)]/2
    S_low = S - yg*c_yg[S] - xyg*c_xyg[S]

Group averaging on C2 is a standard Reynolds projection, not a new theorem.
Each mean keeps all g-even components and removes all g-odd components in that
region. The pre-query/suffix operations commute as state assignments because
they act on disjoint positions; their output effects need not be additive.

Six fixed arms:

1. body_even: (B_even,S).
2. suffix_even: (B,S_even).
3. both_even: (B_even,S_even).
4. body_even_suffix_remove_both: (B_even,S_low).
5. random_body: (B_random,S).
6. random_body_suffix_remove_both: (B_random,S_low).

B_random subtracts a per-token norm-matched random direction with g parity;
one direction per x/y pair, deterministic seed independent of outcomes. Both
members share the direction with opposite signs, preserving their pair mean.
The query-low assignments are exactly the HM74 sealed remove_both payloads.
All full payloads, token proofs, and source hashes seal before any new output.
No same-layer body-to-suffix recomputation is done; upper layers freely run.
These are propagating cut interventions, not edge/path isolation experiments.

Full cut under both_even is identical for paired g inputs; upper-layer position
indices and causal mask also match. Its g/yg/xg/xyg output coefficients must
therefore be zero to numerical replay tolerance. This is an algebraic positive
control, not a discovery that personality or information has vanished.

Measure all output coefficients, not just final bounded score. Score, high/low
log odds and centered ten-candidate logits retain distinct units. The four-cell
route account is

    I_route = F_native - F_body_even - F_suffix_even + F_both_even.

The key new account compares F_S_low with F_(B_even,S_low) and its random
control. It distinguishes functional ownership-route participation from the
tautological both-even zero. It does not distinguish every possible downstream
nonlinear construction or redundant circuit without additional experiments.

Synthetic author evidence profiles P_J,+/-=(1 +/- .5*chi_J)/8 retain matched
lower-order marginals as in HM74. For every arm, their reading difference is
c_J[F_arm]. Hence route participation has a concrete author-composition
measurement consequence without treating these profiles as human authors.

## Verification, stopping and scientific interpretation

Check actual payload assignment, invariant unpatched positions, native-cut
replay and the 66 noops. Capture outcomes once with an atomic attempted/in-flight
ledger, no uncertain retry. Any mismatch/error/cap stops and preserves completed
units. Use the same verified live handle, never restart on observation timeout.
Recompute payloads and finite coefficients by independent arithmetic with no
additional model calls. Tests and hashes do not establish psychological validity.

If body_even alone has little effect but suffix_even does, the suffix already
carries the operative ownership relation at this cut. If each alone retains
action while both destroys it, retain parallel/redundant route explanations.
If body_even further changes S_low's residual action beyond its random control,
retained body ownership participates in that specific configuration. Keep all
six targets, both queries, negative/neutral outcomes and x-conditional effects.
No averaging a counterexample away or searching another layer to rescue theory.

After the full finite result, stop this package. A follow-up must change the
author-observation construction decision, not merely refine a small scalar.
The full psychological/standalone/natural-event/B_u/dynamic scope stays open.

## Prior methods and higher-math inspiration

[Heimersheim/Nanda](https://arxiv.org/html/2404.15255v1) sections 2.2--2.4
distinguish propagation/path interventions and redundant versus serial routes.
[Feng/Steinhardt](https://arxiv.org/html/2310.17191v1) section 2 freezes other
context states to test storage; this is a different, freely propagating cut.
[de Wolf](https://homepages.cwi.nl/~rdewolf/publ/other/fourier.pdf) sections
2.1--2.2 provide the finite Boolean expansion used here.
OpenAI/math fixed adc7f1241b42e322a6451854ab7e4b4c146bf78a, metric/connection
06-matrix.tex lines 1--120, supplies fixed observation-family and joint matrix
information design inspiration. Ellipticity/harmonic jets/uniqueness assumptions
are not established here and its PDE theorem is not imported. New contribution
candidate is the evidence-route to author-composition interface, not group
averaging, activation intervention, or a newly invented mathematical theorem.
