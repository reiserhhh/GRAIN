# Passive Evidence, Narrative Emphasis and Author Measurement

2026-10-09. HM55 finite theory update after HM54. This connects existing GRAIN
history machinery to a measured passive observer, not a new deduplication theorem.

## Result That Changes the Next Decision

The native reader responds to both mention multiplicity and occasion identity.
Across 48 controlled contrasts, repeating the same focal occasion changes the
normalized digit score with RMS .137937 (43/48 toward the focal isolated score).
Changing only two occasion digits to identify distinct occasions adds RMS .056907
(39/48 toward the focal isolated score). Identical behavior words and token
counts therefore do not fix the readout. The two effects coexist; the first is
larger in this finite protocol, not a universal information percentage. Occasion
digits may also carry label/number effects; this does not uniquely identify an
independent-event counting mechanism.

Additional B actions change A's readout by RMS .126823 even though A's qualified
current events are unchanged. Repeated old claims before an explicit correction
change the corrected readout by RMS .116573. Presentation order alone changes it
by RMS .131098. These are observation-law responses, not measured author traits.

## Define the Different Objects

Let H be the full ordered account, with explicit record, event, actor, condition
and revision links. Let pi_u(H) be the current qualified event ledger for author
u. Let nu_u(H) retain that author's action mentions, including aliases and old
claims with their historical status. Keep the full narrative/order/context H;
neither pi nor nu is declared a sufficient statistic for the author.

For frozen isolated-event observations s_q(e), the experiment defines:

    B_{u,q}(H) = (1 / |pi_u(H)|) sum_{e in pi_u(H)} s_q(e)
    N_{u,q}(H) = (1 / |nu_u(H)|) sum_{m in nu_u(H)} s_q(m)
    L_{u,q}(H) = F_q(T_q(H))

B is a controlled event reference. N describes mention composition, not valid
current behavioral evidence. L is the actual observer's judgment. These are
three different objects, not three estimates of a known human personality truth.
Absent own evidence yields an unobserved B/N, not zero. The digit protocol still
forces L to return a number; that response alone does not create own evidence.

The event ledger preserves conditions, choice and source provenance. Repetition
can itself be a narrative behavior worth studying; deduplication applies only to
the question of independent behavior-event evidence. It must not erase the
author's emphasis, framing, corrections or choices from GRAIN.

## Finite Identities and What They Mean

With one focal event f and one companion event o, the event reference is
(f+o)/2. Three mentions of the same focal event leave it unchanged. Three distinct
focal events produce (3f+o)/4, a difference of (f-o)/4. The mention reference
changes by that amount for both alias and distinct histories.

These equalities are specified by event identities and arithmetic. The 240
checks agree within 8.327e-17. They do not discover event identities in natural
language, prove an optimal scoring rule, or estimate a human latent trait.

HM55 shows that L is not determined by this event profile alone. Nor is it
determined by the unordered pair (B,N): reversing report blocks preserves both
but changes L. Consequently, "faithfully compress the LLM" and "define a
psychological measurement" are distinct developmental tasks.

An existing, ordinary quotient/fiber argument makes this precise. If
C(H)=C(H') and g only observes C, then

    max(|g(C(H))-L(H)|, |g(C(H'))-L(H')|) >= |L(H)-L(H')|/2.

For squared error on equally weighted order pairs, the minimum pair RMS is
|L(H)-L(H')|/2. The 192 order pairs imply a .065549 aggregate RMS lower bound for
any function of only the order-invariant profiles. This is a lower bound on
reproducing this reader's outputs, not on personality prediction. It reuses
standard mathematics already present in GRAIN, not a new theorem.

## Vector Geometry Is Not Yet the Readout Law

At block16's query endpoint, define

    d_mention = h(alias) - h(balanced)
    d_occasion = h(distinct) - h(alias).

Their Euclidean cosine is negative in 47/48 cases (mean -.358900), even though
both additions frequently move the digit score toward the focal behavior.
Thus "same output tendency" cannot simply be read as "same Euclidean vector
direction." The full decoder and surrounding states mediate the action.

These are descriptive endpoint differences, not a localized causal circuit.
Alias versus distinct is length/token matched. Alias versus balanced adds text
and changes sequence positions; the geometric statistic includes those effects.
Do not name either displacement a personality axis or transfer a norm between
observers without a calibrated observation metric.

## Development, Not Fidelity Rescue

The useful GRAIN development is a qualified event channel plus a retained
narrative/context channel, followed by a declared target-specific observation
law. Native LLM judgments can inspect their effects, but are not the gold
standard to be copied indiscriminately.

Next mechanism question: which part of the narrative effect is carried by
behavior content, qualification/context, and query-conditioned processing, and
which part remains an event-specific contribution when their composition is
controlled? Reuse this frozen bank and full states; do not start an alpha,
head/layer, rank or teacher-RMSE grid. Before further calls, identify the
observation unit and the GRAIN decision the intervention would change.

Scientific contribution candidates remain the empirical composition law and
the measurement interface across evidence types. Generic repetition effects,
record deduplication, sum/mean pooling, factorization and patching have prior art.
Natural author parameters, psychological meaning, independent LLM-free scoring
and the original18/100/44 scope remain open; this packet does not shrink them.

## Prior Art and Reading Boundary

- Song et al., Mergeable Model-Side Aggregation States (2026),
  https://arxiv.org/html/2607.26448v1 : normalized attention can change under a
  repeated record, and mergeability does not imply duplicate-idempotence.
  The paper uses established HLL/Count-Min operators. We use no such sketch.
- Samanta et al., BayesBench (2026),
  https://arxiv.org/html/2606.30850v1 : latent inference and downstream judgment
  are distinct evaluations; evidence trajectory matters. We do not reproduce
  its environments or infer an exact Bayesian posterior from these digit scores.
- OpenAI/math fixed commit adc7f1241b42e322a6451854ab7e4b4c146bf78a,
  Finite Monoid Computations, algebra.tex (215 lines) and compilation.tex
  (292 lines) read in full. Inspiration: distinguish deterministic episode
  updates from their final output tests and require compatible composition.
  The finite-state/finite-monoid hypotheses and the other proof sections have
  not been established for our observer. No generalized star-height bound is
  applied to GRAIN, and the manuscript's main theorem was not independently
  verified. This is a design analogy, not a mathematical breakthrough claim.

## Reproduce

Run scripts/run_grain_math_passive_event_composition.py with stages prepare,
collect, analyze and scripts/verify_grain_math_passive_event_composition.py.
Existing canonical output: results/grain_math_response_v1/passive_event_composition_r1.
Preserve r1; prepare rejects an existing output. 408 actual forwards in127.863
seconds; 55 focused tests including20 new tests, Ruff lint/format pass.
Alternate scalar/ledger error3.553e-15, zero extra forwards during verification.
Same-agent recomputation is arithmetic verification, not independent replication.
