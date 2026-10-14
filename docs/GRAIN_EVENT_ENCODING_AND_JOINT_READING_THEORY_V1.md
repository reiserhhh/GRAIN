# Event encoding and joint reading: source-qualified construction V1

HM271, 2026-10-14 JST. Mechanism alternatives and mathematical construction;
the one frozen experiment is now complete and verified. No accuracy fitting.

## Two Kinds of Relation Must Not Be Confused

HM270's source relation is statistical: two event observations share one source
author index rather than independently selecting authors within a five-bit
profile. HM271's relation is computational: one evidence event may alter the
encoding of a later event. Statistical dependence does not require such a
computational edge; a later reader can infer a shared cause from two separately
encoded observations. Conversely, a computational edge can carry ordinary
coherence without any person-related meaning.

Thus these operations are not assumed equivalent:

\[
\text{reset source-author coupling}\ne
\text{cut cross-event neural encoding}.
\]

The new empirical question is whether the same source-linked behavioral action
can be constructed without that neural encoding edge. This connection is not
answered by finding a large interaction norm or a better predictor.

## Actual Independent Encoding Contract

Let P be the fixed header, X and Z the two complete evidence token blocks,
and S the fixed seam. Prefix positions remain the original positions. At every
decoder layer, the isolated construction has dependencies

\[
H_X^{\ell+1}=T_\ell(H_X^{\le\ell};H_P^{\le\ell}),\qquad
H_{S,Z}^{\ell+1}=T_\ell(H_{S,Z}^{\le\ell};H_P^{\le\ell}),
\]

with causal masking within each block. The seam belongs to the later bank;
otherwise it would relay first-event content. There is no stochastic claim
that the real texts X and Z become independent. Their authors, choices,
opportunities and positions remain in the construction.

The later reader is deliberately NOT additive or independent:

\[
Q^{\ell+1}=T_\ell(Q^{\le\ell};H_P,H_X,H_{S,Z}),\qquad
\ell_j=\sum_t\{W_{j_t}^{\mathsf T}Q_t+b_{j_t}-\log Z_t\}.
\]

The suffix and candidate tokens can read both complete banks recursively at
every layer. A person-related configuration can therefore emerge at reading,
even when neither event encoder has seen the other. This is a constructive
alternative to requiring a single jointly contextualized text representation.
It is still a neural reader, not a distilled mathematical personality decoder.

Both the early header and first-event caches are unchanged by the encoding
operation. Changing the first event at equal length leaves all later isolated
prefix caches unchanged by causal induction. These are instrument invariants,
not new psychological discoveries. Absolute positions qualify each bank: this
experiment does not create a universally position-independent event cache.

## Target-Indexed Observation and Exact Signed Accounts

All50 fixed public first-person behavioral candidates retain their whole
sentence, EOS, selected-token contribution and full-vocabulary normalizer.
Balanced high/low support B, common support U and equal-item key Q are retained.
No renormalization over the nonexhaustive candidate bank is introduced.

For the four actual own/foreign joins, let H^a_r be the mixed difference of
the fixed behavioral observation under native or isolated encoding. Then

\[
H^N_r=H^I_r+\Delta H^{\mathrm{enc}}_r,
\qquad\Delta H^{\mathrm{enc}}_r=H^N_r-H^I_r.
\]

This is an exact signed difference for the declared attention-support edit,
not a unique fraction of causal personality information. Softmax normalization
and all later recurrence are part of the response to the edit. Positive and
negative contributions can oppose or cancel.

For any HM270 source mixed action M_{cq}, including linked J,
within-profile product I, and linkage correction D=J-I, retain

\[
C^a_{cq,r}=\mathbb E_{\text{registered edges}}[M_{cq}H^a_r],
\qquad C^N=C^I+C^{\mathrm{enc}}.
\]

These finite unnormalized contacts locate an operational correspondence on
identical evidence. They do not assert that Qwen uses the source Gaussian
formula, that the observation is a true human behavior, or that the graph cut
implements a target-independent degradation of an observation experiment.

## Competing Mechanisms and Decision

A necessary-cross-encoding account requires the source-linked contact to
disappear in the isolated construction. A later-assembly account permits it
to remain. Opposing isolated and encoding contributions require both signed
terms, not a winner chosen by norm. All five targets, both block orders, all
four source contracts, common/balanced observations and the inherited profile
shuffle are retained. Order reversal is a serialization comparison, not a
person's psychological change over time.

The result can make the next scientific decision more specific: investigate
an independently encoded evidence bank with a joint reader, or investigate
the qualified cross-event encoding mechanism. More head/layer/mask/donor/rank
grids would merely refine a localization unless a new competing prediction
is explicitly stated first.

## Prior Art and What This Adds

HM111 already implemented an independent event bank and a recursive collector
on designed evidence; HM156 already separated a carrier's encoding and reader
interactions; HM259 already intervened on real-text continuation access.
None is renamed as a new theorem. HM271 adds their missing same-event link to
HM270's marginal-matched source-author relation and HM263's frozen natural-text
behavioral observation.

[Causal Abstraction (2025), sections2.4-2.5](https://arxiv.org/html/2301.04709v4)
provides the intervention/abstraction distinction. No high/low alignment has
been identified merely by taking contacts.
[Xie etal (ICLR2022), sections2-3](https://arxiv.org/html/2111.02080v3) derives
latent-concept inference in a mixture-HMM setting; its delimiter and
well-specification assumptions are not asserted for PANDORA or Qwen. Its
lesson is to separate the observation family from the shared-concept reader,
not to import an asymptotic accuracy guarantee.

[Pinned OpenAI/math foundations166-232](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
requires a declared conditional-independence experiment for product composition
and an input-independent kernel for degradation. Here a graph cut establishes
neural dependency separation, neither assumption about the human-text law.
This is design inspiration, not transfer of its reconstruction theorem.

## Finite Result and Theoretical Decision

All192 prefixes/9600 candidate sequences were observed once,1344 batches,
501.730511 model seconds under the one1200-second cap. The cut had exactly
zero closed-edge weights at all5376 layer receipts. Header/first-event cache
change, later-bank response to the equal-length first-event edit and prefix
mutation were all zero. Native readback error was4.468e-5; no precision rescue.

In whole-sentence balanced units, actual source-linkage D contacts with the
isolated reader were positive for N in both orders of all four contracts.
These are eight descriptions of ONE dependent24-edge panel, not independent
replications. E5/pooled, in units multiplied by10^6 for readability:

| Order | Native N | Isolated N | Encoding increment N |
|---|---:|---:|---:|
| 0 | +240.210 | +300.838 | -60.628 |
| 1 | +23.670 | +120.874 | -97.203 |

This operational contact does not require cross-event prefix encoding. A
later joint reader can retain it from separately encoded evidence banks.
The encoding increment can oppose it; larger native magnitude is not the
criterion. E and O isolated contacts were also positive across all four
contracts/both orders, while C stayed negative and A varied. E5/pooled E
changed from isolated +28.512/+81.132 to native -115.159/-126.015. Do not call
the removed increment psychological noise or prefer the isolated arm as a
better personality predictor. No candidate outcome was used for selection.

All fixed-null and common-support contacts remain. For example, E5/pooled
null isolated N is -203.441/+66.952 in the same units. Nonzero contact alone
does not supply specificity or significance. The result identifies a
computational-stage distinction, not a human trait or a complete high/low
causal abstraction. Same-agent verification, not independent replication.

All14400 cross-target contacts were separately re-summed by direct dot
products (maximum error4.337e-19); signed contact reconstruction2.711e-19.
28 focused tests, three Python lint/format checks and registered seals pass.
Collection/analyze/verify/check are terminal; no experimental process remains.

The next question is the SOURCE-supported rule of joint target valuation:
what evidence relation is used by that reader, and how does it become a
target-specific comparison? Existing HM111/156 already establish generic
independent-bank/reader capability. Repeating that general observation or
adding head/layer/mask/precision grids is not the next contribution. Inventory
existing qualification, source-law and cold-operator interfaces before a new
finite plan; require different intervention predictions or a new reusable
measurement construction. Numerical precision is not the present bottleneck.

Canonical finite experiment: `results/grain_math_response_v1/join_encoding_r1`.
Original18/100/44 and the full mainline goal remain OPEN.
