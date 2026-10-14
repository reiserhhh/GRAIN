# HM33: Earlier Query Versus Text-Side Context

Status: planned, not acquired. Predecessor HM32 complete, including alternate
verification_r2 and diagonal alias accounting. This is a bounded carrier
localization, not an accuracy/rank/prompt search or a new personality scale.

## Scientific Question

HM32 R modulation dominates endpoint-Q modulation. R still contains earlier
query states, so it cannot be called a text-side carrier. Separate:

- H: same four post-layer16 property states.
- Q: same post-layer16 last token.
- P: all earlier query-suffix states, question_start through last-1.
- B: every other text/system state, including receiver layout.

    F(H_i,B_j,P_l,Q_k;q), alpha=d_HF.

P includes fixed scoring instructions/assistant-prefix tokens, not just lexical
trait words. Its token sequence must match exactly within each pair/query.
The token that ends the suffix remains Q, never duplicated in P.

## Design and Resource Bound

Retain all4pairs/4wordings/3queries and source/validation/application roles.
Full2^4design means48sixteen-corner units. Inherit HM32 corners where Pside=Bside
(384references/1536endpoints) without replay. Missing P states need dedicated
captures of all96natural HM30 reference inputs:48source/24validation/24application,
96forward-only calls, no new backward. These captures collect new state arrays,
not new independent materials; replayed natural readouts must match originals.

New off-diagonal P/B corners:384references/backwards plus1536endpoints.
Total newly requested calls including captures:2016forward/384backward,
1800cumulative modelseconds. Combined768refs/1536actions/3072endpoints.
If this bounded cost cannot change the P-vs-B decision, stop at a checkpoint;
never silently replace the design with a convenient subset.

No new fits, labels, text generation, target words, model/layer/radius choices.
All15HM30source fields, two directions, digit alphabet and actual delta stay
fixed. Capture/source acquisitions first, checkpoint, then held acquisitions.

## Implementation Boundaries

Use new helper/runner/verifier/tests/spec names and a fresh result directory.
Do not edit frozen HM32/HM31/HM30 primary or companion dependencies.
Own hook first audits natural receiver H/Q/P; replace only explicitly mapped
H/P/Q addresses; preserve B bitwise relative to the live receiver. Create a
postpatch leaf for gradients, not a donor-dependent prepatch gradient.
Save natural P tokens/addresses/hash, H/Q/readout checks, all actual arrays,
source barrier and every call in a recoverable journal. Ambiguous missing
in-flight outputs stop instead of replay. No automatic gradient inference
from old endpoints or query suffix alignment based only on sequence length.

## Mathematics and Output

Standard characters of the experimental index group Z2^4; product-average
alternate verification. The old P=B restriction aliases the two corresponding
bits. State this finite algebra, not a Lie action or natural independence claim.
Retain16terms, signed collapse terms, and exact inherited HM32 restrictions.

Report property gradient/covector, two fixed task projections, raw/unit actions,
and native levels separately. At fixed H/Q/P vary B; at fixed H/Q/B vary P.
Compute optimistic H+Q, H+P+Q and H+B+Q conditional-variance action bounds.
Compare their MSE removal within this common design, not against HM32 ratios
as though changing design were predictor improvement. Retain all roles/queries.

Synthetic complete-group/alias/known-P/known-B/mixed/no-op/address/gradient-leaf/
shared-delta tests precede freeze; frozen before actual calls. Alternate scores,
exact rational units and conditional means reconstruct every actual output.

## Decision and Continuation

If P materially removes the observed R obstruction, the next packet constructs
a source-only target-conditioned action law on actual H/P/Q units, then cold
applications. If B survives strongly, choose a new explicitly addressed text
carrier intervention, not an undifferentiated larger kernel. Mixed effects
remain a multi-carrier action law. Do not define an arbitrary tiny cutoff.

The useful deliverable is a carrier localization plus its equation and units,
not psychological accuracy or global completion. H/P/Q extraction still uses
the frozen encoder. Original18/100/44/natural author B_u/psychological meaning
stay open. NoAPI/privateMEPS/Claude/automation/neuraltraining/commit/push/tag/
lockbox/old queue. Parent goal remains ACTIVE; this plan is not an execution.
