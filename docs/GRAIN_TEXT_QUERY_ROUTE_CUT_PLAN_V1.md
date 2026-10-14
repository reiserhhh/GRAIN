# Text to Query Route Decomposition

HM42, 2026-10-09 JST. Planned after HM41 verification; NOT implemented or run.
This is a result-guided mechanism follow-up, not a retrospective preregistration.
Keep every frozen HM35-HM41 artifact, including unsuccessful controls, unchanged.
Goal ACTIVE; original 18/100/44/natural-author/B_u/psychological scope OPEN.

## Scientific question and bounded decision

HM41 located useful variation in J_H, but its source mean does not preserve
all wording contrasts. Determine where the actual text-to-logit derivative
travels at a single downstream decoder cut: body states, early-query states,
or remaining endpoint/layout states. This selects the next reading operation
to model, rather than another coefficient, rank, layer or accuracy search.

Use the original 96 HM39 midpoints, all three queries, two roles, two wordings,
eight H/B/Q cells and both physical H directions. Same Qwen2.5-1.5B observer,
eager attention, MPS, frozen parameters, exact tokens/prompt/alphabet and input
leaf at output of block index 16. Capture the entire output S of the next block,
index 17 (zero-based decoder indexing), without changing that output.
Check actual model depth and hook conventions before the first model request.

The next-block states do not exist in the current cache. The experiment needs
new finite local forwards; do not label it a zero-call cached computation.
It remains a mechanism follow-up on existing material, not fresh replication.

## Exact first derivative graph cut

The complete sequence S separates the patched block16 leaf X from logits l.
Use three disjoint, exhaustive masks at S:

- body: all positions before question_start;
- early_query: the exact original ordered P positions;
- remainder: every other position, including endpoint Q and any trailing layout.

Never silently assign the endpoint to the early-query group or discard remainder.
Preserve original token IDs, absolute addresses and group masks per path. H remains
the exact four-token property span; group labels describe positions, not semantics.

For digit logit j and group G, define

```text
J_H,j,G = [D_H S]^T mask_G(gradient_S l_j)
J_H,j = sum_G J_H,j,G
v_link,G(d) = J_P^T C J_H,G d
v_link(d) = sum_G v_link,G(d).
```

Obtain gradient_S l_j once and detach it before each VJP through S into the
block16 leaf. This is the first derivative chain rule, not differentiation of
the gradient weight; differentiating it again would double-count a different
second-order object. Reuse actual frozen HM39 J_P, C, displacements and decoder
after the unchanged logits and Jacobian-sum replay succeeds. The last contraction
keeps J_P full; these are routes of the H leg, not a complete two-leg path census.

The cut exhausts sequential decoder dependencies with use_cache=False. Verify
this on the real model and synthetic residual/branching examples. A bypass of
S invalidates the cut and stops the packet; do not fill a missing route with zero.
Record the sum-minus-parent remainder rather than hiding physical rounding.

This is an exact local derivative allocation at one chosen cut. Backward masks
are not actual forward interventions; the allocation is cut-dependent and does
not uniquely identify attention heads, neural circuits or a psychological factor.
It does not replace decoder curvature, integrate the full path or compile away
the reader. Those distinctions govern how the output can be used, not a reason
to discard a detected route.

## Acquisition order and resource limit

Seal code/spec/plan and verify HM41/HM39 ancestry first. Source 48 paths are
acquired, saved and hash-sealed before application 48. No source fit or target
selection; no application-dependent layer, grouping, radius or prompt choice.
Keep all 96 addresses and the two physical arms. Reuse exact HM39 receiver/donor
construction at t=.5 and its audit before creating the block16 gradient leaf.

Maximum 96 new forwards, 3840 first-gradient calls and zero second-gradient
calls. Per forward: ten digit-to-S gradients and three S-to-H VJPs per digit,
40 first-gradient calls. Compare the route sum to saved HM39 J_H without a new
direct gradient. Log each attempted/completed call and cumulative model time.
Model-time cap 1200 wall seconds includes model loading, failures and replay;
CPU preparation/result/alternate verification cap 180 seconds includes imports
and failures. No cap increase, uncertain-call automatic replay or model switch.
If the cap or replay gate fails, retain artifacts and stop this finite packet.

Keep source-first provenance. Hash all observed node arrays and audit logs before
analysis. No new neural training/API/private data/Claude/automation/publication.

## Analyses without new fitting

Save all three route Jacobians, raw digit gradients at S, masks, logits and state
audits. Reconstruct both physical H actions and three P covectors. Report complete
route norms, signed action moments and all pairwise cross terms by every query,
role, arm and H/B/Q address. No deletion or best-route selection.

For the two wordings, compute individual actions, pair means and contrasts;
retain the negative-affect known_mean contrast counterexample. Component squared
magnitudes do not add without cross terms and are not personality percentages.
Compare source/application route profiles only after keeping query units and
absolute/relative token addresses. No fitted attribution from application labels.

Focus on:

- Does query-position flow carry the variation missed by a mean J_H law?
- Does a body carrier survive even when query states are fixed?
- Is endpoint/remainder aggregation needed, or do signed routes cancel there?
- Are the known_mean and unseen_source wording contrasts different mechanisms?

The groups are fixed before these questions are evaluated. Neither the largest
component norm nor a favorable average alone decides what is preserved.

## Tests, verification and deliverables

Synthetic tests cover exact chain sum, noncommuting ordered maps, disjoint and
exhaustive masks, residual branches, detaching gradient weights, cancellation,
unequal H norms, query layouts, source isolation and journal/cap failures.
Use a known bypass example that must fail the full-cut assumption.

On actual data, unchanged HM39 logits, native score, H/P addresses and route-sum
J_H must replay within the original 1e-8 physical tolerance. Reconstruct HM41's
111 link with both arms, and full action with saved decoder/roundoff. Independent
numerical verification contracts digit-level directional Jacobians with the
saved C/J_P and replays every table and source hash; no new model calls in audit.
If float reassociation fails the physical gate, preserve the failure; do not
silently relax tolerance or substitute an approximate cut.

Outputs: call/resource ledger and source barrier; 96 addressed node archives;
route observations, signed moments, wording mean/contrast table, action replay
errors and frozen output hashes; alternate verification; theory note and Chinese
parent report. Tests and hash counts document execution, not scientific closure.

If query routes carry useful variation, develop source-supported conditional
query reading. If body/remainder routes persist, preserve their addressed carriers
instead of reducing text to a single query coefficient. If routes strongly cancel,
retain the joint route object before transport. Mixed outcomes keep target/arm
branches. Do not continue an accuracy/rank grid or treat a local route as a new
human-personality dimension. Natural authors, B_u and the original scope remain.
