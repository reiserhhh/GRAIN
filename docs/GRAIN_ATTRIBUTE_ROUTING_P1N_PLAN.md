# P1N: crossed evidence attributes and conditional vector composition

2026-09-07. Frozen in8f0673f, now complete and independently audited. Original
prospective protocol retained below; results in ATTRIBUTE_ROUTING_REPORT.md.
No external people, labels, new model, downloads, API, training or layer search.

## Decision

Does an explicitly targeted intermediate state selectively carry the requested
attribute, and does joint-evidence interaction arise from state composition,
downstream response nonlinearity, or both? P1M reduced irrelevant room-anchor
effects sharply, but all room records were open. That could be insensitivity rather
than correct selection. Independently varying the room truth tests responsiveness.

## Frozen material and interventions

Take original-order P1M only; retain event and outgoing-attribution questions for
both people and the one explicit room question. Drop the now-diagnosed person-
prefixed room query from this NEW panel, not its historical evidence. Four settings
x2 behavioral-role assignments x2 room truths x5 questions=80 prompts. Room truth
changes only `The room was open.` to `The room was closed.`; retain all action text.
This states a room condition, not a manipulation of anyone's actual opportunity.
The A=Yes/B=No positive questions stay fixed; P1K code/polarity limits are not erased.

For each case at block16 capture states and swap a single anchor from same-target
counterfactual records: B (opposite behavior), R (opposite room truth), BR (both).
Name donors use that person's event-query cache in each world; room donors use
their own object question. Same-prefix name state must equal trait state exactly
within1e-5. All donor layouts must match before forward, with fixed256 execution
length/FP32/MPS. FullEQD B/R/BR controls must reproduce native donors. Original
open-world40 baselines must replay P1M; closed-world token alignment is mandatory.

Add ONE mathematically composed anchor n_add=n_B+n_R-n_0, retaining recipient's
other block states. This is an artificial off-manifold intervention, not plausible
human evidence. Save its norm, displacement and difference from n_BR; no dosage
search or post-hoc norm clipping. Refuse nonfinite data. Total80 baselines +480
natural-donor patches +80 additive anchors +4 self/restore=644 forwards. Cap750
forwards/600seconds/9GiB MPS; preserve partial output but require full-panel analysis.

## Quantities and exact mathematical identity

Report native behavior role contrasts at both room truths, native room-truth
contrasts at both behavioral assignments, absolute expected preference, and all
relevant/irrelevant B/R patch effects. Use raw logit RMS and expected directions;
do not hide irrelevant errors behind accuracy or divide by tiny irrelevant gaps.
At fixed recipient states other than the anchor, with f the remaining network:

    I_total = f(n_BR)-f(n_B)-f(n_R)+f(n_0)
            = [f(n_BR)-f(n_add)]
              + [f(n_add)-f(n_B)-f(n_R)+f(n_0)].

The first term describes departure from additive state composition on this path;
the second is the remaining readout's nonadditivity on the two observed vector
steps. This exact finite identity does not uniquely partition causal contributions:
off-manifold n_add and a different anchor/reference change the decomposition.
Save the signed cross term when comparing squared energies; components can cancel.
Do not call any term personality synergy. Standard finite differences, not a new
mathematical theorem. Compare with full-state native interaction; anchor is not a
complete mediator. Four authored settings are not80 independent participants.

## Resource/value continuation

If the room anchor responds to true room changes while rejecting behavior changes,
accept finite target-conditioned selection. If it ignores both, the old low effect
was uninformative. If behavior/truth interact, use the exact decomposition before
adding more conditions. Perform a zero-forward independent arithmetic/state audit.
Only then decide whether cached directions support a transferable conditional
operator; neither positive nor negative outcomes automatically authorize a long run.
