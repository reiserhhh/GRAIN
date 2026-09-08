# M28: expand functional construction to every PANDORA target

2026-09-09. Frozen before new neural work. M27 catalog exposes five domains and
four axes. Use the same frozen Qwen2 observer, layer16, native finite alphabets,
S1 24 authors/two views and M25 12/12 reference/evaluation assignment.

## Construction, not supervised score fitting

Reuse M26's exact T/F and neuroticism functions and M25 states. For the other
seven queries construct G, g, b, P_R, Q_R using the **same two original M7
reference authors**, not the new evaluation authors or questionnaire labels.
The 12-author reference partition remains for author norms only.

For each of the two reference authors, capture full and E1 edited states.
Align unchanged question tokens exactly. Average Simpson-integrated question
gradients over E=full/edited evidence contexts while final Q stays at the full
origin, exactly as M7. Average these two authors' matrices to obtain G_q.
Use the full-state final-token gradient for g_q, full-score mean for b_q, and
mean full P/Q states for the reference. Thus the same M26 affine formula is
constructed, rather than replacing it with a human-label-trained model.

Collect full/E1/E2 P/Q and native scores for all 24 authors x 2 views x 7 new
queries: 1,008 full forwards. Reference derivatives add 98 suffix gradient calls
(two authors x seven queries x [six path nodes + one full-state joint gradient]).
Reuse both old core queries' 288 states. Confirmation seeking remains in its
old bundle; it is not silently relabeled as an externally measured scale.

E1/E2 are **legacy common edits** originally selected with the three old
queries. This asks whether the new functions read changes in the same evidence,
not whether these are optimal interventions for all nine traits. Report actual
native response SD; a small response identifies an evidence-coverage problem.

## Outputs and decisions

- Hash-bound per-query functions and exact ordered query-token identities.
- Every author/view/condition mean, configuration, Q, predicted and native value.
- Per-query held-author level and response agreement, MSE/zero baseline, native
  response SD; author-cluster resampling, not treating edits as independent people.
- Mean/configuration/Q ablations using the same frozen function, no new fit.
- Separate reference-standardized author positions; human outcomes join later.
- Fixed-model action is measured by the construction; held natural-response
  agreement tests its usable transport. It is not automatically external validity.

Save each unit and reference node. Cap each run at 1,200 seconds, two CPU threads;
resume only when the remaining named query changes trait coverage or construction.
No new download, training, API call or old lockbox read. Never restart completed
units. If a new query has poor response reading, inspect which of M/C/Q loses the
effect and take one query-specific construction step, not a parameter sweep.
