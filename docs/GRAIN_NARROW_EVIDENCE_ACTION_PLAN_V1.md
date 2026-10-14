# GRAIN narrow evidence actions V1

2026-09-13. All-plan C00/C11 continuation. Freeze before any inference.

## Question and decision

Do the25 old M30B query functions retain changes caused by target-relevant
evidence, or only correlate with the levels of a particular old material set?
Separate native evidence sensitivity from affine-function transport fidelity.
Use the result to locate the next construction gap, not to optimize a leaderboard.

This extends NAV04's three-query behavioral packet. It does not turn the old
46-person MEPS evaluation into a new replication. All new text is authored
synthetic evidence, not attributed to a real participant. Full25 natural-text
edits and34-target prefix independence remain separate requirements.

## Frozen design

-25 existing questions, exact token coordinates/alphabet and model weight hash.
-20 semantic families (some pre/post queries share a family), two independently
  worded high/low evidence pairs per family. No literal questionnaire numbers,
  factor codes or target query instructions inside the evidence body.
-Each of25 targets uses two pairs x two polarities x self/other quotation.
  The quoted body is byte-identical between attribution arms; the wrapper alone
  assigns its experience to the writer or a friend.200 input worlds.
-Each world is read by its target query, one preselected related query, and one
  preselected distant query:600 complete native/state calls. These are paired
  counterfactual descriptions, not600 independent people.
-The distant query is a diagnostic of nonspecific response, not an assertion
  that the two human constructs have exactly zero association. Related query
  signs are not forced; the question identity is retained.
-Use the existing truthful online-post scaffold: these are retrospective posts
  about general AI use or a described interaction, not AI dialogue transcripts.
-Only local inference. No MEPS files, human labels, external APIs, model training,
  source refitting, lockbox access or old queue restart. Max1200s per collector
  invocation, per-unit atomic persistence, resume only this frozen packet.

## Computation and reporting

For frozen targetq and paired evidencei, compute
`D_qi(owner) = score(high,owner) - score(low,owner)` separately for native and
old affine scores. Record source mean/configuration/final contributions without
clipping. The affine anchor cancels. Attribution contrast is
`A_qi = D_qi(self) - D_qi(other)`: a difference-in-differences of fixed model
outputs, not a human causal effect.

Report every target, pair, ownership and reader role; intended-direction counts,
median/mean contrasts, native/affine response correlation, MAE, signed bias and
origin-constrained amplitude slope. Report near-zero native effects separately
(`abs(D)<=.01`, fixed descriptive resolution, not a significance threshold).
Use no tiny-template p-values or population CIs. Preserve failures and amplitude
attenuation. Cross-target data cover75 declared directed links, not a full625
matrix or25 independent psychological constructs.

All reference/input/implementation hashes precede collection. Recompute affine
scores from saved P/Q; verify paired components sum to total response, wrappers
preserve body and all600 identities are unique. Unit tests include malformed
pairs, missing cells, constant native effects, and exact affine decomposition.

## Deliverables and next trigger

`results/grain_vector_mechanisms/all_plan_closure_v1/narrow_actions_r1/`:
manifest, private world/design/packets, scores, polarity responses, attribution
contrasts, query summary, cross-query summary, arithmetic audit and run status.
Public report contains no real private text or identities. This packet is a
controlled evidence-action reference, not yet natural-text target selection.

If native and affine both follow the edits, retain that reference and move to
natural evidence localization. If native follows but affine attenuates/reverses,
inspect the specific context/derivative transport, not more human-label fitting.
If native fails attribution or direction, distinguish reader failure from the
question's insufficient operationalization. Do not call it absent human signal.

## Predecessor

Ribeiro et al. (ACL2020), *Beyond Accuracy: Behavioral Testing of NLP Models with
CheckList*, sections2.1-2.3: capability-by-test matrices, directional expectations
and relation-preserving perturbations on unlabeled text.
[Paper](https://aclanthology.org/2020.acl-main.442.pdf),
[author implementation](https://github.com/marcotcr/checklist).
We use that experimental pattern, not its software or a replication claim.
Our added object is the matched response of frozen vector functions and their
mean/configuration/final decomposition.
