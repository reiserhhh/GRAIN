# HM31: Does Context Act Outside the Property Vector?

Status: execution protocol prepared; model calls not yet acquired at freeze.
Results and status belong in the packet ledger and separate report, not edits
to this frozen protocol. Result-guided follow-up to frozen HM30.
Decision: distinguish context encoded in H from unpatched context c modulating
the action of IDENTICAL H. Do not add kernel/rank/prompt/precision grids.

## Fixed Cross-Recombination Design

Use all HM30 four-token wordings and three original queries, unchanged model,
post-layer16 convention, alphabet, physical directions and .01 relative radius.
Pair contexts (0,1),(2,3),(4,5),(6,7), in that order. First two pairs source,
third validation, fourth application. All4 wordings and all3 queries compulsory.
For each pair/wording/query form four corners:

    F_ij = F(H_i, c_j, q), g_ij = dF(H_i,c_j,q)/dH, i,j in {0,1}.

H_i is the exact HM30 property state of donor i; c_j includes every other
post-layer16 receiver state, including query Q. Donor/receiver property tokens
must have identical token IDs and bytes for that wording; positions may differ
but explicit semantic-span mapping is compulsory. Preserve all other receiver
states. These are teacher-forced synthetic state interventions, not natural
counterfactual author statements or re-encoded conversations.

Diagonals i=j and their +/- actions inherit HM30 frozen artifacts; do not rerun.
Only96 off-diagonal references and384 off-diagonal endpoints are newly acquired:
maximum480forward/96backward,1800cumulative modelseconds,0neural weight updates.
After combining inherited diagonals there are192 addressed references,
384 actions,768 endpoints and48 paired four-corner units, not192 authors.
No new fit is necessary: retain frozen HM30 global/wording fields as cold models.
Acquire source off-diagonal controls before held; no use of held gradients to fit.

## Exact Input and Gradient Rules

In a new baseline hook, first verify live receiver H against its saved natural
reference; then replace only the property with donor H and make that post-hook
state a gradient leaf. Save the actual patched H, receiver original H, gradient,
query state, native score, donor/receiver addresses and hook audit.
Do NOT use HM29 patched() unchanged: its live-reference assertion expects the
same H as the replacement baseline, which is false at off-diagonal corners.

For endpoints use donor H's actual float32 +/- displacements. At a fixed donor i
both receivers see EXACTLY the same H and physical half-delta. Verify equality
before comparing c effects; do not normalize away mismatched donor identities.
Save odd action, even change, candidate logits and all address/hash dependencies.
One committed artifact per call; ambiguous in-flight state stops, not blind replay.

## Analysis and Scientific Branches

Analyze baseline F, local gradient g, covector O4^-T g, actual finite action A,
and A divided by actual delta norm. For every paired four-corner unit use the
fixed2x2 product projection to separate donor-H, receiver-c and H×c terms.
Retain equal weights, raw units and signs. Derivative energy and chosen-direction
action importance are separate estimands; use the fixed action seminorm, no
estimated best rank or thresholds.

Primary outside-context contrasts g_i0-g_i1 and A_i0-A_i1 compare IDENTICAL H.
Native F level changes alone do not establish changed sensitivity.
The H main effect is not a personality effect; it also carries donor context.
H×c is an actual decoder recombination mechanism, not human psychometric variance.
Evaluate frozen HM30 fields on every corner, retaining all failures and zero/tiny
directions; do not fit a new field just to make a positive headline.

If outside c changes action materially in recorded native units, retain c in
the common formula and investigate the smallest observed contextual coordinate
needed for cold compilation. If it does not, prioritize H-based field transport
to new wordings. A mixture demands both accounts, not automatic condition removal.
Use recorded native-score reconstruction error and exact hook audits as numerical
references, not a hand-picked significance cutoff or personality claim. HM30 has
no dedicated new no-op arm; do not invent one from unrelated inherited materials.
If extra no-op inference is needed, specify and budget it before acquisition.

## Acceptance

Verify96 actual new references/384 endpoints,192 total references/384 actions/
768 endpoints,48 four-corner coverage; no diagonal replay or source-role leakage.
Alternate native score reconstruction, rational O4 inverse, explicit2x2 averaging,
realized-delta equality and all cold metrics; synthetic tests for leaf-gradient
addressing, corner decomposition, interaction-only cases and identical-H context
counterexamples. Freeze new code/material addresses BEFORE model calls.
No private text/API/new training/Claude/automation/commit/push/tag/lockbox.
Preserve HM30 and all older primary outputs; original18/100/44/natural/B_u/
psychological meaning scope stays open. This is a finite observation mechanism
packet, not closure of the Project Persona theory.

## Fixed Output Accounting

Per48 units save gradient, covector, two-direction task-gradient, native level,
and both arms' raw/unit/local-unit/cold-unit actions:576 product decompositions.
Save contrasts for gradient/task-gradient/native-level and both arms' raw/unit/
cold-unit actions:1728 addressed contrasts. All36 role/query/arm/diagonal cold
metric rows are compulsory. The task-gradient metric is a fixed rank-at-most2
seminorm, not a complete hidden-space metric or source-estimated optimum.
The receiver-c term includes query and position/layout; do not label it pure
topic. Source checkpoint records96+192source files (48newrefs with arrays and
192endpoints), without learning any new field from recombinations.

## Elementary Descent Test

Write pi(H,c)=H and alpha_q(H,c)=d_H F_q(H,c). An H-only local action law
requires equal alpha_q at identical H for the compared receivers. If
F_q(H,c)=f_q(H)+b_q(c), symmetric finite increments at H+/-delta are also
independent of c. Thus unequal identical-H finite increments refute this
restricted action law even when context-dependent level offsets are allowed.
This elementary necessary condition is the actual experiment, not a claim
that decoder contexts form a smooth principal bundle or obey a PDE inverse
theorem. Finite equality alone would not establish a global descent law.
