# HM135: unchosen alternatives as conditional evidence

2026-10-11. Codex bounded mechanism packet. Original18/100/44 scope OPEN.
This does not optimize prediction accuracy or fit an observer to teacher scores.

## Missing operation and prior inventory

HM134 changes whether the exact performed action was voluntary or imposed.
M03/F03 already model opportunity supply, selection, and missing denominators.
HM119 already has behavior/owner/query interactions. These are not new claims.
The missing native experiment asks whether an **unchosen option's availability**
modifies its effect on trait judgment while performed action and voluntariness
remain fixed. An option can be described without having been available.

Competing explanations: semantic option words affect all readings irrespective
of feasibility; versus a feasibility-dependent operation that changes how the
same performed act supports a target. Neither is assumed absent. A nonzero
four-corner contrast establishes an operation, not a unique Bayesian mechanism.

## Standard mathematical construction, not a new theorem

For offered options M and declared feature f, a finite choice model gives

    p(a|theta,M) = exp(theta^T f(a)) / sum_{j in M} exp(theta^T f(j)).
    L(a,M) = log p(a|theta_high,M) - log p(a|theta_low,M).

The normalizer depends on options not selected. In an explicit one-dimensional
example theta=+/-1, chosen f=0, an available alternative f=-1/+1 gives L=+1/-1;
if that option was unavailable, L=0. This is a model consequence used to design
a discriminating contrast, NOT observed human truth or a novel mathematical result.

Read Cao/Cohen/Szpruch2021, pp1-5, especially Eq5 and Theorem1:
https://proceedings.neurips.cc/paper/2021/file/671f0311e2754fcdd37f70a8550379bc-Paper.pdf
Its dynamic MDP identifiability assumptions are not transplanted to this text task.
Read the existing20260912 seminar's offered-set warning before this design.
OpenAI/math adc7f124 Hellinger induction/compression sections1-80 preserve one
observable while changing a conditional profile; this inspires the controlled
comparison, not a transferred Boolean-noise inequality or personality theorem.

## New public materials and locked operation

48 new fictional records: 2 wording banks x6 behavior families x2 unchosen
alternatives x2 availability conditions. A performs the same moderate act within
each family in all4 corners, freely. B only observes. The unused option was never
performed. Families index E/N/A/C/O/anthropomorphism; no human labels or fitting.
Here engine field b means ALTERNATIVE polarity, not performed-behavior polarity.
Engine c means availability, not voluntary-vs-assigned action.

Use frozen Qwen2.5-1.5B float32/MPS/eager, H-only native suffix access; this is
the existing experimental reading program, not stock all-access inference.
Keep the12 frozen A/B trait questions plus2 inherited action-fact questions.
Fixed187-token body, cause clause19 tokens at149-167. Only open/closed changes
within each alternative pair. Masked padding is retained; no token truncation.
The preceding performed action and unused-option description do not change
when availability changes, and their causal K/V must match exactly.

Reuse the sealed HM134 addressed-KV engine via a scoped config adapter, without
editing sealed source. Separate files/manifest/outcomes describe HM135. The engine
may print its historical HM134 label; CLI also identifies the HM135 experiment.
Swap ONLY the19 availability-clause K, V or both at all28blocks/all suffix rows.
Recipient body remains unchanged; four self swaps and complete K/V swaps are
execution controls, not independent mechanism discoveries.

## Complete estimands and stopping rules

    D_closed = Y(high_option,closed)-Y(low_option,closed)
    D_open = Y(high_option,open)-Y(low_option,open)
    I = D_open-D_closed

Report all6family x6target fields for A and B, plus unchanged-action facts and
digit mass/entropy. Report I_A-I_B. Retain signs, off-target results, all units,
and exact K/V/joint components. A negative matching diagonal is consistent with
the simple relative-choice example; it is not a calibrated personality verdict.
No p-values/people-CIs for these designed fictional records. No picking a best
target/bank, polarity flipping or output-unit rescue.

Exactly196 batches/2744 endpoints, max1100 cumulative model seconds including
loading. No retries after ambiguous failure, renewed cap, neural training, API,
private text, Claude, Git, automation or lockbox. One collect, analyze, read-only
verify. If results show only generic option-word response, record that boundary;
do not tune prompts, heads, layers, wording or ranks to recover a desired effect.

The next construction decision is whether GRAIN evidence units need an offered-set
coordinate and a separate unperformed-option relation, beyond action/owner/cause.
This packet can reveal a native feasibility-dependent operation; it cannot by
itself estimate people's preferences or establish a new global personality theory.

