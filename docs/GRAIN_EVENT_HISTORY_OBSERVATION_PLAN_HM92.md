# HM92: history-conditioned event observations before answer computation

This plan and all primary code/material are frozen before new captures.
No head/layer/rank/strength/teacher-fidelity search, new source calibration,
application personality question or upper answer is allowed.

## Decision, not accuracy

HM56 already excluded a common weighted combination of isolated answer scores.
HM61 already found interactions before answer softmax. M02 already handles
designed evidence history and revisions. We do not rediscover those claims.
The new question is whether an otherwise identical current behavior span has
a history-conditioned vector response to an earlier event's negation scope,
without whole-body probability competition. If so, a fixed event-only measure
is incomplete: an observation should retain history/event coupling. This is
an observer-side mechanism, not a human's response parameter or trait change.

## Frozen material and controls

Two report frames, two orders, six fresh behavior families, all 64 corners of
(x,y,g,h,a,b). x/y are behavior polarity, g/h the K1/K2 actors, a/b inner versus
outer negation in K1/K2. Reporters are the opposite person. K3 is neutral.
Total 1,536 bodies. All behavior wording is new relative to HM91; two frames
share those actions, so this is not independent lexical-population replication.

Moving negation from action to report preserves the full token bag and length.
Actor flips also preserve token bags. Verify these before calls. K2 action
token strings and absolute positions are identical under an a flip. In the
reverse order, the complete prefix through K2 is identical under that flip.
The forward-order K1 prefix is likewise identical under the future b flip.
Do not remove report context or translate intention/report to completed action.

The inherited observer is Qwen2.5-1.5B through blocks0--16 and layer17 Q/K/V,
with generic assessment system, frozen source queries/behavior covectors and
all12 heads. Scope/actor coordinates are evaluation metadata, not score inputs.
Literal subject and behavior anchors are design-provided, not a natural parser.

## Observation units and identities

Keep full cold and paired raw readings, global event contributions, and each
behavior span separately. In a span j define

    local p = softmax(source-query logits restricted to that span)
    local F_j = sum_hi local p_hi * H_hiA * c_hi
    uniform V_j = sum_h mean_i c_hi.

Local p removes competition for mass with other events. Uniform V removes
both query selection and subject gating. Save the full unprojected mean value
vector too. No contextual token is deleted from the full observation.

For an earlier-scope flip, the local F difference has the seven exact product
terms in delta p/delta H/delta c, as in HM91. This is signed algebraic
accounting, not a causal activation patch or independent mediator estimate.

Use the complete 64-point Walsh expansion. Terms using variables from both
event groups (x,g,a) and (y,h,b) are the exact nonadditive part on this design.
This is standard finite product-space ANOVA, not an invented personality axis.
Contrast future-scope effects on prefix-local/uniform fields with the actual
future prefix control. The frozen 1e-5 tolerance is a numerical control, not a
scientific effect size. A failure is retained; no precision rescue is permitted.

## Author-law consequence

On 64 corners set Pplus/minus=(1+/-epsilon*a*y*h)/64, epsilon=.5.
Both complete event marginals agree. Summing over a also preserves the entire
(x,y,g,h,b) joint law. Since a flips identical bags, full token-bag laws agree.
Every additive event-only function has identical author-law expectations.
Actual contrast equals 2*epsilon times the Walsh coefficient of a*y*h.
Compare full versus local K2 versus uniform K2 readings and both orders.
These are synthetic evidence accounts, not people or personality truth.

For each field keep the entire six-family by six-reader contrast matrix.
Describe its battery-common row mean and orthogonal reader-specific remainder;
do not pick a rank, name a singular vector or delete the common part as noise.
This decomposition has an exact energy identity but no universal-factor claim.

## Evidence, ancestry and permissions

Primary: precall controls, raw captures, all fields and raw value means, full
Walsh coefficients, all author contrasts, past/future controls, seven-term
accounts and numerical verifier. No p-value for designed corners as people.
All opposite signs/failures remain. Original18/100/44 scope remains open.

Coecke/Sadrzadeh/Clark2010, https://arxiv.org/pdf/1003.4394, introduction and
compositional-contraction definitions: joint typed composition has precedents.
It does not prove a pretrained encoder implements that categorical functor.
OpenAI/math pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a, metric/connection
06-matrix lines146--280: fixed responses and matrix remainders inspire separating
history-conditioned effects from normalization; no smooth-radius/PDE bounds
transfer to discrete text. Walsh/product identities are established math.

No API/private/MEPS/new neural training/Git/Claude/automation/lockbox/old queues.
The collector is finite1536 calls/600modelseconds and resumable. Findings that
change the observation law, rather than classification accuracy, are the
acceptance object. No positive result is required for numerical completion.
