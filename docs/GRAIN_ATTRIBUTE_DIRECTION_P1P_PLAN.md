# P1P: transfer a fixed attribute direction to new synthetic records

2026-09-07. Freeze before new forwards. Decision: can a single direction estimated
from the old anchor geometry reproduce the relevant natural-donor effect on new
records, rather than merely decoding an old response? P1O alone cannot answer.

## Estimation and held material

Use P1O's four old settings only. For Alex and Blair separately, normalize the mean
behavior main-effect vector u across settings. For the room normalize the mean
room main-effect vector v. No label fitting, rank selection, output-based weighting
or held-vector fitting. These are three fixed rank-one directions, not human traits.

Create four new contexts (museum guide briefing, neighborhood book exchange,
weekend cooking workshop, amateur photography meetup), two new initiation/waiting
wording families, counterbalanced name order and initiating person. Retain the
canonical event, outgoing-attribution and room questions/answer code from P1N.
4 settings x2 assignments x2 room truths x5 questions=80 native prompts.
These are new authored materials within a deliberately narrow task family, not a
representative sample of situations or independent humans.

For every prompt and B/R donor displacement delta, test at block16's single target
anchor, holding all other recipient states fixed:

    natural = n0 + delta
    projected = n0 + d * dot(d, delta)
    complement = n0 + delta - d * dot(d, delta)
    matched_random = n0 + q * dot(d, delta)

q is one seeded random unit vector orthogonal to d for each target (seed20260907).
It matches the projected displacement norm and reversal sign, not its direction.
One random direction is a perturbation control, not a random-subspace null
distribution. All artificial patches can be off-manifold. Save the exact FP32
patch vectors and their float64 reconstruction errors. Natural person sources
are canonical event anchors, shared exactly with attribution before its suffix.

## Analysis and decision

Save every native and intervention candidate logit, geometry, donor map, controls,
token positions, source/model hashes and directions. Primary outcomes, by setting
and criterion, are projected-minus-natural normalized RMS error, complementary
and matched-random effect RMS relative to natural, and relevant-direction counts.
Keep irrelevant effects and baseline truth errors. Ratios are not mediation shares.

A compact-channel *candidate* requires projected normalized error<=.5 and
complement relative RMS<=.5 in at least3/4 new settings for both event and
attribution. This is a declared descriptive engineering gate, not a psychometric
cutoff or significance test. Also report whether matched random is weaker in each
setting; do not grant direction-specificity if the control performs similarly.
Room is a separate target, not pooled into a favorable average. Failure rejects
this fixed rank-one approximation, not the existence of encoded behavioral cues.
No post-result rank/seed/layer/wording search or automatic larger-model run.

## Resource-value boundary

80 native +640 patches +6 identity/restoration +6 full-body restoration=732
forwards, hard cap800 calls/600seconds/9GiB allocated MPS. Existing local observer
only, FP32 eager/padded256; no training, API, private corpus, labels or download.
Flush after each complete20-prompt setting (and native stage). The four settings
are the minimal planned comparison, not a long queue; stop early only for material,
instrument or resource failure and report partial status without promotion.
Independently replay source vectors, projection geometry and metrics from cache.

After the complete panel, choose the cheapest next experiment based on the failure
mode. Geometry-transfer failure and functional-transfer failure are different.
If only whole-state interventions work, do not advertise an isolated personality
direction. Old source queues remain paused; no Claude assistance needed here.
