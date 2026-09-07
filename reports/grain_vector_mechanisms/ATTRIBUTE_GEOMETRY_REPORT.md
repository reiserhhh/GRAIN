# P1O: encoding geometry and conditional finite gain

2026-09-07. COMPLETE, independently replayed. Source frozen in535a044. Zero new
neural calls,0.248seconds cache computation. All12 rectangles/48 canonical vectors
and160 B/R response rows used; no new human text or labels.

## Two different sources of selectivity coexist

| Query | B displacement RMS | R displacement RMS | B logit-effect RMS | R logit-effect RMS | B finite gain | R finite gain |
|---|---:|---:|---:|---:|---:|---:|
| Event |13.81942|1.97464|1.26045|.06795|.09121|.03441|
| Outgoing attribution |13.81942|1.97464|1.24325|.07953|.08996|.04028|
| Room |4.72158|5.94848|.03628|.25456|.00768|.04279|

Event and attribution use exactly the same name states. In them, the B-to-R
displacement ratio is about7.00; the subsequent finite gain ratio is2.65 and2.23.
Their products give the observed effect ratios18.55 and15.63. For the room, R-to-B
displacement ratio is only1.26 but finite gain ratio5.57, together giving7.02.
These are descriptive factorings of a ratio, not independently identified causal
contributions, information fractions or gradients. Gain depends on query, other
recipient states, path and displacement scale.

Thus both target-indexed encoding and downstream functional use matter in this
panel. In particular, the room anchor changes substantially when the people's
behaviors swap, despite its weak effect on the room answer. Weak response is not
proof that irrelevant information is absent, and indiscriminately deleting vector
variation is not justified. The name anchor already emphasizes behavior more than
the room fact; a pure readout-only explanation is also incomplete.

## Exact finite vector decomposition

For each target and setting, four independently crossed worlds give:

    n(b,r) = mu + b*u + r*v + b*r*w,  b,r in {-1,+1}.

The four coefficients are balanced finite contrasts, not discovered factors or
personality directions. B indexes the queried person's initiation for names and
record assignment for the room. B/R main-effect angles range79.61--102.16degrees;
mixed-to-main norm ratios range.0591--.1169. Near-orthogonality in a1536-dimensional
representation does not by itself identify semantic independence or unique axes.

The mixed displacement nBR-(nB+nR-n0)=4*b*r*w holds exactly in float64; actual
FP32 additive-state construction differs by at most3.81e-6. Modest mixed-vector
norms do not imply an additive decoder: P1N already measured opposing state and
readout departures. Norm and output effect must remain separate endpoints.

## Independent checks and reproducibility

`scripts/audit_grain_attribute_geometry.py` uses a direct four-column solve rather
than the collector's signed averages; it reconstructs norms/angles,160 changes
from raw candidate logits,40 rectangle gains and6 pooled summaries. Coefficient
reconstruction error0, geometry2.84e-14, raw-logit replay7.16e-15, gain3.89e-15.
Source/output snapshots and P1N parent identity verified. Three synthetic tests
cover exact expansion, rotation invariance and incomplete-grid rejection.

Inputs: `results/grain_vector_mechanisms/attribute_routing_r1/`.
Outputs: `results/grain_vector_mechanisms/attribute_geometry_r1/`:
`coefficients.npz`, `coefficient_index.json`, `geometry.csv`,
`mixed_state_replay.csv`, `displacement_rows.csv`, `gain_by_rectangle.csv`,
`summary.csv`, manifests/hashes/status. Independent receipt:
`attribute_geometry_audit_r1/summary.json`.

Next P1P freezes rank-one directions from old geometry and intervenes on four NEW
settings with new behavioral wording. This tests causal response transfer, not
merely vector alignment. No old long queue resumes; no personality accuracy claim
or GRAIN scorer change follows from P1O. One local model and one layer remain a
major scope limitation.
