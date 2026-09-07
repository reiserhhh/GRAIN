# P1P/Q: transferable directions, remaining effects and nonlinear readout

2026-09-07. COMPLETE with independent cache audits. P1P frozen in fbd7165:
80 new prompts,640 patches,732 forwards,311.41seconds. P1Q protocol frozen in
fb6187a before reading P1P response summaries; implementation in5cf0594. P1Q used
zero forwards and0.333seconds. No new model, training, API, human text or labels.

## What the rank-one intervention does and does not reproduce

One direction per target was fixed from the four OLD P1N settings: average B
contrast for each person, average R contrast for the room. Four NEW contexts with
two new behavioral wordings supplied all held data. All future criterion suffixes,
code, observer, layer and execution shape remained fixed.

| Query / relevant change | Natural effect RMS | Projected RMS | Complement RMS | Equal-norm random RMS | Projected normalized error | Settings meeting both bounds |
|---|---:|---:|---:|---:|---:|---:|
| Event / B |.50033|.41497|.28308|.08477|.54374|2/4|
| Outgoing attribution / B |.47299|.40883|.23007|.07904|.49326|2/4|
| Room / R |.18433|.17081|.02276|.01665|.15257|4/4|

Bounds were projected error/natural RMS<=.5 and complement RMS/natural RMS<=.5
in at least3/4 settings for BOTH person criteria. **The person-channel candidate
gate FAILS.** Books/cooking pass; museum/photography do not. Room passes separately,
not pooled to rescue the person claim. These bounds describe this engineering
approximation, not psychological significance or an accepted population effect.

Nevertheless, projected B effects have the intended sign in32/32 event and32/32
attribution cases; room R16/16. The random direction is substantially weaker in
the pooled relevant responses and has50% intended signs. Thus a fixed direction
carries transferable functional sensitivity in this task family, but not a
sufficient isolated explanation. One seeded orthogonal random direction is only
a norm-matched control, not a null distribution or uniqueness proof.

Projection retains mean25.07% of person B displacement energy but only.98% of R;
the room direction retains57.12% of R and.34% of B. These are Euclidean energy
fractions, not fractions of semantic or personality information. Remaining vectors
still affect the output. Nonlinearity prevents interpreting response RMS ratios as
independent causal shares.

## Native correctness remains a separate limitation

Native event preferences match the authored role in21/32 cases; attribution16/32;
room16/16. Attribution chooses Yes in ALL32 cases. Consequently100% directed
intervention shifts do not establish accurate native attribution or calibrated
personality prediction. This panel retains P1K's unresolved code/polarity concerns;
an intervention can alter an output in the intended direction without correcting
its large baseline bias. The authored initiation/waiting contrast is a behavioral
proxy, not a measured Big5 trait or a human psychological ground truth.

## The remainder is neither zero nor automatically noise

P1Q uses the exact finite response identity, for baseline f0, natural fn,
projected fp and complementary fc:

    D=fp-f0; C=fc-f0; I=fn-fp-fc+f0
    fn-f0 = D+C+I
    SSE(fn-fp) = SSE(C)+SSE(I)+2*sum(C*I).

| Relevant change | Projected error SSE | Complement SSE | Interaction SSE | Signed cross term |
|---|---:|---:|---:|---:|
| Event B |2.368278|2.564281|.992385|-1.188388|
| Attribution B |1.741809|1.693870|1.052856|-1.004918|
| Room R |.012655|.008292|.004285|+.000078|

The person conditions show material cancellation. A seemingly adequate projected
response can coexist with a consequential remainder and interaction. The direction
and complement add in vector space within4.77e-7 FP32 rounding; their outputs do
not simply add. This is readout nonadditivity on a specified artificial-state path,
not a new theorem, unique mediation decomposition or personality synergy. Neither
the remainder nor a low-magnitude vector coordinate can be deleted on this evidence.
All irrelevant B/R rows remain in the stored tables.

## Audit and the next discriminating control

P1P independent audit refits training directions by a pooled linear solve directly
from old states, verifies disjoint contexts/unchanged questions/role ownership and
all token anchors,160 saved patch arrays and640 effects. Direction error7.81e-16,
FP32 patch replay1.88e-6, raw logits7.11e-15, metrics2.12e-13, common prefix0.
Twelve collector identity/restoration/full checks record error0, but lack separate
raw control logits; the audit verifies those as recorded, not as an independent
neural rerun. Future collectors should save their candidate logits as well.

P1Q independently replays all160 cells/30 summaries using a longdouble contrast
matrix on raw logits, dot-product energies and saved arrays; max2.69e-14. Extended
audit_r2 adds direct vector-rounding replay; audit_r1 remains as historical receipt.

**Design limit found:** both passing scenes use wording family1 and Blair-first
records; both failing scenes use family0 and Alex-first. This confounds wording and
order. Do NOT attribute the difference to language semantics or name order alone.
P1R will reverse order within the same records while keeping the fitted directions
fixed, before considering rank increases. The room finding does not remove this
person-channel ambiguity. No need to restart old long queues or ask for new data.

Artifacts under `results/grain_vector_mechanisms/`:
`attribute_direction_r1/` (directions, materials, all native states,160 exact patch
archives, raw logits, geometry,120 summaries, controls and immutable provenance),
`attribute_direction_audit_r1/summary.json`, `direction_interaction_r1/`
(160 components,30 energies), `direction_interaction_audit_r2/summary.json`.
P1P identity: `6dec702c5a73c43b1822b9fb21e913252483e7440e4d3ce4a15bd4b509c1e0c4`.
The released GRAIN scorer is unchanged. No author/group personality validity claim.
