# HM49: locate ownership participation without a predictor search

2026-10-09. Implementation prepared for a finite run; results still open at freeze.
Depends on actually verified HM48. HM45 remains deferred. Original scope stays open.

## Scientific question

HM48 shows strong person-qualified behaviour and appraisal readings on identical
actual token bags. Where does behavioural ownership participate in the vector
reading, and can a source-mean ownership displacement act across other behaviours?
First isolate behavioural ownership without appraisal; appraisal interaction is
retained for a later targeted extension, not declared noise or already located.

## Fixed materials and pair construction

Reuse both HM48 behaviour families and all six content/matched-query pairs; query
bothA/B. Pair b=0,o=0 with b=1,o=1, absent appraisal. The high behaviour remains
first and the low remains second; only the names preceding them exchange. Both
directions give48receivers (24source,24application), not48independent people.
Material and baseline outputs have already been seen: a result-guided follow-up,
not a blind material replication. No output-dependent choice of receiver/target.

Before model calls verify full contexts, query suffix identity, semantic high/low
span token identity/order, length and positions on each pair. Preserve actual
offsets and property tokens rather than resampling them. Misalignment stops that
protocol, not rescued by interpolation, a new layer or relaxed identity rules.

## Eight fixed arms, no search

At the existing block16 output, leave all other receiver positions unchanged:

1. Replace high-property states with the exact matched donor.
2. Replace low-property states with the exact matched donor.
3. Replace both property spans with the exact matched donor.
4. Replace the body endpoint with its exact donor.
5. Replace the final query token with its exact donor.
6. Add source-mean ownership differences to both property spans.
7. Matched per-span norm random changes, one frozen seed, on both spans.
8. Self replacement of both property spans (no-op).

Arms1-5 are actual state interventions; their donor controls are not standalone
text-only predictors. The query-end arm can carry the entire processed account;
success does not localize an abstract Binding ID or prove exclusive storage.
Replacing states leaves later computation free to respond; unlike Feng's storage
experiment it does not freeze all other downstream context positions.

## Source-only shared displacement

For the fixed forward orientation(highA/lowB -> highB/lowA), compute separately
for each source content the donor-minus-receiver state difference, averaging
within each high/low property span and over the two queried persons. Then average
the six content means equally, yielding two fixed vectors Delta_high/Delta_low.
Reverse orientation uses their negatives. Use current receiver states plus these
uniform vector additions at its actual high/low tokens; keep all original states.
No source selection, rank/kernel/alpha/head/layer sweep or norm scaling to outputs.

Freeze both vectors, source arrays and the random seed before any application
intervention. Source arms may use these vectors after sealing; their results are
construction diagnostics, not independent transfer. Application native outputs
are known HM48 references, never fitted or used to alter vectors/arms.

## Evaluation and boundaries

For every receiver retain native target score, full donor score and each patched
score. Report signed patch change and full ownership change, direction agreement,
absolute amplitudes and all content/person/direction rows. Ratios only when the
denominator supports them; do not turn amplitude ratios into information shares.
Compare source/application actual donor and shared displacement; success indicates
a participating shared ownership component. Failure of a mean vector while actual
donors act indicates context-dependent or distributed representation, not a call
to train a more accurate classifier. No psychological truth labels are available.

Maximum384newforward (48x8),600cumulative model seconds incl loading and failures;
0gradient/fit/generation/API/newtraining/private data. One live collector, attempts
before calls, atomic checkpoints; no uncertain-call repeat.24source receivers
(192arm calls) sealed before24application receivers(192arm calls). The random
arm uses one frozen vector per property, matched to the corresponding source-mean
vector norm; reverse orientation negates both fixed vectors. It is not matched
after observing output effects or rescaled to exact-donor amplitudes. No-op
replay is a numerical implementation check, not a
microscopic scientific effect gate. Do not repair a small genuine effect.

Outputs: all384arm addresses and results, source assets, alignment proof, native
and patch ledgers, no-op/random checks, per-target effects and alternate scalar
reconstruction. Tests cover pair span identity, sign reversal, equal-content
weighting, constant-null, unchanged outside positions and lifecycle/resource cap.
Frozen HM48 inputs/code/results remain untouched. No Claude/automation/commit/
push/tag/lockbox/oldqueue. The packet does not settle original18/100/44,
natural-author response, psychological meaning or a fully independent calculator.

The source-mean intervention is motivated by Feng and Steinhardt (2023), section
4.1, https://arxiv.org/html/2310.17191v1 . Their other-position freezing identifies
storage under a different intervention. Here later computation is free; compare
participation, not replicate their exclusive storage result. OpenAI/math's fixed
signed-sector manuscript inspires retaining components, not a transferred theorem.

Entry: scripts/run_grain_math_relational_carrier.py (prepare/collect/analyze).
Verification: scripts/verify_grain_math_relational_carrier.py . All used parent
captures, actual token alignments, code and source vectors are hash-bound before
calls; no private texts or external API. Code, plan and results remain frozen
after preparation. Status after execution belongs in a separate report/checkpoint.
