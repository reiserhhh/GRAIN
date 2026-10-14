# HM10B3: Local Owner Binding and Target-Conditioned Reading

2026-10-08. Frozen before collecting local activations or intervening. This
mechanism follow-up uses already observed HM10B2 controlled texts, not a new
independent text replication or actual human trait data. No source/query fit.

## Scientific Decision

Does an unchanged property span carry an owner-dependent vector action that
different fixed queries read differently? Test a complete local-span donor,
a source-only shared owner frame, its complement and a norm-matched random
frame. The experiment concerns local causal perturbation in a reader, not
proof that the same state uniquely stores a person's psychological trait.

Body tokens precede the Question suffix. Their activations should not depend
on future query tokens in this causal model. Verify exact input prefixes and
actual numeric states across all queries, separately from question endpoints.

## Fixed Design and Costs

Keep all 16 existing source/validation/application families (8/4/4), three
unchanged queries and the original conditional digit alphabet. Select A/U
four corners at T=1,V=0 and T/V corners at A=0,U=0, both orders. The two
sets overlap in two cells: 14 texts/family, 224 distinct selected texts.
Capture five decoder-block outputs, indices 4/8/12/16/20, in 672 forwards.
Inspect actual model depth; block indices are outputs, not hidden-state
tuple indices. No application-driven layer search.

Capture explicit actor, property and neutral-action tokens for both relations,
plus the question endpoint at these layers. Raw addresses come from declared
material syntax. A token may overlap only surrounding whitespace; boundaries
entering other words or the query fail closed. Do not use a model score to
choose a span. Repeated body states are checked across queries; retain a
canonical body capture and actual deviations, not redundant full trajectories.

For each family: two properties x two polarities x two orders = eight
other-to-self pairs. Donor and receiver property/neutral tokens must be
byte-identical even if absolute positions differ. Each pair is crossed with
five layers x five arms x three queries: 600 interventions/family. Add six
no-op units/family (both high properties at order0, layer16, all queries).
Total 9,696 intervention forwards, separately from 672 captures. Full address
table and hashes precede intervention; each invocation <=2,000 forwards and
1,800 seconds, with recoverable units. Do not infer completion from a process
exit or run additional accuracy/rank/prompt grids.

## Source-Only Frame and Five Arms

For each layer, each source pair supplies mean token displacement d_A or d_T
from other to self, with property words held constant. Average the 32 source
pairs separately for A and T, form D=[mean(d_A),mean(d_T)], and take its
orthonormal column basis U by SVD. Numerical rank <=2 is fixed by machine
precision; it is not selected by validation scores. The same context frame
is used for all three queries. Freeze its hash before any interventions.

For receiver token matrix H and corresponding donor H', delta=H'-H:

    full_property: H* = H'
    projection: H* = H + delta U U^T
    complement: H* = H + delta - delta U U^T
    random_matched: H* = H + scaled(delta R R^T)

R is a fixed label-independent orthonormal frame of the same rank; each random
token displacement is scaled to the norm of its U-projected counterpart.
Neutral controls copy same neutral-action words from the same owner-swapped
donor. Neutral spans can carry context; they are not assumed causally null.
No-op returns the actual current states without changing them.

These are ordinary source contrasts and orthogonal projectors. PLS loadings
are not assumed orthogonal, and transpose is not used as a general inverse.
Projection plus complement reconstructs delta algebraically; their scalar
effects need not add after downstream nonlinear computation.

The intervention recomputes the remaining network in the unchanged receiver
prompt. It can propagate to later text/query states. It is a local total
perturbation effect, not the storage-only mediation experiment that freezes
all downstream text activations. Preserve that distinction when comparing
with [Feng and Steinhardt (2024), Sections 2-4](https://proceedings.iclr.cc/paper_files/paper/2024/file/9d1b7fc578c0d2d6431fc26d736ecaf3-Paper-Conference.pdf).
[Dai et al. (2026), Section 3](https://arxiv.org/html/2604.19052v1)
motivates property-token/relational-subspace interventions; do not assume this
model already has its reported cell geometry.

## Outcomes and Next Decision

Report each query x property x layer x arm in all three family roles, both
polarities and orders. Measure signed intervention delta and donor-minus-
receiver delta separately; compute their polarity contrast as the local
binding action. Do not divide by tiny natural deltas to claim transfer.
Whole-span, shared-frame and complement effects are different results.
No-op and old-readout replay diagnose the actual intervention interface.

If full property spans act but source frame does not, investigate context-
conditioned or distributed binding before fitting a larger rank. If only
neutral or broad context acts, retain that positive mechanism and expand
observation units; do not force a keyword-path narrative. If a shared frame
acts on different queries with different signs, test target-specific reading
of a common context representation. Psychological interpretation and person
response parameters remain separate open requirements.

No API, private text, neural training, publication, lockbox, Claude or old
queue. Preserve every failure and frozen result. The original scope stays open.
