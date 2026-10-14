# Matched-Token Whole/Piece Construction Results

2026-09-13. C17 / V4 Study B semantic construction reference completed within the
pinned encoder's input support. **No universal slicing upgrade. Keep observation
coverage, contextual encoding, aggregation and readout as separate design choices.**
The all-plan goal remains ACTIVE.

## Findings and resulting production rules

Three observations change the construction guidance:

1. With exactly the same content-token sequence, whole and piece representations
   are related but not identical. About92% of their squared vector distance in
   PANDORA and94% in Essays is angular rather than radial, under the Euclidean
   law-of-cosines decomposition. Renormalizing pooled vectors does not undo most
   of the context difference.
2. Fixed semantic anchors and label-trained heads respond differently. Direct
   anchor mean-r increases by about.033/.036 in PANDORA/Essays with token-mass
   pooling; trained PCA/Ridge mean-r changes by-.013/+.010. Neither trained-head
   interval excludes zero. Therefore better alignment to a fixed semantic key
   is a distinct outcome, not an automatic prediction-model upgrade.
3. Truncated document and complete author material are very different designs.
   The first500 tokens cover a median17.1% of PANDORA and66.1% of Essays material.
   PANDORA's full36-comment reference is r=.318143 versus prefix-whole .145333.
   This comparison changes coverage AND unit boundaries/weights, not slicing alone.

Retain the existing full-material semantic reference. Retain the matched-piece
anchor route as a local construction option, not a universally optimal cut.
Do not expand the window/alpha grid on these held results. Next decisions concern
native readout integration and target-defined condition/path evidence, not
another round of generic pooling optimization.

## Protocol and participants

Frozen plan: [matched context plan](../../docs/GRAIN_MATCHED_CONTEXT_CONSTRUCTION_PLAN_V1.md).
The same624 PANDORA Big5-only and2467 Essays authors as `local_bilinear_r2` were
used. PANDORA train378/validation126/refit504/held120; Essays1929/284/2213/254.
Official Big5 fold0 and Essays fold9 are the held partitions. These are previously
explored developmental materials, not a new lockbox or fullCV evaluation.
PANDORA labels are continuous0-1; Essays labels are binary high/low0/1.

The [official E5-large-v2 card](https://huggingface.co/intfloat/e5-large-v2), W68,
provides the query-prefix, masked mean/L2 extraction and512-token input cap.
We use pinned revision `f169b11e22de13617baa190a028a32f3493550b6`, cached safetensors,
local MPS float32. No weights are trained and no raw texts leave the machine.

Original aggregated documents were tokenized without personality-term cleaning.
First min(500,T) content wordpieces were retained, then encoded as one whole or
consecutive <=128-token pieces. Source content IDs concatenate identically in
both arms. Original character offsets and omitted counts are retained. Pieces
can begin inside a wordpiece sequence; they are computational token partitions,
not validated utterances or psychological events. Splitting jointly changes
context, position reset and repeated prefix/special tokens; it does not isolate
attention as a causal mechanism.

| Context | Authors | Complete documents within500 | Complete among held | Median coverage, all |
|---|---:|---:|---:|---:|
| PANDORA |624|0|0/120|.170853|
| Essays |2467|432|44/254|.661376|

Four Essays authors have only one piece, including one held author. Their whole
and piece inputs are identical; normalization differs at most by floating-point
roundoff. All authors remain in the primary comparison, not only short documents.
Long-document whole-context inference remains beyond this encoder's support.

There are15,225 logical input packets and15,218 distinct model inputs; maximum
actual encoded length504, with zero hidden truncation. Stable length sorting
reduces padding; vectors are restored to their original packet indexes.

## Mathematical objects and exact accounting

Let H be the normalized whole-input vector, h_j the normalized piece vectors,
w_j=n_j/sum(n), z=sum_j w_j h_j, r=||z||, v=z/r, and D the frozen30 keyed IPIP
facet directions. The four fixed arms are H, mean(h_j), z and v.

```text
H D - z D = (H-v) D + (1-r) v D
||H-z||^2 = (||H||-r)^2 + 2 ||H|| r (1-cos(H,v))
```

The second equation retains actual floating-point ||H|| in replay; the ideal
unit-H formula sets it to1. These are inherited linear algebra identities, not
novel mathematical theorems. In semantic-coordinate space, the directional and
radial terms are generally not orthogonal; the cross-term is retained.

| All-author geometric summary | PANDORA | Essays |
|---|---:|---:|
| Median pooled radius r |.920087|.932897|
| Median cosine(H,v) |.960115|.965405|
| Mean total squared vector difference |.080531|.069387|
| Radial contribution |.006391|.004457|
| Angular contribution |.074140|.064930|
| Mean semantic difference energy |.00076072|.00088256|
| Semantic direction energy |.00076578|.00086342|
| Semantic radial energy |.00001780|.00002841|
| Semantic cross-term |-.00002286|-.00000927|

High raw cosine is not evidence of personality validity. Across held authors,
the five whole/mass-pooled anchor score correlations range .801-.883 in PANDORA
and .843-.876 in Essays. This is same-material operator agreement, not test-retest
reliability across occasions. The order information changes less than the full
vector, but it is not identical.

Each source-fitted PCA/scaler/Ridge is exported as a raw affine function
`prediction = x beta + intercept`, with source-only reference means/SDs. A pooled
head also has exact per-piece contributions. For normalized pooling, weights
become w_j/r; the intercept is added once, not to every piece. This is numerical
attribution to the chosen computational units, not proof that a unit carries an
independently identified trait.

## Complete held comparison

Mean Pearson across allfive traits; continuous scores are not clipped. PCA64,
scaling and alpha {1,10,100,1000} are source-training/validation only, with final
source refit. All four arms and all outcomes are retained.

| Fixed observation + local PCA/Ridge | PANDORA120 | Essays254 |
|---|---:|---:|
| Whole prefix |.145333|.218350|
| Equal piece mean |.134828|.222245|
| Token-mass piece mean |.132048|.228562|
| Token-mass mean, L2 normalized |.132174|.229535|
| Previous full-material semantic reference, different coverage |.318143|.233362|

| Mean-r difference versus whole | PANDORA paired95% CI | Essays paired95% CI |
|---|---|---|
| Equal mean |-.010506[-.075056,.048230]|+.003894[-.025794,.031067]|
| Mass mean |-.013286[-.079523,.045546]|+.010212[-.018134,.038027]|
| Mass normalized |-.013159[-.077027,.045495]|+.011185[-.017263,.039373]|
| Full material, NOT slicing-only |+.172810[.079169,.270481]|+.015012[-.022605,.051927]|

Intervals use1,000 paired author draws conditional on fixed fitted heads, not
refit uncertainty or familywise confirmation. The inherited scope string says
`oof`; here it denotes one held fold, not a complete pooledCV. None of these
figures is a new comparison against historical pooled r~.196.

Chosen PANDORA alphas are whole10/equal1/mass1/normalized1000; Essays all1.
Source-validation favored pieces slightly in PANDORA, whereas the held ranking
is opposite. Do not choose a new alpha or claim global equivalence from that.

## Fixed semantic anchors, without human-label fitting

All30 signed facet directions are fixed from the same300 IPIP item embeddings;
six facets per domain are averaged to form five semantic outputs. Their score
units are affinities, not calibrated questionnaire percentiles, so no raw-anchor
MAE/RMSE is assigned against0-1 outcomes.

| Raw semantic mean-r | PANDORA120 | Essays254 |
|---|---:|---:|
| Whole |.089134|.047748|
| Equal mean |.123284|.084216|
| Mass mean |.122453|.084159|
| Mass normalized |.122377|.084575|

An explicitly secondary, post-hoc descriptive bootstrap of all three differences
gives mass-vs-whole +.033319[-.026613,.093836] in PANDORA and
+.036411[.007868,.065364] in Essays. No parameter/sign was changed, and no formal
multiplicity-adjusted promotion is made. The three pooling variants share almost
the same inputs and are not three independent replications.

| Direct mass-pooled anchor | PANDORA r | Essays r |
|---|---:|---:|
| Extraversion |.055233|.067332|
| Neuroticism |.049591|.149817|
| Agreeableness |.201369|.130090|
| Conscientiousness |.224627|.184215|
| Openness |.081445|-.110658|

The negative Essays Openness result is retained, not sign-flipped. Thus the
positive aggregate does not mean that every pre-existing trait definition is
well measured by semantic affinity. The useful construction result is that
observation and reader must be evaluated jointly: changing context can help a
fixed semantic key while a learned criterion readout shows no aggregate gain.

## Reproduction, checks and remaining work

Encoding completed in881.33s in one bounded invocation after preparation. Eight
final heads and32 validation fits ran, with zero neural training. R1 preparation
stopped before inference because the installed tokenizer no longer offered a
legacy helper; explicit special-token assembly was checked against the native
template. R2 completed preparation. Before inference, R3 introduced length-sorted
batches; R2/R3 private material hashes are identical. No active run was restarted
on a polling timeout.

70 selected tests pass, including13 new context tests. Ruff and whitespace checks
pass. Independent arithmetic replays every15,225 input packet, original offsets,
token partition, all15,218 cached vector positions, source splits/transforms,
all40 trained and40 anchor criterion correlations, and per-piece sums. Maximum
pooling error5.56e-17; affine-head error2.45e-15; piece-attribution error2.34e-15.
This is a same-agent arithmetic audit, not an independent human replication.

Local ignored artifact root:
`results/grain_vector_mechanisms/all_plan_closure_v1/matched_context_r3/`.
Includes input contracts/addresses/coverage, cached vectors, per-domain models,
standalone coefficients, all source tuning, metrics/anchor links, private
per-piece contributions, source norms, geometry summary and audit.json.

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
"$PY" scripts/run_grain_matched_context.py
"$PY" scripts/audit_grain_matched_context.py
"$PY" -m pytest -q tests/test_grain_matched_context.py tests/test_grain_local_bilinear.py tests/test_grain_local_measure.py tests/test_grain_predecessors.py tests/test_grain_common_factors.py tests/test_grain_semantic_link.py
```

The default main command resumes only hash-matching cached inputs/results.
For a genuinely new run use a new `--output`; do not reuse this identifier with
changed code. Private data, vectors, heads and participant scores remain ignored.
No commits, push, lockbox opening, MEPS/bridge fitting or raw-text uploads occurred.

Native coverage is separately reconciled in
[the cache inventory](NATIVE_CONSTRUCTION_CACHE_INVENTORY.md):213 matching PANDORA
authors,43 held,3,195 token-exact raw-native rows. Historical cue-masked M29
all-nine-query tables also replay byte-identically. Neither substitutes for the
full current-material native/condition-path/psychological-interpretation work
still present in the all-plan ledger.
