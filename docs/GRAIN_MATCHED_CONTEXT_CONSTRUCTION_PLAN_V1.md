# Matched-Token Context Construction Reference

2026-09-13. All-plan C17 / V4 Study B, fixed semantic anchors: whole versus
predetermined pieces. Execute after local_bilinear_r2, not a rescue interaction
grid. Freeze this protocol before creating any new encoder or criterion scores.

## Question and decision

Does splitting the SAME available token sequence change the semantic object and
its local external readout? Existing full-material comment/chunk means remain a
separate reference. Their comparison to a capped document mixes context and
coverage; it must not be described as the effect of slicing alone.

Use all 624 PANDORA Big5-only and 2467 Essays authors from the preceding study,
same source/validation/held indices. No bridge/MEPS fitting, new lockbox, changes
to frozen GRAIN, external text upload, neural training or new downloads.

E5-large-v2 pinned revision f169b11e22de13617baa190a028a32f3493550b6 has a
512-token input limit. Inherit query prefix, masked mean and per-input L2 from
[the official model card](https://huggingface.co/intfloat/e5-large-v2), W68.
Its embedding objective is not a personality measurement model.

## Fixed observations

Tokenize original aggregated documents with the cached fast tokenizer. Verify
that prefix-token concatenation equals ordinary query-prefixed tokenization.
Keep first min(500,T) content wordpieces, with exact original character offsets.
This is the entire document only if T<=500. Preserve the omitted count otherwise.
The underlying uncased tokenizer remains unchanged; no personality-term removal.

Encode the prefix once as a whole and as consecutive <=128-wordpiece pieces.
Pieces may cut a wordpiece continuation: this is a token/position/context
intervention, not a claimed psychological utterance boundary. The concatenation
of content IDs must reproduce the whole content exactly. Every input gets the
same query prefix and special-token template; splitting repeats this framing.
No model-side truncation is permitted. Deduplicate exact input IDs in the cache.

Four predetermined matched-token arms:

1. whole: one normalized whole-prefix vector;
2. piece_equal: equal mean of normalized piece vectors;
3. piece_mass: mean weighted by content-token count;
4. piece_mass_unit: L2-normalized piece_mass.

Let z=sum_j (n_j/sum n) h_j, r=||z||, v=z/r, H=whole and D=frozen IPIP30
directions. Then exactly

```
H D - z D = (H-v) D + (1-r) v D
```

Report directional and radial terms and their cross-term, not only their squared
norms. They are not orthogonal. Separately compare equal and mass weighting.
One-piece authors provide an identity control. This algebra is Euclidean vector
decomposition, not a new theorem or identification of a psychological cause.
Local position reset, context restriction and repeated special tokens are
jointly changed; no claim to isolate attention alone.

## Frozen scoring and local construction

Compute all 30 fixed anchor contrasts, plus five averages of six keyed facets.
Report all five direct semantic criterion correlations and within-author
whole/piece score agreement; do not flip signs after viewing labels. Fit local
PCA64/scaler/Ridge for each vector arm using only the same source train/valid
partitions, alpha {1,10,100,1000}, mean-five-Pearson selection, source refit.
This matches the preceding study's semantic head procedure. Save source norms,
standalone affine beta/intercept and raw per-piece contribution accounting.

Report all traits: Pearson/Fisher95CI, Spearman, MAE/RMSE for trained heads and
AUC only for Essays. Direct affinity is not a calibrated 0-1 questionnaire score,
so no MAE/RMSE against questionnaire numbers is assigned to it. Report whole
versus each alternative with 1000 paired author bootstrap draws conditional on
fixed heads; no formal equivalence claim or held-driven parameter expansion.
Retain the previous full-material semantic-PCA reference separately, with exact
author matching. Report token coverage and T<=500 counts including held authors;
do not silently select only short documents for the primary analysis.

## Execution and acceptance

New script/module/tests, immutable input contract, resumable hash-checked local
encoder batches; nominal max 900s per invocation, pause at a completed batch.
Resume only the same confirmed-terminal command/contract, not on an observation
timeout. Measure runtime and remaining decision value before extending.
No remote processes or old queue are automatically resumed. Native LLM scores
are an independent remaining comparison, not satisfied by this encoder study.

Acceptance: exact token partition and coverage accounting for every author;
all four arms and targets retained; source-only transforms; cold fixed scoring
matches batch scoring; radial/directional identity and all headline metrics
independently replay. Deliver report and all-plan links whatever the ranking.
Stopping this comparison means it has answered this design question, not that
all context sizes, measures, datasets or Study B are complete.
