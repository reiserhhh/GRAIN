# NEXT40: source-fixed downstream reading of conditional attention

2026-09-10. NEXT39 supplies actual q/k/v, RoPE and a numerically exact last
attention value/key decomposition. Develop this interface instead of fitting
another unconstrained H-to-J regression or expanding the author panel.

## Question and exact scope

Can a source-fixed downstream cotangent, combined with a new author's current
query/key/value geometry, recover the local GRAIN action of the last attention
block without reading that author's output, score, gradient or cotangent?

The target is the last block's normalized BODY-input gradient of the fixed
digit-logit contrast and its declared GRAIN directions. NOT block16 gradients,
full scalar personality scores or human questionnaire accuracy.

## Frozen programs

- Reuse the same old4/new8, N/confirmation and NEXT39 contrast c. No new native
  calls, raw material collection, label reads or updated weights.
- Current numerical inputs: q,k,v,cos,sin and normalized body hidden/GRAIN
  coordinates. These are native forward activations, still an encoder dependency.
- Two g programs only: source mean from the same query; source mean pooled
  across both queries. No alignment, per-head rescaling, fitting or selection.
- Old4: hold one author out, average g from the other3 (or those3 x2 queries).
  New8: average the4 source authors (or4 x2); no new-author gradient enters.
- Each program computes a/o from current q/k/v and uses the exact frozen
  value+key formula. Value-only is the matched omission control, not a second fit.
- Four declared outputs per author/query: own/full, pooled/full, own/value,
  pooled/value. Retain key-only as a component, not a winner-search candidate.

## Runtime separation and metrics

First extract a restricted input-only packet and source-only g asset. Freeze
predictions in a separate process that cannot read the mixed native packets,
current g, logits, gradients or old labels. Test one denied-target read.
Then, in an evaluator, open native gradients to calculate:

- Field cosine and relative squared error, plus signed key/value cancellation.
- Four GRAIN tangent actions: r, RMSE, SSE/zero, amplitude and signs, by cohort,
  query and factor; pooled action r is not an author-only trait correlation.
- Factor-centered action association as a separate descriptive summary; do not
  combine old LOO and new fixed-source estimates or rank query winners by Y.
- Own versus pooled g and full versus value-only on the identical input/target.
- Source/current g geometry only after the prediction freeze, as evaluation.

Bound at120s numerical execution after verified loading, no hyperparameter grid
or neural fallback. Source mean construction is not gradient fitting on new
authors. Zero-call cache replay and small synthetic tests suffice for this packet.

## What it changes

Useful new-author action reading justifies retaining this exact-structure,
source-compiled local program and developing its earlier-layer inputs. If the
source g substitution loses useful reading, inspect its dependence on the final
query/context state; do not discard the verified value/key mechanism. Keep
usefulness local and report magnitude rather than inventing a personality gate.

Outputs: private `logic_source_cotangent_reader_v1/`, aggregate public report,
restricted-runtime test and dependency update. No release/lockbox/clinical
interpretation changes. This closes one computational dependency, not the goal.
