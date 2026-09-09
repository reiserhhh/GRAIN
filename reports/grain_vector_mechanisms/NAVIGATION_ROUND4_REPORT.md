# GRAIN NAV round4: author reference, groups, caching and portable programs

2026-09-09. NAV goal V6 remains **ACTIVE**. NAV06/08/09/11 now have actual
experiments and delivered artifacts. The remaining scientific development is
conditional state/projection construction on natural materials, not unfinished
packaging or another human-label accuracy search.

## Main findings and decisions

1. Fixed reference-author scoring is batch invariant: maximum change0.0 when
   adding100 extreme rows or scoring authors individually. The instrument no
   longer changes a person's coordinate merely because other people joined.
2. Existing GRAIN functional-score groups do not currently justify automatic
   path-sharing weights. The primary native response grouping gain is3.4%,
   author-permutation p=.2784; the secondary nine-coordinate grouping is worse
   than unmatched grouping. Keep individual/context-dependent operations.
3. Identical-prefix caching passes pre-fixed numerical tolerances and gives
   median1.372x speedup. Memory rises, so this is opt-in, not a service migration.
4. Four self-contained candidate programs now take raw excerpts, not P/Q caches.
   All60 replay outputs match prior frozen computations to at most2.95e-15.
   The token-only program imports neither Torch nor Transformers. Other programs
   execute exactly four extracted blocks, without the teacher or LM head.

This is a measurable reduction of runtime dependence. It does not turn the
negative-transfer state constructor into a better predictor; earlier T/F and
state external effects remain the same observations, not new replications.

## NAV06: do GRAIN-defined groups share evidence-response pathways?

The existing24-author/two-view panel retains12 reference and12 evaluation
authors. Groups are fitted on reference authors' **view1** functional profiles,
then assigned to evaluation view1. Responses come from evaluation **view0**,
the12-author/eight-condition natural-edit packet. Exact excerpt overlap across
views was checked in the existing hashed overlap audit: zero for all24 authors.

The primary partition uses three levels: T/F, neuroticism, confirmation-seeking.
The secondary uses these levels plus six prior configuration components.
These are existing GRAIN functional profiles, not the older SUICA-19 item bank.
KMeans K=2, seed20260909, n_init10 is frozen before reading response outcomes;
cluster names G0/G1 are not psychological types. Held group counts are6/6 and4/8.
No human labels or MEPS data enter fitting.

Each author has12 response coordinates: three new queries times
actor/endorsement/time/removal contrasts. We compare average squared distance
within groups to distance between groups:

~~~text
D_uv = mean_j (delta_f_uj - delta_f_vj)^2
gain = mean_between(D) - mean_within(D)
relative_gain = gain / mean_all_pairs(D)
~~~

Permutation moves whole author group labels4999 times; the66 pairwise distances
are not treated as independent observations. Distance-correlation tests permute
whole author identities.

| Reader | Partition | Relative grouping gain | One-sided permutation p |
|---|---|---:|---:|
| Native | three levels | .0342 | .2784 |
| Native | nine coordinates | -.1396 | .7470 |
| Synthetic function | three levels | -.0062 | .3810 |
| Synthetic function | nine coordinates | .1864 | .1484 |

The secondary synthetic distance association is Spearman .2343,p=.1526;
primary native distance association is .1026,p=.3942. This small packet does
not motivate adding a group multiplier to the main formula. It tests these
particular old profiles and partitions, not the impossibility of all shared
group mechanisms. Earlier cross-query intervention findings remain intact.

## NAV08: fixed reference positions and their uncertainty

The panel contains144 actual P/Q units and432 native/real-function/synthetic-
function scores.72 P/Q units were reused only after exact token, mask, identity
and artifact-hash checks;72 new native passes took104.45seconds. Old T/F native
candidate mass was not recorded and remains unknown, not imputed to1.

Do not confuse the two kinds of reference:
- Function construction uses the old two real or two synthetic texts per query.
- Author coordinates use12 different fixed reference authors, averaging their
  two views into one row per author before calculating mean and standard deviation.

~~~text
a_uq = mean_over_views f_q(text_uv)
mu_q = mean_reference_authors a_uq
sigma_q = population_std_reference_authors a_uq
position_uq = (a_uq - mu_q) / sigma_q
~~~

These reference moments are never re-estimated on the evaluation batch.
The0.0 batch error is an implementation check of this algebra, not evidence
that the underlying score is a temporally stable personality trait.

Bootstrap2000 samples of reference authors, keeping evaluation authors fixed.
Mean95% position interval widths are1.35-1.67 z units across the nine conditions.
This is a usable **local research reference**, not a population/clinical norm.

Two-view correlations among12 evaluation authors:

| Reader | E | T/F | Low state |
|---|---:|---:|---:|
| Original native score | -.2053 | -.2978 | .4669 |
| Real-reference function | -.0399 | .0766 | .5316 |
| Synthetic-reference function | -.0147 | -.2689 | .4560 |

This panel's low-state ordering survives the change of text sample better than
E/T/F. The two views are disjoint text samples, not an observed cross-day design;
their relation does not reclassify a state query as a personality trait.
Fixed positions and textual sampling uncertainty are separate problems.

## NAV09: exact-prefix native cache benchmark

Two fixed real-author cases and one controlled-world case, three queries,
three repetitions. The E and T/F questions share their entire pre-question
context; the low-state query uses a different system prompt and takes an
uncached fallback. No target-specific prefix is forcibly shared.

Both arms use float32/eager on the same local frozen Qwen observer, in separate
processes with one excluded full-pass warmup. Timing includes prefix construction,
cache cloning, suffix inference and P/Q/native copies, not just cheap suffix
execution. The parent KV cache is cloned per branch and checked for mutation.
Source inspiration: exact-prefix reuse in [SGLang/RadixAttention](https://arxiv.org/abs/2312.07104);
this is a small native-cache implementation, not a replication of that paper's speed claims.

| Metric | Uncached | Cached |
|---|---:|---:|
| Timed forward passes |27|36|
| Total timed inference seconds |29.397|21.979|
| Peak process RSS, decimal GB |6.665|9.859|
| Maximum sampled MPS allocation, decimal GB |6.175|6.401|

Median paired speedup=1.37245; long cases around1.25-1.39x, short case1.12-1.13x.
More forward invocations still process fewer tokens because the long context is
not repeated. RSS includes loading/warmup and is not GPU VRAM; sampled MPS memory
is explicitly not a measured instantaneous peak.

Maximum P/Q absolute difference=5.34e-5; maximum score difference=8.44e-6.
Pre-fixed tolerances were1e-4 and1e-5; minimum speedup1.10. All pass.
Decision: optional exact-prefix cache for compatible multi-query research,
uncached path retained. This remains full-LLM computation and is not NAV11 evidence.

## NAV11: raw-text runtime dependency closure

The local bundle includes a frozen tokenizer, query-rendering recipes, synthetic
derivative charts, a token table, optional four-block weights and compiled state
operators. No private author excerpts or human questionnaire data are inside
the transferable bundle. Replay inputs stay in an adjacent ignored private file.

The four runtime selectors are:

| Selector | Actual arithmetic | Transformer layers | Teacher P/Q |
|---|---|---:|---|
| token | token table + fixed semantic/order alignment + two derivative projections |0|absent|
| context4 | four contextual blocks + two construction charts, mean pooling |4|absent|
| state1 | four blocks + directly compiled linear-kernel state reader |4|absent|
| state2 | four blocks + directly compiled quadratic-kernel state reader |4|absent|

The state programs retain their previously reported poor natural transfer;
they are exported for reproducibility, not recommended over useful older candidates.
None outputs calibrated human personality probabilities.

Raw-text replay uses three held scenes and two evaluation authors, all three
queries:15 rows per program. It runs the bundle CLI from a temporary external
working directory. A Python file-access audit denies the two source repositories
except the bundle, explicit replay input/output and installed virtual environment.
Network-dependent Hugging Face loading is disabled. No source-access attempts
were blocked because none occurred.

| Program | Max absolute error against prior frozen score | Dependency boundary |
|---|---:|---|
| token |5.55e-17|PASS; Torch/Transformers both absent|
| context4 |5.55e-17|PASS; exactly four blocks|
| state1 |2.83e-15|PASS; no generated full P/Q needed|
| state2 |2.94e-15|PASS; no generated full P/Q needed|

The1.8GiB local combined bundle retains pretrained knowledge; Transformer-free
does not mean learning-free. Publication/distribution of weights is separate
from this local export and remains subject to the source model's license.
[Build, equations and usage](../../docs/GRAIN_NAVIGATION_RUNTIME_BUILD_V1.md).

## Artifacts and next decision

All private numerical artifacts are below ignored
results/grain_vector_mechanisms/navigation_v1/:

| Packet | Directory | Key files |
|---|---|---|
|06|NAV-06/author_groups_r1|group_freeze.json; analysis/group_condition_effects.csv|
|08|NAV-08/author_panel_r1|analysis/author_norms.json; reference_sensitivity.csv; batch_invariance.json|
|09|NAV-09/shared_prefix_r1|summary.json; paired_timings.csv; numerical_equivalence.csv|
|11|NAV-11/runtime_export_r1|bundle/manifest.json; replay_*/summary.json; comparison.csv|

Code supplies reusable group statistics, checked raw-text rendering, cache
branching and source-denied export replay. Eight new focused tests passed;
the full suite finished with2759passed,3skipped,129warnings in257.46seconds.

**Next mainline experiment:** address the identified residual-space coverage
gap using only the already allowed, separate non-MEPS reference authors.
Compare a frozen synthetic-only operator with one bounded source-span expansion;
freeze the extension before held natural responses or external criteria.
Keep useful token-only T/F and shallow-state branches intact. This develops the
missing conditional computation rather than restarting a personality-label grid.
