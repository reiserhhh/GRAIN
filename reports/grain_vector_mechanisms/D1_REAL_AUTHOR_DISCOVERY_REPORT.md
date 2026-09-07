# D1: real-author personality-query discovery

2026-09-07. Completed cache-only exploration under the user's discovery-first rule.

## Main finding

**The real-author material contains a concrete native personality-query signal to
trace.** On the paired Big5 subset, native query scores have mean Pearson r=.1993;
the strongest mean statistical-head comparator here is E5 at .1372. On MBTI,
native T/F has AUC=.6600, and its score correlates .5127 with the E5 head.
This gives the next mechanism experiment two useful objects: a shared T/F pathway
and the additional Big5 response accessed by the native query.

## Material and score definition

- Copied all47 completed Windows shards,2,679 cached authors; no queue restart.
- Snapshot membership and file hashes were fixed before outcome joins.
- Included213 Big5-only and2,328 MBTI-only reference-excluded source authors.
  Of the available records,99 bridge users and39 source reference users were not
  evaluated. No bridge labels or MEPS questionnaires were read.
- Frozen native Qwen prompts; three12-comment views per author, identical raw,
  cue-mask and placebo-mask material arms across the comparison.
- Native Big5 score: expected digit/9 from the candidate-token distribution,
  averaged over all three views. Native MBTI: the fixed-positive candidate
  probability averaged over those views. No generated explanation was scored.
- Statistical comparators are existing source-trained, official-fold OOF heads,
  evaluated on exactly these same authors. No new head fitting, calibration,
  axis flipping, user selection by score, or new model forwards.

This is an availability-fixed discovery cohort, not the old full128-shard
benchmark. Full-source estimates such as E5 mean r=.202 use a different population.
The comparison below is deliberately paired; point differences guide exploration.

## Paired raw-text results

| Target | Authors | Native query | BGE head | E5 head | Qwen block10 head |
|---|---:|---:|---:|---:|---:|
| Big5 mean Pearson r |213|.1993|.1110|.1372|.0993|
| EI AUC |2328|.5218|.6235|.6389|.6286|
| SN AUC |2328|.5963|.6753|.6744|.6651|
| TF AUC |2328|.6600|.7532|.7542|.7432|
| JP AUC |2328|.5003|.6056|.6076|.5970|

| Big5 trait | Native r | Fisher95% interval | E5 r |
|---|---:|---:|---:|
| Extraversion |.2671|[.1377,.3877]|.2776|
| Neuroticism |.2244|[.0928,.3483]|.0417|
| Agreeableness |.1994|[.0668,.3251]|.0797|
| Conscientiousness |.1676|[.0340,.2954]|.0434|
| Openness |.1379|[.0036,.2674]|.2435|

Intervals describe each correlation, not the difference between methods. All
five traits and all four axes are retained. EI/SN/JP fixed-threshold native
balanced accuracy is .500; TF is .6065. Continuous scores make the detected
ordering visible without a threshold-calibration detour.

## Mechanism leads

1. **Shared route:** native TF and E5 scores correlate r=.5127 on2,328 authors;
   BGE/native r=.4969. Both relate to the recorded TF labels. Start evidence and
   vector-path tracing here, where there is a sizeable shared observable signal.
2. **Different access:** native neuroticism r=.2244 versus E5 r=.0417, while the
   two scores correlate only .0697. Conscientiousness has the same qualitative
   pattern (.1676 versus .0434; query/head r=.0852). Use neuroticism as the
   secondary case for comparing query-conditioned access with pooled-vector access.
3. **Trait-dependent commonality:** agreeableness query/E5 r=.5749, extraversion
   .3388 and openness .3574. Do not force one universal readout for all traits.

The useful mathematical starting point is:

    s_it = g_t(H(x_i, q_t))
    b_it = f_t(pool(E(x_i)))

Both are observed functions of real-author material. D2 asks which evidence and
intermediate vector changes drive s_it, and which changes also track b_it. Direct
and statistical observers differ in training and processing, so the difference
motivates this mechanism experiment rather than identifying its cause by itself.

## Existing mask contrasts

| Arm | Native Big5 mean r | Native TF AUC |
|---|---:|---:|
| raw |.1993|.6600|
| cue_mask |.1671|.6591|
| placebo_mask |.1925|.6647|

TF remains an attractive tracing target across all three saved arms. Big5
query scores respond more to the cue-mask arm. The mask effects are recorded
here as leads; no new mask search or wording optimization was performed.

## Next mainline experiment

**D2 priority: TF shared path, neuroticism contrasting path.** On a bounded new
source-author panel, ask the fixed trait query, record query-conditioned states,
and compare original evidence with query-selected versus count/length-matched
other evidence. Use a small intervention set to see which vector changes move
the native score. Apply independent frozen GRAIN neighborhoods to pool the
response profiles. Keep discovery authors separate from later hypothesis checks.
Do not return to P1U, generic code-invariance sweeps, or all-layer search.

## Reproduction and artifacts

Entry point: `scripts/analyze_grain_author_discovery.py`, first `--phase snapshot`,
then `--phase analyze`, using these paths:

```text
--queue-dir results/grain_vector_mechanisms/d1_direct_snapshot_47
--primary-dir results/grain_vector_mechanisms/vm05_full_neural_source_heads_r1
--material-dir results/grain_vector_mechanisms/vm05_direct_trait_material_r1
--output-dir results/grain_vector_mechanisms/d1_author_discovery_r1
```

Output: `headline.csv`, `metrics.csv` (108 rows), `query_head_links.csv`,
`populations.csv`, `native_quality.csv`, `summary.json`; private membership and
prediction files remain in ignored results/. Snapshot SHA256:
`ab7e65442e96b2b34fb61cb2d8104af73af4b60a172c1aed6cbc5361975d36c0`.

Validation:14 targeted tests passed; Ruff passed. This packet used zero new
model forwards and zero new fitted heads. The only network work was copying
the completed cache from the paused Windows queue.
