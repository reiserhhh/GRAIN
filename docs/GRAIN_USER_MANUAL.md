# GRAIN user manual: from a corpus to an interpretable estimate

English · [简体中文](GRAIN_USER_MANUAL_ZH.md) · [日本語](GRAIN_USER_MANUAL_JA.md)

For changing membership, irregular arrivals or overlapping readers, see [Open-stream operators: comparison and worked example](GRAIN_OPEN_STREAM_OPERATORS_G1_G6.md).

A writer proposes alternatives frequently. Does that mean they revise a plan when its premise changes? These are different studies. Frequency needs an opportunity denominator; revision needs a speaker, a correction, a response and a cutoff. GRAIN lets you keep the same material and ask both questions without treating one answer as the other. This book explains how to build a reference from scratch, apply it to new people, and choose among the static, path, interaction and inverse tools. It does not assume access to the original project's research data.

## 1. Choose a deliverable before choosing a model

| Decision | Observation unit | Deliverable |
|---|---|---|
| Place one new writer among peers | One compatible vector per independent author | Marginal percentiles and joint distance |
| Study preference versus response | Self-selected venue and expression under each available condition | Choice composition alongside conditional expression |
| Describe movement in an episode | Ordered events with intact episode boundaries | Node path, changes, order control |
| Check a proposition about an interaction | Addressed event and focal person's response by a cutoff | Source address, support, counterevidence or incomplete evidence |
| Combine two measurements | Two aligned views and independent calibration | A set of compatible common states and an incremental comparison |
| Build a measure for a new setting | Construction, calibration and application cohorts | A frozen scoring contract with examples and refusal conditions |

The four questions *where, what, who, when* refer to opportunity and selection, observed expression, position relative to a named population, and time or event order. A chosen topic can itself distinguish authors. Keep it alongside behavior within that topic rather than automatically removing its mean.

For repeated frozen scoring, see the [pipeline benchmarks and batch interface](GRAIN_PIPELINE_PERFORMANCE.md), including CPU/GPU transfer costs and numerical parity.

## 2. Run an entire small study offline

From the repository root, with Python 3.12:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock-main.txt
python -m examples.handbook_study /tmp/grain-first-study
```

Use a fresh output directory. Installation needs access to packages; the study itself needs no downloads, credentials, neural network or corpus. The [generator and analysis](../examples/handbook_study.py) create fictitious route notes. Twelve discovery writers supply English texts under two conditions and two views. Ten **other** writers define population ranks. Two new writers produce the same three planning and three exploratory sentences per episode; one switches once, the other repeatedly. The generator's descriptive labels do not go to the estimator.

| Inspect in order | Question to ask |
|---|---|
| `discovery.json`, `native-reference.json` | What words, projections, node map, conditions and A/B candidate loadings were learned, and from whom? |
| `calibration.json`, `calibration-units.json`, `chart.json` | Does each writer contribute one row? The prespecified axes are mean membership of the first node and mean Euclidean distance between neighboring node vectors. Compute within episodes, then weight episodes equally. |
| `target.json`, `target-units.json`, `positions.json` | Compare raw units before ranks. In the reference run the first-node ranks were both 0.15; step-length ranks were 0.40 and 1.00. A rank of 1.00 means above all ten reference values, not certainty about a person. |
| `target-families.json` | These are a separate output: features and cold projections on **discovered** static, dynamic and conditional axes. The two demonstration axes above were specified before looking at the writers. |

The portable result is the contrast: identical bags of events preserve static composition while the ordered path can differ. Node numbers or ranks may change if you rebuild the atlas with different dependencies. To reuse the existing coordinates instead:

```sh
python -m grain_core.liquid_reference_error locate --chart /tmp/grain-first-study/chart.json --input /tmp/grain-first-study/target-units.json --out /tmp/grain-first-study/recheck.json
```

## 3. Prepare real material and split it once

Keep pseudonymous author, conversation, speaker, addressee if known, original event ID, condition, carrier, timestamp/time zone, edit version and what was visible at the cutoff. Keep quotes attributed; a quote and an endorsement are different actions. Label assigned conditions separately from choices. Use authorized data locally; share synthetic examples or aggregates instead of original turns.

```json
{"language":"en","clock":"opportunity","episodes":[
  {"id":"case-01","author":"p17","condition":"detour","carrier":"route-note",
   "view":"A","events":["I checked the route on paper.","I tried another route."]}]}
```

This is one valid episode for an existing reader, not a training population. See the [dyadic schema](../examples/liquid_new_dialogue.json) for alternating speakers. Split authors into discovery, calibration and application first, then form disjoint A/B materials within an author. Hold out partners or time periods if the intended use demands it. The English text frontend counts retained slices, not elapsed seconds. A timestamp-free archive cannot support a claim about decay per second. Getting partial lexical coverage is not the same as distinguishing two meanings.

**Use your own corpus in the worked pipeline.** Put `discovery.json`, `calibration.json` and `target.json` in a private input directory in the schema above, then run `python -m examples.handbook_study /tmp/grain-my-study --inputs /path/to/prepared`. It reads those files instead of generating sentences. Use the same author namespace, with no author in two cohorts. For this two-axis recipe each episode needs at least two events; the chart needs more calibration people than axes and variation along both. A wholly uncovered application/calibration event stops the example rather than becoming a fictitious step. Its output **copies the source text**: keep the directory private. For questionnaire-defined coordinates or true elapsed time use the corresponding recipe below, not this teaching measure.

Add top-level `population` (who defines the norm) and `namespace` (the shared author-ID scheme) to `calibration.json`. To score later English text without access to the source corpus, retain just `native-reference.json` and `chart.json` in a private directory. The following invocation does **not** fit the atlas, factor loadings or chart:

```sh
python -m examples.handbook_study /tmp/grain-placed --frozen-from /tmp/grain-first-study --new /tmp/grain-first-study/target.json
```

## 4. Construct a coordinate system

Start with a whole event. Use fixed token widths to match text budgets; a source-fitted semantic cut to track boundaries; a rind cut only with the matching learned condition vocabulary. Treat each cut as its own observation operator. Compare operators on common held material, not by equating their third axes.

The native lexical atlas fits TF–IDF, truncated SVD and scaling on discovery texts. A hard tree assigns a leaf; a soft Gaussian mixture returns memberships plus an out-of-vocabulary channel. A **node** is an address; a **factor** combines patterns that recur across authors and views. Read central, borderline and contradictory slices when interpreting an address. Save the cut, vocabulary, node order, source digest, condition list, centers, scales and loadings together. Do not refit a vectorizer on new authors.

For the historical battery, the [ten-step construction record](WORKED_EXAMPLE_MANUAL.md) documents cohort tiers, slicing, planted controls, disjoint retest, condition choice, candidate discovery, blind item coding and external links. The [construct registry](CONSTRUCT_REGISTRY.md) describes the operational inventory; a component from a fresh mixture does not inherit a historical construct's name.

## 5. Preserve level, choice, response and residual

If $p_t$ is an event's vector, its mean is the observed level. Retain a separate vector of condition shares. To describe response under $c$, report the difference of the condition mean from that person's level **and its cell count**. An absent cell is missing, not zero. Dynamic descriptors summarize volatility, inertia, roughness, trend, change point and multiple scales without crossing sessions. Scramble the middle while keeping both ends, or compare with a simple smoother, before attributing a difference to temporal structure.

In the public native/liquid reader, the `hybrid` feature family is a vector of conditional node deviations. A historical static/motion shared subspace is a different construction. Report the definition with the name. Fit directions and their signs/scales on source A/B only; a new person gets a projection, not a new axis. Keep the raw descriptors if no candidate axis is supported. Test whether residual similarities follow length, venue, repeated wording or author identity before assigning a psychological interpretation. If interested in Big Five or MBTI, use independent author-level external criteria and report the actual association; author matching answers a different question.

## 6. Interpret a frozen score

A marginal percentile is the proportion of reference writers below the target plus half the proportion tied. Joint whitened distance measures departure from the center, not the direction or desirability of that departure. Supply named raw axes for interpretation. The historical `legacy_batch_relative_node_scores` changes with the comparison batch; it is not an individual norm.

Assess uncertainty by sampling the relevant independent people or whole episodes. Reference uncertainty, variability across occasions and bounds due to entirely missing opportunities have different meanings. For a repeat assessment freeze the old reference and log changes in exposure, volume and clock. To transfer a measure to another population, collect bridge material and check opportunity support, coordinate comparability and calibration separately. The [contextual construction handbook](GRAIN_CONTEXTUAL_CONSTRUCTION_HANDBOOK_V1.md) works two different target definitions through this process.

## 7. Trace persistence, diffusion and withdrawal

The optional liquid backend needs PyTorch and ncps (qualified version 1.0.1). A released reader for new English text can be built using your own source corpus; the [liquid manual](GRAIN_LIQUID_TOOL.md) also documents numeric-only remote execution.

```sh
python -m grain_core.liquid_instrument fit --input /tmp/grain-first-study/discovery.json --out /tmp/grain-first-study/ltc.json
python -m grain_core.liquid_instrument score --reference /tmp/grain-first-study/ltc.json --input examples/liquid_new_episode.json --out /tmp/grain-first-study/episode-scores.json
python -m grain_core.liquid_dialogue --reference /tmp/grain-first-study/ltc.json --input examples/liquid_new_dialogue.json --out /tmp/grain-first-study/dialogue-scores.json
```

Read four channels separately: observed memberships (`native`), fixed 0.5 retention (`fixed_half`), learned next-node distribution (`liquid`), and signed history against a one-event reset (`anchored_signed`, not a probability). Blanking source slice k while letting the clock advance gives a signed effect at each subsequent t (`missing_effect[k][t]`). Compare joint blanking with the sum of individual effects via `joint_minus_sum`. The middle-swap contrast tests order, not deletion. Each omission episode is capped at 128 slices.

For viscosity compare matched formats at matched opportunities. First check whether the input was written into the reader at all; then inspect retention and how much source variation survives. A rigid template can suppress both drift and individual differences. For stirring, record a baseline, introduce information, then withdraw it; compare branches with identical observed prefixes and a different continuation. Report distance to baseline and leave zero-denominator recovery ratios undefined. Reversing the order of two interventions is not withdrawing one. With actual timestamps, `liquid_event_time` separates an idle interval from the arriving event; [Riccati flows](GRAIN_RICCATI_AND_INTERPERSONAL_UPDATE.md) treat continuous time under held inputs. An inverse source-weight interval describes the model's compatible explanations, not a reconstructed person.

## 8. Put evidence back in an interaction

For “the speaker used the corrected premise”, identify *who was corrected, when, and who answered*. Run:

```sh
python -m grain_core.event_response examples/event_response.json
```

The partner corrects red to blue; the focal person's next turn uses blue. The address of the reply supports uptake at that cutoff. Future action is a different record. Separate support, refutation, inconclusive reply and no observed reply. An observed stratum with weight 0.75 and all support leaves the target rate in [0.75,1] if the rest was unobserved. This is an identification range, not a sampling interval. Explicit English patterns or independently precoded evidence are required for this reader; lexical similarity alone does not cast a vote.

For dyadic trajectories, align speaker and joint paths by the same slice fingerprints. Joint minus own-only removes both partner words and opportunities; to isolate sensitivity to prior inputs compare own and partner omissions **on the joint clock**. Keep the direction of a rating (who understood whom), not just the participants' names.

## 9. Separate common structure from convenient agreement

Two readings of one event can agree because of a common signal, aligned private signals or a shared reader bias. `liquid_common_range` gives compatible reconstructions under a declared private-norm budget; with no informative restriction the all-private explanation remains. Compare displacement between sets on the same chart, not similarly numbered factors on two different charts.

```sh
python -m grain_core.dual_view_increment examples/dual_view_increment.json
```

The worked intervals are [−0.4,2.0] after propagating two separate histories and [0.8,1.2] when the histories must admit a single common trajectory. Only the latter rules out zero. If the threshold decision is already settled, say so; do not manufacture an incremental factor. [Input and alternatives](GRAIN_EVENT_AND_VIEW_INCREMENT.md). Query-based text-to-trait evidence and actual neural activation substitution are further, distinct research procedures. Freeze source, reader, query and intervention; don't present node omissions as neural tracing. See the [mechanism supplement](paper/GRAIN_MECHANISM_SUPPLEMENT_EN_V1.md).

## 10. Make a new measure and hand it to another team

Define a bounded target (e.g. “revise after an addressed correction”), the population, opportunity denominator, positive/negative examples, and clock. Gather free-choice and independently repeated fixed conditions if both selection and conditional behavior matter. Deduplicate retellings. Build evidence items without the new authors; have a blind coder inspect same-topic counterexamples and different-wording positives. Freeze calibration, coverage rules and thresholds. Apply cold, then compare against the simplest usable rate, lexical reader or direct judgment at the same cutoff. Name a psychological construct only with appropriate external evidence. The [serialized bundle handbook](GRAIN_F16_PUBLIC_CALIBRATED_BUNDLE_HANDBOOK_V1.md) describes an implementation for declared calibrated targets, not an automatic free-text trait battery.

For recommendation, return neighboring materials and the situation or expression supporting the match; test usefulness separately from similarity. For market text, aggregate by eligible day and instrument, preserve overlap between full and filtered pools, and compare simple differences/lags before the decomposition. Next-open direction is a separate outcome. For interviews, return addressed observations and alternative readings; keep transcripts private. For another language, test paraphrase, negation, quoted speech and vocabulary coverage before transporting English coordinates.

## How earlier studies guide a choice today

The original registry separates first-person and directive register, a novelty/play vocabulary candidate reclassified as chiefly venue choice, and a negative-affect vocabulary rate retained at slice level rather than as a mood score. Read [the reasons for the reclassifications](CONSTRUCT_REGISTRY.md) before transporting the names. V6 separates level, conditional slopes and motion. V7 indexes a score by its observation operator, reference and error. V8 further separates an individual vector, a relation and a population field: a reliable relation between two people does not furnish a third person's score. To reuse *numerical* signed axes, use `liquid_axis_series` with the original axis order, coordinates and a strictly later application period. Replaying actual V7 assets with `liquid_frozen_bank` requires the original serialized banks; a newly fitted map is a different instrument.

Public dialogues illustrate two different mistakes. In RealPersonaChat, comprehension describes the rater's understanding of their partner and proactiveness is self-rated: check the questionnaire target before naming the outcome. In ESConv, separating support strategies did not establish emotional relief. Report the helper's observable action separately from the seeker's outcome. Picture stories can test an ordering hypothesis by perturbing sentence order; implicit tasks can test timing on trials. Neither licenses a single pooled personality norm. For observed numerical task paths use `liquid_task_paths`, not a fabricated sequence of words. [Cohort-specific results, controls and nulls](paper/GRAIN_LIQUID_RICCATI_EVIDENCE_PROVENANCE.md).

## 11. Reporting and troubleshooting

Report reference/version, population, input and basis digests, opportunity/clock, coverage, native units, repeatability or uncertainty, source addresses and a counterexample. Example: “Of five opportunities with an observed reply, three revisions were explicit, one reply retained the old plan and one was inconclusive; support on these five lies between 3/5 and 4/5.” This is more reusable than giving the writer a personality label.

| Symptom | Check | Remedy |
|---|---|---|
| Everyone has the same path | Token coverage and minimal word replacements at the input | Repair the frontend before fitting a new time constant. |
| Factor score is null | Paired source people, varying dimensions, retained axes, supported target cells | Report descriptors and missing support, not invented zeroes. |
| Score changes when an unrelated person is added | Batch versus frozen normalization | Use the saved independent chart. |
| Clock result equals simple smoothing | Endpoint-matched or fixed-half control | Report the simpler explanation. |
| Old bank cannot load | Coordinate and asset digests, local historical weights | Obtain authorized assets or build a **new** reference; do not silently substitute. |

Read the [theory manuscript](paper/GRAIN_THEORY_MANUSCRIPT_EN_V1.md) for individual, relational and population objects; the [liquid supplement](paper/GRAIN_LIQUID_RICCATI_SUPPLEMENT_EN_V2.md) and its [evidence provenance](paper/GRAIN_LIQUID_RICCATI_EVIDENCE_PROVENANCE.md) for the optional path instruments; the [V8 operations manual](V8_MANUAL.md), [claims ledger](CLAIMS_LEDGER.md), [data access](DATA_ACCESS.md) and [analyst guide](AI_ANALYST_GUIDE.md) for reproduction and handoff.
