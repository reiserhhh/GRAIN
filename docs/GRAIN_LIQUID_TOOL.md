# Frozen liquid reader for native GRAIN nodes

This optional research instrument keeps the original question, "where does this
expression sit in a reference group's node map?", and asks a second one on the
same words: "which earlier expressions are still changing the reader now?"
Existing GRAIN scoring entry points are unchanged. A node is an address for
expression, not the name of a psychological construct.

## Module Quick Reference

| Module (`grain_core.*`) | Mode | Input Schema / Coordinates | Primary Role |
|---|---|---|---|
| `grain_core.liquid_instrument` | CLI (`fit` / `score`) | Text episodes JSON (`examples/liquid_new_episode.json`) | Fits frozen native node atlas + LTC transition reader on source data and scores new episodes without re-fitting |
| `grain_core.liquid_dialogue` | CLI | Dyadic turn JSON (`examples/liquid_new_dialogue.json`) | Separates joint vs. own-only trajectories at matched slice fingerprints (`speaker_paths`) |
| `grain_core.liquid_common_range` | CLI / Python API | Two whitened numeric views + optional budget | Partial identification balls (`all-private` vs `minimum-private`) and pairwise `motion(left, right)` extremizers |
| `grain_core.liquid_event_time` | Python API | Numerical coordinate episodes + timestamps in seconds | Compares event-only, equal, measured, reversed, rescaled, and subdivided inter-event delays |
| `grain_core.liquid_task_paths` | Python API | Contiguous observed trial/story value matrices | Compares fixed-half, liquid, and endpoint-retention-matched readouts; conditional `pull_axis` and `crossing` |
| `grain_core.liquid_reference_error` | CLI (`fit` / `locate`) | Independent source/new numeric unit tables | Computes per-axis midrank percentile in `[0, 1]` and joint whitened distance |
| `grain_core.liquid_opportunity` | CLI (`fit` / `score`) | Condition-stratified node scores (V6 compat) | Estimates static, dynamic, and episode-reset opportunity slopes under `>=5` supported conditions |
| `grain_core.liquid_axis_series`, `liquid_frozen_bank`, `liquid_inverse`, `liquid_v6_author` | CLI / Python API | Signed numerical series / V7 frozen bundles | Reads pre-scored series, verified frozen banks (SHA-256 checked), and source-weight inverse intervals |
| `grain_core.riccati_mobius` | Python API | Irregular `dt`, multi-view series, affine maps | Exact constrained ZOH Riccati-Möbius flow, $O(\log T)$ parallel scan, dual-pathway episodic retrieval, dynamic LP bounds ([detailed manual](GRAIN_RICCATI_AND_INTERPERSONAL_UPDATE.md)) |

## Use it

**Unknown common and view-specific contributions.** Use
`grain_core.liquid_common_range.fit(paths, axes, split, basis, quantity)` to
freeze two whitened views on independent source subjects. Use a separate source
sample with `calibrate`, or pass no private budget to `estimate`. Pass two
numeric blocks in the fitted coordinate order (a whole block may be null).
The result includes an all-private reconstruction and a minimum-private
reconstruction of the same observation. An explicit joint private-norm budget
yields a ball, or an incompatibility. `motion(left, right)` returns the possible
displacements between two balls in the same frozen chart, with extremizers.
Rank is not a count of personality traits. To use a saved chart:

```sh
python -m grain_core.liquid_common_range --reference chart.json --input observations.json --out ranges.json
```

The input lists `axes`, `basis`, `quantity` (event, level or move), `budget`
(number or null), and `observations` (an ordered list of pairs of blocks in
that quantity; null steps break the path). The CLI accepts a bare chart or a
calibration bundle containing `chart`. The Project Persona English-text
front end computes the event, within-episode mean and consecutive difference
quantities separately; it does not request a questionnaire.

**Event time versus elapsed time.** For a frozen numeric task reader, call
`grain_core.liquid_event_time.score(doc, reader, calibration)`. The document has
`axes`, `basis`, `time_unit: "seconds"`, and episodes each with `id`, `values`
(original coordinate rows), and `times` (message times in seconds). The calibration
contains the reader digest and `seconds_per_unit`, estimated from source people only.
Use separate episodes for different speakers/sessions and unknown gaps. During the gap
the model receives a mask, then assimilates the arrival, never the reverse. It returns
the same frozen network's head outputs under event-only, equal, measured, reversed,
rescaled, and subdivided gaps and the effect of omitting the initial event. The time
ratio is an explicit instrument assumption; output mass is not a person's memory.

**Observed task episodes.** `grain_core.liquid_task_paths` reads observations in their
original units (e.g. a story's sentence coordinates or contiguous trial measurements).
`freeze_clock(examples, axes, basis, train)` takes triples of (source person, episode ID,
value matrix), selects at most one eligible path per calibration person, and binds the
numeric reader to an explicitly named coordinate system. The training callback must
return the frozen model parameters. `read_paths(paths, ref, axes, basis, gate)` takes
a numeric gate callback returning (event-by-cell retention matrix, official solver
comparison error); it returns fixed-half, liquid, and endpoint-retention-matched
readouts for each separate input path. Pass only ordered and observed runs.
`observed_episode` splits mixed or missing records and marks absent/withheld material;
`pull_axis` fits a source-only conditional reference, and `crossing` retains
first passage as well as the endpoint. Never interpret a missing reference as zero.
The matched clock is an offline control. With just two observations it exactly
reproduces the liquid observation mixture even if cell gates differ. The research
adapters for public PSE, evaluative priming trials and RealPersonaChat live in Project
Persona; no public corpus, private record or credential is bundled with this release.

**Population placement and error.** After aggregating one observation per
independent unit in compatible, named axes, fit a reference file using
`python -m grain_core.liquid_reference_error fit --input reference.json --out chart.json`.
Input keys: `population` (description), `namespace` (identity system), `basis`
(frozen coordinate digest), `axes` (ordered names), and `units` (objects with
string `id` and numerical `values`). For new units run `locate --input
new.json --chart chart.json --out positions.json` with the same module.
Output is a per-axis midrank percentile in [0,1] plus one joint whitened
distance. Input rows must already be independent units; this is not a raw
text scorer. Duplicate, calibration-overlap and incompatible-basis inputs
are rejected. Charts include marginal distributions and hashed identifiers;
keep person-level charts local unless disclosure is authorized.

`matched_endpoints(values, retention)` (episodes by events by observed axes,
and episodes by events by cells) compares the actual clock with per-cell
constant clocks preserving initial survival. `endpoint_weights(retention)`
returns the conditional history mixture for each endpoint, not the causal
effect of changing a token and thereby changing subsequent gates.

The companion Project Persona command `python -m scripts.score_grain_liquid_v6_author --help`
loads a frozen V6 author bank and processes ordered English comments. The
numeric backend `grain_core.liquid_v6_author` applies the five-fold opportunity
correction, static levels, opportunity slopes, dynamics and shared static/motion
axes. Use the discovery-oriented reference for new studies; use the separately
archived signed reference only when matching historical coordinates. Insufficient
run or condition support leaves a family missing, not zero. With
`scripts.grain_liquid_live_patch --reader hybrid`, the separate opted-in Linux
path offers real interventions in recurrent memory, attention and MLP. Tokenwise
routing/content swaps require aligned positions; gate/output swaps do not.

Four additional interfaces retain original numerical coordinates. All are optional:

- `python -m grain_core.liquid_frozen_bank --input EPISODES --reference WHOLE_READER --out NEW_FILE` applies the archived V7 factor and cross-view bundles. Requires the local historical assets in `results/v7_operator_smoke/initial_full_final` and `results/v7_operator_family/e4_v72_full_20260715_r3`, not just the Python package; the six serialized representations are checked against known SHA-256 values before loading. The input is the English opportunity-clock episode schema below. Independent episodes never share memory. Three outputs (original, fixed-half and LTC-gated) keep the old factor scales. Use `--transport ssh` for numeric-only remote inference.
- `python -m grain_core.liquid_axis_series fit --input CALIBRATION --out REF`, then `score --input HELD --reference REF --out NEW_FILE`: read existing signed factor scores rather than passing them through the English text frontend. A panel declares `coordinate_system`, ordered `axes`, `clock: observed_opportunity`, and `series: [{id,time,values}]` with finite rectangular arrays. Times must sort chronologically and held streams must begin strictly after calibration (use ISO dates or numeric times). Scores remain in the original units. This interface also supports `--transport ssh`.
- `python -m grain_core.liquid_inverse --help` describes the inverse for a scored episode with learned field paths and a digest-bound observation query. It returns source-weight intervals, not a unique latent person. Input alternatives refer to distinct trials, not simultaneous edits.
- In the companion Project Persona repo, `python -m scripts.grain_liquid_live_patch --help` links source deletion at the node level and actual same-material activation replacement in the project Linux Qwen reader. It requires explicit permission for the supplied text; it is not part of the local numeric-only path.

From the repository root, use a Python environment with the core scientific
dependencies (NumPy, pandas, SciPy, scikit-learn, threadpoolctl) and, on the
compute host, PyTorch and ncps (qualified with ncps 1.0.1).

```sh
python -m grain_core.liquid_instrument fit --input source.json --out ref.json
python -m grain_core.liquid_instrument score --reference ref.json --input examples/liquid_new_episode.json --out new_score.json
python -m grain_core.liquid_dialogue --reference ref.json --input examples/liquid_new_dialogue.json --out conversation_score.json
```

The examples are input shapes, not training corpora. Use a diverse English
reference with repeated events, multiple authors, conditions and two disjoint
views per author (A/B); with fewer than four paired source authors the
cross-view factor discovery step explicitly abstains. Increasing the number
of events by splitting one utterance does not create additional authors.
Choose the cut at fit time: whole utterance (default), fixed width
(`--width 16`), frozen reference semantic split (`--cut semantic`) or a
previously learned rind vocabulary (`--cut rind --rind-dir PATH`). Choose
soft mixture nodes (default) or `--method kmeans_tree`. No input at score
time updates cut boundaries, vocabulary, atlas, weights or loadings. Supply a
new output path. The ref includes the worker digest; retain the matching
worker source with its ref. Fit data and reference vocabularies may be sensitive.

For numeric-only remote computation add `--transport ssh` to each command and
set `GRAIN_LIQUID_HOST`, `GRAIN_LIQUID_PYTHON`, and optionally
`GRAIN_LIQUID_NCPS_PATH`. The node frontend remains on the caller; the remote
receives numeric node sequences/parameters. No hosted model service or
questionnaire is required. The separate Persona adapter can opt into sending
English words to the project encoder for nine additional fields.

## Read the output

| Output | What changed compared with a static node histogram? |
|---|---|
| `native`, `fixed_half`, `liquid`, `anchored_signed` | Observed node membership, a convex average of past memberships (0.5 retention per step), an official LTC's next-node distribution, and observation plus signed history relative to a one-event reset. Only the third forecasts the next membership. The signed channel is not a probability. |
| `families` / `feature_names` | Per author/view: level and choice, six path descriptors, and conditional deviations. Source-discovered loadings can be applied cold; missing support is reported. Also returns the historical batch-relative node comparator separately. |
| `missing_effect[k][t]` | Signed change in readout at slice t when only slice k is blanked and the opportunity clock still advances. Not a deletion of a person's real experience. |
| `source_node_to_readout_node[t]` / `joint_minus_sum` | Distribute individual omission effects over source node memberships; retain the difference between blanking everything jointly and summing single omissions. |
| `same_ends_middle_swap_rms` | Readout difference after permuting the middle with the first and final slice fixed. `null` if the path is too short. The bag of nodes is identical. |
| `contrasts` | Matched branches: keep write strength separate from normalized retention; for withdrawal also return distances from each branch to its pre-source position. Zero-denominator ratios are null. |
| Dyadic `speaker_paths` | For each named speaker, compare joint vs own-only readings at exactly matching utterance/slice fingerprints, then separate sensitivities to earlier own and partner turns **on the same joint clock**. Own-only removes both partner input and partner opportunities: the joint-minus-own difference is not a pure partner effect. |

The sample unit is a retained slice, not a second. Record elapsed real time
separately in any application that needs it. Novel out-of-vocabulary text is
retained as a coverage indicator, not assigned the nearest semantic meaning.
Episodes are independent; never concatenate people or unrelated conversations.
The full finite-omission tensor is limited to 128 slices per episode.
The core accepts observed words; it does not reconstruct hidden states inside
a language model or the speaker's personality vector.

## Reuse the released opportunity / response-slope estimator

For the historical V6 comparison, first score the original source texts with
the node reader frozen. Then fit the opportunity coordinates on that source
only. All four channels keep **the same native opportunity PCA coordinates**;
each channel has its own source condition means and A/B factor loadings.

```sh
python -m grain_core.liquid_instrument score --reference ref.json --input source.json --out source_score.json
python -m grain_core.liquid_opportunity fit --reference ref.json --input source_score.json --out opportunity_ref.json
python -m grain_core.liquid_opportunity score --reference opportunity_ref.json --input new_score.json --out slopes.json
```

The original three-axis design requires at least five supported conditions and
twenty distinct source authors per condition; participant slopes require five
supported, nonzero-coordinate conditions in that view. The input and reader
digests are checked. Missing design support returns its counts, not a fitted
slope. Each channel exports the historical static, dynamic and hybrid estimates
plus an episode-reset dynamic variant for independent episodes in the same
condition. The historical output keeps the release's condition-pooled clock.

## Exact Riccati-Möbius Flows & Dynamic Set Inversion

When continuous-time inter-event delays, long idle gaps, or cross-view
exponential rate separation matter beyond fixed discrete LTC steps, pair these
liquid trajectory readers with `grain_core.riccati_mobius`:

- **`ConstrainedRiccatiCell`**: Exact normalized $2\times 2$ ZOH Möbius flow
  preserving `[-B, B]` bounds, `dt = 0` identity, time-subdivision semigroup
  consistency, and exact $O(\log T)$ Kogge-Stone parallel prefix scans.
- **`DualPathwayRiccatiObserver`**: Pairs fast contractive decay over idle gaps
  with content-addressable cosine retrieval so early high-signal events remain
  addressable without violating `mask_first` causality.
- **`estimate_cross_view_lag1_rates`, `solve_axis_trajectory_lp`,
  `stack_transition_contrasts`**: Unbiased cross-view rate separation, HiGHS
  dynamic trajectory polytope bounds, and multi-window contrast stacking.

See the runnable practitioner guide in
[`GRAIN_RICCATI_AND_INTERPERSONAL_UPDATE.md`](GRAIN_RICCATI_AND_INTERPERSONAL_UPDATE.md)
and the formal/empirical synthesis in
[`paper/GRAIN_LIQUID_RICCATI_SUPPLEMENT_EN_V2.md`](paper/GRAIN_LIQUID_RICCATI_SUPPLEMENT_EN_V2.md).
