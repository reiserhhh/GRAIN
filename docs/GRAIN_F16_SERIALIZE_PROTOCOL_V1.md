# F16-R03-SERIALIZE protocol v1: public calibrated bundles and sealed cold scoring

Frozen 2026-09-26 before any new scoring. Worker: Claude (Opus 5.5) under
TASK-CLAUDE-GRAIN-FULL-LOOP-20260926. Scope is the finite serialization packet
named in `docs/GRAIN_CLAUDE_RESEARCH_LOOP_HANDOFF_20260926.md` (project persona repo).

## Mainline decision this packet changes

Before this packet, the completed transport estimators (numeric T1, numeric
dynamic T2/T3, changed text reader T1/T2/T3) existed only as research runner
state. A downstream user could not receive one authenticated object and
reproduce a transported score, its full calibration uncertainty, its support
and its explanation without the research archive. The text payloads also carry
stale inherited reader fields. This packet decides whether the completed
calibration mixtures can be published as self-contained instruments, and which
reference positions they may legitimately report.

## Reused, not rerun

- Source models: `F16/t1_source_full_r1/<seed>/bundle.json` (numeric T1),
  `F16/t2_cold_full_r1/<seed>/bundle.json` (numeric dynamic), text engine
  payloads `F16/text_transport_full_r1/scores/<seed>/{T1,dynamic}_primary_bundle.json`.
- Calibrations: `F16/t1_transport_full_r1/<seed>/<world>_<mode>.json`,
  `F16/dynamic_transport_full_r1/<seed>/<world>_<mode>.json`,
  `F16/text_transport_full_r1/channels.json` and `missing_anchor_controls.json`.
- References: numeric T1 source reference in the T1 source bundle; numeric
  dynamic `dynamic_transport_full_r1/<seed>/reference.json`.
- Canonical application outputs used only as replay targets:
  `t1_transport_full_r1/<seed>/application_scores_before_truth.json`,
  `dynamic_transport_full_r1/<seed>/<world>_<mode>_{mixture,plugin}.npz`,
  `text_transport_full_r1/scores/<seed>/{T1,dynamic}_changed_pilot_{mixture,plugin}.json`.
- No source refit, no pilot refit (calibrators are copied and re-derived only
  as a provenance check), no neural reading, no API, no truth file opened.

## Bundle format `F16_public_calibrated_bundle_v1`

One bundle = one target x route x changed-reader world x estimator x seed.
Blocks (all required; canonical JSON; SHA-256 of the canonical body is the
bundle identity):

1. `target`: target id, object class, estimand, unit, time contract
   (T1 t<=31 four conditions; T2/T3 cutoff 47), condition plan (T3 future
   +1,-1,+1,-1).
2. `observation`: route (`numeric` / `controlled_text`), reader definition and a
   recomputed `reader_id` over definition + readout cache or channel world.
3. `source`: engine formula version, embedded source models, provenance file
   hashes, component count.
4. `calibration`: kind (`numeric_affine_pilot` / `categorical_channel_pilot`),
   data provenance (pilot file hash, roles, partition, rows), plugin, all draws
   (numeric: 200 bootstrap calibrators; text: 200 Jeffreys-Dirichlet channel
   draws + counts + generator seed), status.
5. `integration`: `cartesian_source_by_pilot_exact_v1` (numeric T1),
   `cartesian_source_by_pilot_eta_gauss_hermite64_v1` (numeric dynamic) or
   `paired_source_channel_monte_carlo_qmc_v1` (text: component b pairs source b
   with channel draw b; 200 pairs, not 40,000). Numerical diagnostics recorded,
   not claimed zero.
6. `reference_policy`: one of `INHERITED_COMMON_UNIT` (numeric T1: calibrated
   observations map into the source unit and the identical target function
   scored the source reference), `FRESH_SOURCE_SCORED` (numeric dynamic
   conditional-Gaussian reference; text references newly scored in this packet
   under the changed reader and the bundle's own estimator), or
   `RAW_UNIT_ONLY`. Historical references are carried only under
   `historical_positions` and never used for `reference_score`. Application
   cohorts can never define a reference.
7. `known_validation`: the archived finite SIM metrics for exactly this
   seed/world/estimator/target including failed coverage gates, plus
   retained failures (cubic, missing anchors) for the route.
8. `legacy_reconciliation` (text only): every inherited inner field whose value
   is stale for the changed reader is moved out of the engine payload with its
   old value hash and reason; score-determining engine fields are proven
   byte-identical to the frozen primary payload.
9. `code_manifest`: hashes of the serializer and all engine modules.

## Build refusals (actual attempts, retained)

- `MISSING_ANCHOR`: text missing pole/qualification controls (4 per packet),
  numeric missing-condition anchors (4 worlds x 3 seeds x 2 routes).
- `NONMONOTONE_INVERSE`: nonlinear cubic calibrations (3 seeds x 2 routes).
- `APPLICATION_CALIBRATION_NOT_OPERATIONAL`: text known-pool law, oracle labels.
- `APPLICATION_REFERENCE_FORBIDDEN`: any reference derived from application inputs.
- Numeric uncalibrated/known-inverse arms are information references, not
  public estimators, and are not built.

## Sealed cold scoring

`python -m grain_research.gptpro_foundation_v3.f16_serialize score --bundle B
--sha256 H --input I --output O` runs in a fresh process with a Python audit
hook installed before the bundle is read. Opening any path outside the Python
installation, the repository code package, the bundle, the input and the output
raises; sockets and subprocesses raise. Every opened path is logged into the
cold output. The input carries only the observation envelope; a request whose
contract (bundle id, target, reader id, unit, clock, plan) differs is rejected
per row with a code; truth fields are rejected.

Outputs: score, variance, variance components, 95% moment-normal interval,
support block, reference block (policy + position), explanation block:
- numeric T1: exact additive event contributions plus intercept (identity checked);
- numeric dynamic: posterior-weight-conditional linear attribution (exact
  identity at the observed node weights; not a deletion effect);
- text: category counts, active materials, stale annotations; on request exact
  deletion effects (T1 per condition/category cell; dynamic per event).

## Checks and frozen tolerances

1. Cold replay equals canonical archived application outputs: max abs difference
   <= 1e-10 for score, variance and every stored component.
2. Independent formulas: numeric T1 event-level Woodbury-free Gaussian
   conditioning for all application authors (<= 1e-9); numeric dynamic direct
   joint-covariance conditioning per eta node, first/last author per bundle
   (<= 1e-6 on mean and total variance).
3. Provenance re-derivation: numeric calibrators refit from pilot rows and
   bootstrap indices; text counts recounted from `reader_scores.csv`; text
   mixture draws regenerated from counts and declared seed. Exact equality.
4. Reference replays: numeric T1 source reference recomputed by the kernel
   formula from source observations; numeric dynamic reference recomputed;
   text fresh references rescored in a sealed cold process. <= 1e-10.
5. Rejections: every refusal above plus wrong reader, wrong target, wrong
   bundle version, tampered draw, restored stale inner reader hash, truth field,
   unregistered text, incomplete condition support, off-clock/missing step,
   undeclared file read inside the sealed process.

## Fresh text source references (new computation)

For each seed, the 300 archived source authors (F04 L for T1, F05 core for
T2/T3) are rendered onto the 80 source-role materials with the same rendering
rule as application inputs and a separate stream
`SeedSequence([seed, 172, dynamic])`; they are scored by the exact public
engine payload of each estimator (mixture/plugin). The mean and SD (ddof=1) of
source scores are frozen. They describe source authors under this reader on
source-role controlled materials; they are not population norms, not human
norms, and not application-derived.

## Non-claims

Serialization does not make any estimator valid for natural language or human
personality; all known failures remain attached. Passing this packet does not
promote any parent card (F12-01..05, F16-02/03/04/06).
