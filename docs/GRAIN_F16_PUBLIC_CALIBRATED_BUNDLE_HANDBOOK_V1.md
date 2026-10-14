# Handbook: public calibrated F16 bundles (`F16_public_calibrated_bundle_v1`)

Scope: how to build, check and apply a self-contained calibrated scoring bundle
for the three F16 targets after a reader change. Mathematical production on
declared SIM / controlled-text worlds; not a validated psychological instrument.
Protocol: `docs/GRAIN_F16_SERIALIZE_PROTOCOL_V1.md`. Module:
`grain_research/gptpro_foundation_v3/f16_serialize.py`.

## 1. What a bundle is

One JSON object per target x route x changed-reader world x estimator x seed.
Its identity (`bundle_id`) is the SHA-256 of the canonical body; the file bytes
also need an externally supplied trusted SHA-256 at scoring time.

| Block | Holds | Checked at load by |
|---|---|---|
| `target` | estimand, unit, fixed clock (T1=31, T2/T3=47), condition plan | exact re-derivation |
| `observation` | reader definition; for text also the readout-cache digest | `reader_id` recomputed |
| `source` | source-fit provenance hashes, component count | informational |
| `calibration` | pilot provenance, plugin, every draw | refit (numeric) / regeneration (text) |
| `integration` | estimator rule and component counts | rule name must match route |
| `reference_policy` | kind, mu, sigma, scope, historical positions | kind per route; never application-derived |
| `known_validation` | archived finite metrics incl. failed gates | carried, not re-scored |
| `legacy_reconciliation` | text only: stale inherited fields and why | stale fields must be absent from engine |
| `engine` | the unchanged numerical engine payload | engine's own validator |
| `code_manifest` | serializer + engine code hashes | must equal running code |

## 2. Two estimators that must not be conflated

- **Numeric affine transport** (`cartesian_source_by_pilot_*`): every one of the
  200 source components is paired with every one of the 200 pilot bootstrap
  calibrators (40,000). Score = grand mean; variance = conditional (or
  state/future + personal) + source + pilot. The bundle embeds the 60 pilot fit
  pairs and bootstrap indices so every calibrator is refit at load (tolerance
  1e-11).
- **Categorical text channel** (`paired_source_channel_monte_carlo_*`): the b-th
  source component is paired with the b-th Jeffreys-Dirichlet channel draw (200
  pairs). This is Monte Carlo integration of the independent product law, not
  the 40,000-term Cartesian sum; source and reader spread are one combined
  component. Draws are regenerated at load from pilot counts + declared seed.
- `plugin` bundles are point calibrations: no calibration uncertainty is
  propagated (`pilot` component is 0 / channel draws identical).

## 3. Reference policy (which "position" a score may report)

| Route | Kind | Why |
|---|---|---|
| numeric T1 | `INHERITED_COMMON_UNIT` | calibration maps changed-reader observations into the original response unit; the same four-cell target function scored the F04 source reference (replayed exactly) |
| numeric T2/T3 | `FRESH_SOURCE_SCORED` | the conditional-Gaussian 64-node integration is a new estimator version; its source reference was rescored, the old Laplace reference is not reused |
| text T1/T2/T3 | `FRESH_SOURCE_SCORED` | the estimand is unchanged, but the changed reader changes the scoring function (observation law) and hence the source score distribution; 300 source authors are rendered on source-role materials, read by the changed reader and scored by the bundle's own engine and estimator |

Historical references appear only under `historical_positions` and never define
`reference.position`. An application batch never defines a reference. A
reference position is a location within declared simulated source authors, not
a population, human or clinical norm.

Raw-unit scores never change with the reference; a reference only changes the reported position. Keep raw-unit comparisons and position comparisons in separate tables.

## 4. The legacy reader-hash problem and its resolution

The text-transport primary payloads were built by deep-copying the old reader
bundle and replacing materials/channels. They therefore still carried inner
fields describing the OLD reader: `reader_hash` (hash of old materials + old
channel draws), `reader_bundle_sha256`, `channel_counts`, `calibration`, and
old-bank references. The public format:

1. authenticates the frozen primary payload by its outer run manifest hash;
2. proves every score-determining field is byte-identical to that payload;
3. recomputes the legacy formula and records that the inner `reader_hash`
   identifies the old reader and not the changed one;
4. moves all stale fields into `legacy_reconciliation.moved_out` with value hash
   and reason, and refuses any bundle whose engine still contains them;
5. binds the changed reader by a new `reader_id` over its definition (pinned NLI
   model/revision, queries, pole/qualification formulas, category alphabet,
   engine slot alias) and the digest of the actual readout cache.

## 5. Building

```bash
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY="/Volumes/mobile3/projects/project persona/.venv/bin/python"
$PY scripts/run_grain_gptpro_f16_serialize.py --output <NEW_DIR> --stage prepare
$PY scripts/run_grain_gptpro_f16_serialize.py --output <NEW_DIR> --stage build
```

`build` refuses (and records in `refusals.json`): `MISSING_ANCHOR`,
`NONMONOTONE_INVERSE`, `INCOMPLETE_CALIBRATION_DRAWS`,
`APPLICATION_CALIBRATION_NOT_OPERATIONAL` (known-pool, oracle),
`APPLICATION_REFERENCE_FORBIDDEN`. Refusals are results, not errors to retry.

## 6. Applying (sealed cold scoring)

```bash
$PY -m grain_research.gptpro_foundation_v3.f16_serialize score \
  --bundle B.json --sha256 <trusted> --input requests.jsonl --output scores.jsonl
```

Each request line is `{"contract": {...}, "data": {...}, "explain": "summary"|"deletion"}`.
Build the contract with `f16_serialize.make_request(bundle, data)`; data carries
only observations:

- numeric T1: `{"author_id", "records": [{"t","z","y"}]}`, t<=31, all four conditions;
- numeric T2/T3: one record per step 0..47, z in {-1,1};
- text: the registered engine history (`author_id,target_time,history,world_contract`).

The process installs an audit hook before reading anything: it can open only
the Python installation, the code package, the bundle, the input and its own
outputs. Any other open, socket or subprocess raises. `<output>.manifest.json`
lists every opened path.

Row-level rejections (batch continues): `WRONG_VERSION`, `WRONG_TARGET`,
`WRONG_READER`, `WRONG_UNIT`, `WRONG_CLOCK`, `WRONG_CONDITION_PLAN`,
`HIDDEN_OR_UNDECLARED_FIELD`, `INCOMPLETE_CONDITION_SUPPORT`,
`UNSUPPORTED_CONDITION`, `INCOMPLETE_CLOCK_SUPPORT`, `OUTSIDE_CALIBRATED_SUPPORT`,
`UNREGISTERED_TEXT`, `INVALID_HISTORY`, `PRIOR_ONLY_NOT_TRANSPORTED`.

## 7. Reading an output row

`score`, `variance`, `lower/upper` (moment-normal 95%), `components`,
`support`, `reference` (policy + position), `explanation`:

- numeric T1: exact additive decomposition `score = intercept + sum(weight x calibrated mean)`;
- numeric T2/T3: posterior-weight-conditional linear attribution (exact at the
  observed eta weights; not a deletion effect) plus the joint T2/T3 mean/cov;
- text: evidence trace (material, category, condition, time); with
  `explain="deletion"`, exact score changes from withdrawing one report (dynamic)
  or one event per condition/category cell (T1).

## 8. What a bundle does not do

- It does not read new text: unregistered materials require running the
  declared reader in a separate reading stage.
- It does not fix a wrong observation law (nonlinear worlds keep low coverage),
  certify population-model calibration (original-unit T2/T3 coverage failures
  are attached), or measure human personality.
- Integrity against deliberate consistent edits of all copies of a value relies
  on the trusted file hash; internal checks catch inconsistent edits.
