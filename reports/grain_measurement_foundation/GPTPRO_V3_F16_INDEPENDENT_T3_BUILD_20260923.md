# F16 T3 independent-agent construction, 2026-09-23

## Scope and status

This report covers only the independently implemented T3 numeric predictive
instrument, tests and source-only reference construction required by
`docs/GRAIN_F16_INDEPENDENT_DYNAMIC_HANDBOOK_V1.md` and
`configs/grain_f16_dynamic_briefs_v1.json`. It is not an acceptance claim for
F16 as a whole, the parent research plan, T1/T2, empirical interval calibration,
human reconstruction H06, or human personality validity.

All three source-only references are frozen and the saved-artifact verification
passed. The 21 standalone tests, Ruff check and Ruff format check passed. No
application/validation/truth input was opened. No global status file or goal was
changed; no process from this construction remains running.

## Changed paths

All paths below are relative to
`/Volumes/mobile3/projects/Sliced Utterance In-Context Assessment`.

- `grain_research/gptpro_foundation_v3/f16_independent_t3.py`
- `scripts/run_grain_gptpro_f16_independent_t3.py`
- `tests/test_grain_gptpro_f16_independent_t3.py`
- `reports/grain_measurement_foundation/GPTPRO_V3_F16_INDEPENDENT_T3_BUILD_20260923.md`
- `results/grain_gptpro_foundation_v3/F16/independent_t3_v1/`

The result root contains the construction log, source-only bundles/checkpoints,
read manifests, synthetic CLI evidence, unit-test artifacts and an own-output
verifier. The verifier is not a second independent agent or human evaluator.
No repository modules, tests or reports outside this write scope were imported
or read. The parent T2 files were not inspected or modified.

## Implementation

The standalone implementation has no repository imports. The CLI and tests load
this file directly with `importlib.util.spec_from_file_location`, bypassing
repository package initialization. The mathematical code uses NumPy/SciPy only.

- Stationary Gaussian AR(1) filtering with exact arbitrary integer-gap
  transitions; duplicate/mixed/malformed observations fail closed.
- Two specified L-BFGS-B starts with analytic forward likelihood gradients,
  the handbook's optimizer tolerances, central Hessian step 1e-4, and no
  curvature repair or unconverged fallback.
- Exactly 200 common PCG64 normal draws from
  `SeedSequence([seed,16,3,bootstrap])` for each personal Laplace integral.
- Four-step target under `[1,-1,1,-1]`, scalar predictive variance checked
  against separately assembled full 4-by-4 predictive covariance for every draw.
- Event/state, personal-parameter and source variance components retained;
  mixtures use ddof=0 and the source reference SD uses ddof=1.
- A failure in any frozen source replicate makes the personal score unsupported.
  Unsupported source reference scores stop construction rather than being
  excluded. Diagnostics for both optimizer starts and all replicates are saved.
- Empty inputs are explicitly prior-only. The working Gaussian moment interval
  remains `NOT_EVALUATED` for calibration.
- Strict JSON bundles embed all source fits, target/reader/world/support/error
  contracts, algorithm/RNG settings, hashes and frozen source reference.
  The CLI checks an externally supplied full-file SHA256 before parsing the
  bundle and separately verifies the expected source hash.
- Cold scoring has no source-file read or stored-author-score lookup. It reads
  the supplied observed prefix and embedded ensemble only, apart from hashing
  its own implementation for integrity.

`build_instrument(brief, source_manifest, ...)` returns `(bundle, checkpoint)`.
The manifest contains `seed`, `world`, `source_prefixes` keyed by author,
`ensemble`, `provenance` and `source_hash`; the CLI constructs it exclusively
from the permitted source files. Optional keyword arguments implement bounded
construction, checkpoint callbacks and resume. `cold_score(input, bundle)`
requires `construction_status=SOURCE_REFERENCE_FROZEN`.

Cold input is exactly:

```json
{"world":"core","reader_hash":"<bundle reader_hash>","prefix":[]}
```

Nonempty prefixes contain only the eight documented source-style observation
fields, one author/world/role/episode, and strictly increasing integer times.
Application prefixes are left to the coordinator. Extra future/truth fields
are rejected; the scorer cannot detect values falsely labeled as observations
without external provenance. Frozen reference files must not be normalized
again on application inputs.

## Source construction

Construction uses three seeds, 300 source authors per seed, the first 48
observations per author, 200 already-fitted source bootstrap models and 200
personal draws per replicate. These are simulated author instances, not people.
No source distribution is re-estimated.

| Seed | Authors | Reference mean | Reference SD, ddof=1 | Status |
| --- | ---: | ---: | ---: | --- |
| 20260915 | 300 | -0.05340180411823948 | 0.9738031197136882 | frozen |
| 20260916 | 300 | -0.047855942846306766 | 0.9573140176144749 | frozen |
| 20260917 | 300 | -0.006518002404736321 | 1.0933754520676233 | frozen |

Production construction scored all 900 source prefixes: 43,200 observed source
rows, 180,000 personal MAPs, 360,000 converged optimizer starts and 36,000,000
personal draws. These counts exclude additional synthetic tests and source-only
cold replays. All 600 supplied bootstrap fits were FITTED, with verified source
identities; 603 source/fit file hashes were recorded. No source score or
bootstrap replicate was dropped, no MAP/curvature failure occurred in production,
and no fallback or Hessian repair was applied.

The final passes took 150.069 seconds (resumed seed 20260915), 153.436 seconds
(20260916), and 153.515 seconds (20260917), in addition to the initial ten-author
pass of approximately 5.5 seconds. These are execution times, not total agent time.

Every ten newly scored authors are checkpointed. Seed 20260915 was first run
for ten authors and then resumed using the exact external checkpoint hash;
the other seeds run sequentially. Six numerical-thread environment variables
are forced to one before importing NumPy/SciPy, no pools are created, and no
two construction/test numerical processes run concurrently.

The old ten-author `core_20260915/draft_bundle.json` is deliberately retained as
an explicitly unscoreable partial artifact. Use only each seed's `bundle.json`
after the final source-reference verification.

## Verification

All 21 standalone tests passed again in the final verifier (1.204 seconds).
They cover dense-Gaussian equivalence
of the stationary/gapped filter (including negative and zero rho), analytic
gradient finite differences, central Hessian calculation, independent
loading-matrix predictive covariance, uncertainty decomposition, source
standardization, prior-only scoring, batch/author-ID invariance, wrong
world/reader rejection, malformed input and future-field rejection, cold-file
isolation, failed bootstrap/MAP/Hessian retention, checkpoint/resume integrity,
ten-author checkpoint cadence, and actual warm/cold CLI execution.

Tiny synthetic warm and cold CLI score JSON files matched byte for byte.
Negative bundle/source-hash CLI tests exited 2 with explicit hash mismatch
errors. Initial Ruff style findings were fixed before production construction;
the frozen scorer hash has not changed during source construction.

Final all-seed output arithmetic audit, production cold replay and repeated
test/Ruff evidence are recorded in
`results/grain_gptpro_foundation_v3/F16/independent_t3_v1/verification.json`.
This audit recomputes mixture and reference moments with Python's `statistics`
module rather than importing the scorer's aggregation functions. Production
cold replay uses one permitted source prefix per seed, never application data.

The maximum recorded scalar/full-covariance disagreement over all production
draws was 2.220446049250313e-16. The separate saved-moment/reference recomputation
had maximum absolute disagreement 4.440892098500626e-16. Minimum personal Hessian
eigenvalues by seed were 13.8901772443, 10.4917127588 and 11.6578701727, respectively.
All three production cold CLI outputs matched the corresponding warm source
scores and full per-replicate diagnostics exactly; standardized values were
separately checked against the frozen reference. The three cold CLIs, test
runner, Ruff check and format check all returned exit code 0.

Average total variance was 0.219239, 0.224729 and 0.224505, respectively. The
reported variance decomposition is a working bootstrap/Laplace moment
calculation, not evidence of calibrated coverage or human validity.

### Frozen bundle hashes

Scorer implementation SHA256:
`d48866805ef7cef7ebb46c5bdc0bf0b3ebcaf22ca944bbba1685371f1fc8d5ed`.

| Seed | bundle.json SHA256 |
| --- | --- |
| 20260915 | `5706c42eb637ff35e507bf91f3060b339e461c837612f788fba1b67a5499e14e` |
| 20260916 | `7624b77f8d5aca53b80a1c95907e0d6677526e72663297eaedb497944bee8594` |
| 20260917 | `6f71d7f06669de98155e02cf1c5bdc0bfea8704d6652e83520adf389a4b33d3e` |

All bundles are under the result root's corresponding `core_<seed>/` directory.
`verification.json` and `run_summary.json` also record the expected source and
checkpoint hashes. Each JSON artifact has a SHA256 sidecar. The tiny synthetic
bundle is labeled test-only and is not a production source reference.

## Exact commands

Working directory for these commands:
`/Volumes/mobile3/projects/Sliced Utterance In-Context Assessment`.

Commands actually used for construction:

```sh
"/Volumes/mobile3/projects/project persona/.venv/bin/python" -B scripts/run_grain_gptpro_f16_independent_t3.py build-source --seed 20260915 --max-new-authors 10
"/Volumes/mobile3/projects/project persona/.venv/bin/python" -B scripts/run_grain_gptpro_f16_independent_t3.py build-source --seed 20260915 --resume-sha256 7611b90a05ea2eb7a4a3f615471cfbd7ba99599cebdd9410431ff8609b6cd39a
"/Volumes/mobile3/projects/project persona/.venv/bin/python" -B scripts/run_grain_gptpro_f16_independent_t3.py build-source --seed 20260916
"/Volumes/mobile3/projects/project persona/.venv/bin/python" -B scripts/run_grain_gptpro_f16_independent_t3.py build-source --seed 20260917
```

Standalone tests and final verifier:

```sh
"/Volumes/mobile3/projects/project persona/.venv/bin/python" -B tests/test_grain_gptpro_f16_independent_t3.py
"/Volumes/mobile3/projects/project persona/.venv/bin/python" -B results/grain_gptpro_foundation_v3/F16/independent_t3_v1/verify_artifacts.py
```

The verifier records exact cold CLI commands, stdout/stderr and exit codes for
each seed, then executes the standalone tests and isolated/no-cache Ruff check
and format check. It does not invoke pytest discovery or read repository config.
Production rebuild commands deliberately refuse to overwrite an existing
checkpoint without its externally supplied SHA256. Fully populated checkpoints
can be reloaded without redoing personal fits.

## Read paths and added rules

Exact permitted source/fit paths and file SHA256s are in each seed's
`read_log.json` and bundle source provenance. The fixed paths are
`F05/source_full_r2/core_<seed>/source.jsonl` and
`F13/dynamic_source_full_r1/core_<seed>/fit_<000..199>.json` under
`results/grain_gptpro_foundation_v3/`, for seeds 20260915, 20260916 and 20260917.
The handbook, frozen brief and new Python files are also recorded. Cold CLI
read logs and `verification.json` record subsequent own-output reads and the
permitted source-prefix replays. The construction log records initial schema
reads and manual evidence reads.

All questions and added rules are in
`results/grain_gptpro_foundation_v3/F16/independent_t3_v1/construction_log.md`.
No clarification was requested. Notable additions are the explicit numeric
reader hash, exact cold JSON schema, single-episode contract, separate reference
per seed, minimum-norm least-squares initialization for rank-deficient z,
explicit PCG64, first-start tie rule and checkpoint/hash contracts.

## Unmet and outside-scope requirements

- No implementation/test/source-only construction step in this packet is left
  unfinished; this is not a broad F16 or parent-plan completion claim.
- Application/validation/truth scoring, application interval calibration and
  comparison with the parent's T2 remain unrun by design.
- Independent human H06 reconstruction is not provided by this separate agent.
- No broad parent-card or full-research-plan acceptance is asserted.

No API/neural training, private data, source-model refit, remote execution,
lockbox, commit, push, publishing, parent status update or goal-tool change
was performed.
