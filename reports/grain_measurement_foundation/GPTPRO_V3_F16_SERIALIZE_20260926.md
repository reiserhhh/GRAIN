# F16-R03-SERIALIZE: public calibrated bundles and sealed cold scoring

Date: 2026-09-26 JST. Worker: Claude Opus 5.5 (TASK-CLAUDE-GRAIN-FULL-LOOP-20260926).
Protocol frozen before new computation: `docs/GRAIN_F16_SERIALIZE_PROTOCOL_V1.md`.
Handbook: `docs/GRAIN_F16_PUBLIC_CALIBRATED_BUNDLE_HANDBOOK_V1.md`.
**The finite serialization packet is executed and independently audited.** It
does not promote any parent card and does not add human validity evidence.

## Main findings (revised 2026-09-26 01:50 JST after Fable seminar 01 and supervisor check)

**Terminology.** The T1/T2/T3 *estimands* are fixed latent quantities
(m(+1)-m(-1); E[s_47|prefix]; E[mean(y_48..51)|prefix, plan]) in their original
numeric source units; no packet changes them. A reader change alters the
*scoring function* (the observation law through which text categories update the
posterior over the same estimand) and therefore the *source score-reference*
(the mean/SD of source-author scores produced by that scoring function), unless a
calibration maps the new observations back into the identical unit and scoring
function (the numeric T1 case below).
Raw-unit scores and reference positions are reported separately below.

1. **The completed transport estimators can now be published as self-contained,
   authenticated instruments.** 162 bundles (numeric T1 48, numeric T2/T3 96,
   changed-reader text T1/T2/T3 18) reproduce, in fresh processes given only the
   bundle and observation envelope, the archived raw-unit application score,
   variance and every variance component: max difference 2.13e-14 (numeric T1),
   1.14e-13 (numeric dynamic), 0 (text full engine objects; this is replay
   fidelity of the same engine code, not an independent recomputation, which for
   text rests on `text_transport_audit_r2`).
2. **Score-references must be versioned with the scoring function and material
   bank; the reader-attributable part is modest.** Retained analysis
   `F16/serialize_reference_shift_r1` keeps raw-unit application scores fixed and
   compares positions against three source score-references: H historical (old
   reader, old 96-text bank), C control (old reader cache + old source law on the
   new bank, same rendering stream and estimator family), F fresh (changed reader
   + pilot law, published). Mean absolute position shift (units: SD of the
   respective reference: a shift is a difference of two z-scores, each on its own
reference SD, not a single common unit; paired author bootstrap within seed,
3 seeds, n=600; the three columns are separate comparisons and do not add):

   | Target | F vs H (total) | C vs H (bank/rendering/law re-estimation) | F vs C (reader + its pilot law) |
   |---|---|---|---|
   | T1 | .095 [.092,.098], max .24 | .124 [.121,.126] | .050 [.047,.052], max .23 |
   | T2 | .245 [.228,.265], max 1.62 | .307 [.282,.331] | .065 [.059,.071], max .41 |
   | T3 | .167 [.159,.175], max .40 | .106 [.100,.112] | .122 [.1215,.1229], max .14 |

   Raw-unit reference statistics (source authors, `source_references.csv`): the
   reader change moves the T3 reference mean by about -.11 raw units (e.g. seed
   20260915 C -.046 -> F -.154) with similar SD (.896 -> .901); T2 source scores
   have small between-author spread (raw SD .095-.109 for the historical
   reference H, .133-.162 for C and F), so small raw shifts become large T2
   position shifts. The earlier sentence attributing the
   1.62 SD maximum to the reader change was wrong: that maximum is mostly the
   bank/rendering part. Numeric T1 legitimately inherits its reference (same
   unit, same scoring function; replayed to 2.2e-16); numeric T2/T3 already used
   a freshly scored conditional-Gaussian reference (replayed exactly).
3. **The stale inner reader hash is resolved, not blessed.** In every text
   primary payload the inherited `reader_hash` reproduces exactly the hash of the
   OLD reader's materials/channels and not the changed payload. The public format
   moves all stale fields into `legacy_reconciliation` with value hashes and
   reasons, proves every score-determining field byte-identical to the
   manifest-authenticated payload, binds the changed readout cache and its
   definition by a recomputed `reader_id`, and refuses any bundle whose engine
   top level still carries a stale field. (What travels is a readout cache for
   registered materials, not a runnable reader.)
4. **Calibration uncertainty travels as data that re-derives itself, as an
   internal-consistency check.** Numeric bundles embed pilot fit pairs and
   bootstrap indices (every calibrator refit at load, tolerance 1e-11); text
   bundles embed pilot counts and generator seed (draws regenerated exactly).
   This catches inconsistent edits; a consistent re-derivation from edited pilot
   pairs followed by re-identification would pass, so integrity rests on the
   trusted out-of-band file hash. Pilot-file provenance inside a bundle is a
   declaration.
5. **Sealing scope.** The CLI's in-process audit hook refuses Python-level file
   opens outside declared paths and blocks sockets/subprocess events; it is not an
   OS sandbox and its "0 undeclared reads" count is self-reported. An additional
   OS-level check (`F16/serialize_os_seal_r1`, macOS sandbox-exec denying content
   reads under the results/data trees of both repositories and all network)
   reran one mixture bundle per route per seed (12 runs): all byte-identical to
   the hook-sealed outputs, and an OS-level truth-file read was refused (EPERM).
   The other 150 bundles have hook-level sealing only.

## What was built

| Object | Content |
|---|---|
| `grain_research/gptpro_foundation_v3/f16_serialize.py` | bundle format `F16_public_calibrated_bundle_v1`, verification, per-route cold scoring, explanations, sealed CLI (audit hook) |
| `scripts/run_grain_gptpro_f16_serialize.py` | prepare / build / cold / reject stages |
| `scripts/audit_grain_gptpro_f16_serialize.py` | independent formulas and replay audit |
| `tests/test_grain_gptpro_f16_serialize.py` | 7 focused tests (incl. sealed subprocess probe) |

Estimators kept distinct: numeric routes integrate every source x pilot
calibrator pair (200 x 200 = 40,000; T2/T3 with the 64-node eta Gauss-Hermite
rule); text routes pair source b with channel draw b (200 paired Monte Carlo
components, combined source/reader spread). Plugin bundles propagate no
calibration uncertainty and say so.

## Build refusals (retained as results)

`refusals.json` holds 49 refusals: 18 `NONMONOTONE_INVERSE` (every nonlinear
cubic calibration: 3 seeds x [T1 two estimators + T2/T3 two estimators x two
targets]); 24 reused `MISSING_ANCHOR` pilot certificates (4 worlds x 3 seeds x
2 numeric routes); 4 text `MISSING_ANCHOR` pole/qualification controls; 2
`APPLICATION_CALIBRATION_NOT_OPERATIONAL` (known-pool law, oracle labels); 1
`APPLICATION_REFERENCE_FORBIDDEN`.

## Checks (actual outputs)

| Check | Scope | Result |
|---|---|---|
| Sealed cold replay vs archived application outputs | 162 bundles x 200 application authors each | max 2.13e-14 / 1.14e-13 / 0 |
| Numeric T1 event-level conditioning (independent formula) | 48 bundles, all 200 authors each | score 6.97e-13, variance 2.66e-15 |
| Numeric T2/T3 direct joint-covariance conditioning per eta node | 96 bundles, sampled first/last author each | score 1.49e-13, variance 2.95e-14 |
| Explanation identity (additive / posterior-weight attribution) | all numeric rows | 2.22e-14 |
| Fresh text reference replay in sealed processes | 18 bundles x 300 source authors | exact (0.0) |
| Provenance re-derivation | calibrator refits, bootstrap indices, channel counts, channel draws, inherited/fresh references | all exact / <=2.2e-16 |
| Sealing | 180 hook-sealed processes (self-reported); 12 OS-sandboxed reruns | hook: 0 refused/undeclared opens; OS: 12/12 byte-identical, truth read refused |
| Rejections | 61 actual cases across 4 routes (wrong reader/target/version/clock, truth field, incomplete support, uncalibrated condition, missing step, unregistered text, late report, undeclared truth read in sealed process, tampered/re-identified draws, edited pilot pair, edited/app-derived reference, restored stale reader hash, edited reader cache, wrong schema, changed code hash) | 61/61 expected code |
| Tests / lint | `tests/test_grain_gptpro_f16_serialize.py` | 7 passed; Ruff clean |

Audit history retained: r1 stopped on a comparator bug (required identical key
sets; the cold text engine path adds `active_materials`, `stale_annotations`,
`reference_score`, `reference_variance`); r2 was launched before the fix
applied and stopped by the worker; r3 computed every check but failed writing
its summary (numpy bool); r4 is the complete audit. No bundle, cold output or
scientific number changed between audit revisions.

## Retained failures and limits (attached to each bundle)

- Nonlinear world: affine calibrators remain misspecified (T1 mixture coverage
  .663, T2/T3 .885/.882); the cubic inverse is refused.
- Original-unit numeric T2/T3 references keep their failed Wilson coverage
  gates; unit recovery does not certify population-model calibration.
- Text pilot calibration still trails the known-pool reference; T2 benefit
  crosses zero; application materials share a finite controlled bank.
- Cold scoring accepts only registered materials; a new text needs the declared
  reader in a separate reading stage.
- The dynamic numeric estimator requires a complete 48-step prefix; the missing-
  step target-clock adapter is not part of this compiled transport estimator.
- Same-agent alternate formulas and sealed replays are engineering evidence,
  not independent-human reconstruction or psychological validation.

## Artifacts

- Runs: `results/grain_gptpro_foundation_v3/F16/serialize_full_r1/` (bundles,
  refusals, provenance checks, fresh references, cold outputs + manifests,
  rejection cases); smoke `serialize_smoke_r1/`.
- Audit: `results/grain_gptpro_foundation_v3/F16/serialize_audit_r4/` (r1–r3
  retained with failure notes). Tests: `serialize_tests_r1.xml`.

Reproduce into NEW directories: runner stages `prepare`, `build`, `cold`,
`reject`, then the audit script with `--run`/`--output`. Shared venv,
`PYTHONPATH=.`, single-thread BLAS.

## Requirement evidence (no parent promotion)

R22 (reference policy per route, no application norms), R23 (changed readout cache and
version published with identity), R24 (full calibration uncertainty in the
public score), R25 (pilot provenance and refusal of application calibration),
R33 (evidence explanations with exact identities/deletion effects), R34
(reusable construction: self-contained bundle + sealed cold path + handbook),
R35 (frozen protocol, retained failures, audit revisions). Parent cards
F12-01..05 and F16-02/03/04/06 receive evidence; none is promoted.

Next: requirement ledger reconciliation and remaining original gaps
(F15, F01, F10, then packaged reviews), without repeating this packet.

## Declared deviations

- SERIALIZE is a recovery subtask, not an original card; it does not emit the
  COMMON eight-file set or `--stage <card>` entry. Evidence is attached to cards
  without promotion.
- Protocol v1 said NONMONOTONE_INVERSE "3 seeds x 2 routes"; actual refusal rows
  are per estimator and target (18). Stale-field refusal checks the engine top
  level, where the inherited fields lived.
- Retained analyses added after the audit: `serialize_reference_shift_r1`
  (script `scripts/analyze_grain_gptpro_f16_reference_shift.py`) and
  `serialize_os_seal_r1` (script `scripts/run_grain_gptpro_f16_os_seal_check.py`).
