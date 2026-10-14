# F16-R03: public scoring contract and T1 matched pilot transport

Date: 2026-09-24. Status: **public-contract and numeric-T1 finite subpackets complete;
dynamic/text transport and parent F16 remain open**. No new neural calls, private
human data, source-response fitting, remote work, lockbox, commit or push.

## Findings and their use

1. Six actual numeric/text target routes now share an executable, authenticated
   public contract. Existing source models and scores are preserved, rather than
   replaced by similarly named fields or a new estimator.
2. Correlation can stay unchanged while units and interval coverage fail badly.
   An independent paired pilot recovers an affine reader change on the actual
   T1 target. This is a measurement-transport finding, not personality accuracy.
3. Condition-dependent distortion needs a condition-dependent calibration law.
   Bootstrap uncertainty around the wrong global law does not remove its bias.
4. The prescribed cubic inverse fails its monotonicity gate. A known nonlinear
   inverse works, separating information availability from estimator inadequacy;
   that known mapping is not a learned solution.

## Public contract: what was actually integrated

`grain_research/gptpro_foundation_v3/f16_public.py` implements
`build_instrument(brief, source_manifest)` and a cold scoring CLI. Build compiles
an already source-fitted bundle; it does **not** itself estimate source parameters
or replace the independently built T3 model.

The public bundle binds target, unit, condition/world, target time, observed role,
reader/material/channel identity, source provenance, fixed reference and code
hashes. Internal legacy `future_response` is exported as canonical
`predictive_state` without rewriting frozen payloads. This release's target clocks
are T1=31 and T2/T3=47; arbitrary-time source norms are not supplied.

Three seeds, numeric/text T1/T2/T3 give 18 route cells. The text routes replay all
200 application objects per cell; numeric routes select the first/last original
application objects, reusing prior full numeric replay evidence.

| Check | Actual result |
|---|---:|
| Public warm/cold score objects | 1,818: 1,800 text + 18 numeric |
| Maximum warm/cold discrepancy | 0 |
| Maximum discrepancy against original archive | 2.78e-17 |
| Batch/renamed-ID/extreme-stranger invariance discrepancy | 1.20e-12 or less |
| Deliberate wrong-contract cases rejected | 234, 13 per cell |
| Additional actual CLI wrong-file-hash rejection | passed, no output created |

The source reference stays fixed. Adding a stranger or changing an ID does not
restandardize the evaluated person. Missing tails and empty histories are checked;
prior-only output is not a measured zero. Text routes use the frozen controlled
material bank, not arbitrary natural language.

## T1 transport design

Reuse F12's three seeds and four observation maps, its independent paired pilot
100 authors (first60 fit, last40 validation), and 200 application authors per seed.
Source is the actual F16 four-condition model with all200 source-bootstrap fits.
T1 remains `m(+1)-m(-1)`, evaluated as `2*b` in this simulator. The old source prior
is not replaced with the shifted application population's true distribution.

All32 events are used, matching the F16 source/reference design. Old F12 OLS used
24; its published scores are not a matched estimator comparison. No application
truth enters calibration; application score objects are saved before truth is
opened. Each calibration uses 200 whole-pilot-author resamples. All40,000 independent
source-by-pilot combinations enter its moments, not diagonal pairs.

Writing corrected condition means as `v_j`, a source component has
`m_s(v_j)=a_s+w_s'v_j` and conditional variance `V_s`. The working mixture uses

```text
score = E_j E_s[m_s(v_j)]
variance = E_s[V_s] + E_j Var_s[m_s(v_j)] + Var_j E_s[m_s(v_j)]
```

These are conditional, source and pilot components of a cut/bootstrap working
model. They are not an exact universal psychological posterior. Exact deterministic
paired affine maps legitimately leave almost zero pilot variance. No additional
reader noise was invented to make that component positive.

## Quantitative results

Means across three equal-sized seeds, 600 application authors total. These high
correlations are **numeric latent-response simulations**, not Big5/MBTI/text
prediction. Coverage is for the stated working 95% interval.

| Observation map / calibration | MSE | Mean r | Coverage | Mean width |
|---|---:|---:|---:|---:|
| Original units / none | .057575 | .990701 | .948333 | .912457 |
| Global affine / none | .638787 | .990701 | .438333 | .915368 |
| Global affine / global pilot | .057575 | .990701 | .948333 | .912457 |
| Conditional affine / none | 6.216723 | .929158 | .001667 | .913757 |
| Conditional affine / global pilot | 4.911296 | .929158 | .005000 | 1.003416 |
| Conditional affine / conditional pilot | .057575 | .990701 | .948333 | .912457 |
| Nonlinear / none | 5.478938 | .884198 | .458333 | .919588 |
| Nonlinear / global pilot | .712226 | .884198 | .646667 | 1.093609 |
| Nonlinear / conditional pilot | 1.052051 | .872473 | .663333 | 1.328463 |
| Nonlinear / prescribed cubic inverse | unsupported | NA | NA | NA |
| Nonlinear / known inverse reference | .057575 | .990701 | .948333 | .912457 |

Matched affine inverses recover original-unit scores to at most1.28e-14.
Stratified paired application-bootstrap MSE differences (2,000 draws, conditional
on the fitted source/pilot artifacts): global calibration versus none
-.581213[-.658658,-.508294]; conditional versus global calibration in the conditional
world -4.853721[-5.082698,-4.635543]. The latter rejects the idea that one pooled unit
correction is enough for all conditions in this world.

For nonlinear/global calibration, adding pilot uncertainty changes coverage
.571667 to .646667 and width .912997 to1.093609, but MSE .694055 to .712226.
Conditional-world/global calibration remains at .005 coverage despite its added
pilot variance. More honest uncertainty is useful; it does not repair a wrong law.
The cubic plugin is nonmonotone in all three seeds. All bootstrap statuses and
unsupported rows remain; no successful-draw-only renormalization. Single-condition
pilots have rank2 versus required4 and return `MISSING_ANCHOR` in all12 controls.

## Verification, failures and scope

Canonical runs:

- `results/grain_gptpro_foundation_v3/F16/public_contract_full_r1/`
- `results/grain_gptpro_foundation_v3/F16/t1_transport_full_r1/`
- `results/grain_gptpro_foundation_v3/F16/public_transport_audit_r3/`
- `results/grain_gptpro_foundation_v3/F16/public_transport_tests_r2.xml`

Audit checked475 artifact hashes;600 source kernels by dense32x32 observation-space
conditioning;5,427 pilot fits by a QR solver;134,400 target moments using covariance
contractions instead of the runner's mixture enumeration;18 actual public T1 cold
scores after calibration; and all saved per-seed metrics. Maximum discrepancy
1.20e-12. This is the same agent using different formulas, not an independent
human builder.110 focused tests and Ruff pass; test counts are not claim counts.

Preserved failures: public smoke r1 used the wrong legacy T3 score field, smoke r2
misclassified NumPy scalar comparison; smoke r3 and full r1 pass. Audit r1 matched
Mac `._` sidecars; r2 supplied an undeclared `episode_id` to the strict T1 schema;
r3 excludes sidecars and uses the actual seven-field input contract. Original
scientific results and frozen scorers were not edited to rescue these checks.

Reproduce into **new** output directories from the implementation repository:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. '../project persona/.venv/bin/python' scripts/run_grain_gptpro_f16_public.py --output results/grain_gptpro_foundation_v3/F16/public_contract_replay_new
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. '../project persona/.venv/bin/python' scripts/run_grain_gptpro_f16_t1_transport.py --output results/grain_gptpro_foundation_v3/F16/t1_transport_replay_new
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. '../project persona/.venv/bin/python' scripts/audit_grain_gptpro_f16_public_transport.py --public-run results/grain_gptpro_foundation_v3/F16/public_contract_replay_new --transport-run results/grain_gptpro_foundation_v3/F16/t1_transport_replay_new --output results/grain_gptpro_foundation_v3/F16/public_transport_audit_new
```

No routine rerun is needed. Public replay took280seconds; compact T1 transport
1.86seconds because it reuses source models and exact four-dimensional kernels,
not because it substitutes random scores or repeats neural training.

## Next bounded work

1. Apply matched pilot calibration to actual T2/T3 dynamic targets; preserve time,
   future-condition identity, and accumulated future covariance.
2. Calibrate changed controlled-text reader/qualification channels on independent
   pilot material. Do not apply a numeric affine inverse to categorical labels.
3. Serialize the learned calibration and its uncertainty into public bundles;
   the current18 corrected public examples verify plugin composition, not a cold
   serialized full-pilot mixture. Retain unsupported/anchor failure rules.
4. Reconcile remaining original F01/F15/F10 and all44 computational questions plus
   six human-stage preparations. No parent promotion from this finite packet.
