# F16 four-cell source uncertainty and independent T3 application

## Decision

Two executable gaps are completed at their finite scope: the actual four-cell T1 source-bootstrap construction, and application/cold verification of the independently built T3 instrument. T1, T2 and T3 now have actual frozen-source numerical implementations and application evidence, rather than surrogate scores. F16 parent acceptance remains open: target evidence cards, F08 qualification/history integration, F12 target-matched transport and canonical card adoption are not supplied by this packet.

This is synthetic mathematical construction. No new personality accuracy, private-text assessment, neural training, external API use or human H06 reconstruction is claimed. All jobs in this packet are terminal. The full original goal remains active, with no commit, push, remote execution or lockbox action.

## 1. Independent T3 construction and application

Hume (`01a0cdec-a67e-7e73-9e6e-742b8feecc7c`) implemented T3 using only the frozen brief, mathematical handbook and allowed source inputs, without inheriting coordinator conversation or reading T1/T2 code. The independent construction supplied three frozen reference panels of 300 simulated authors each, retaining all 200 source fits and 200 personal Laplace draws. Its own tests and one source-prefix cold replay per seed passed.

The coordinator inspected the independent implementation and its twelve recorded added rules. They specify numeric reader identity, single-episode/ordered input, initialization, analytic likelihood gradient, tie handling, deterministic integration and checkpoint contracts. The rules are visible additions, not proof that a handbook uniquely determines every implementation choice. A current check covered 616 declared read entries, including 603 immutable source/fit hashes; no application/truth/T1/T2 code entry occurred. This is declared-log plus code review, not operating-system access tracing or human blindness.

The coordinator then scored all 600 application authors, first48 observations only, using the frozen bundles. Separate cold processes received a copied standalone scorer, bundle and input prefix. All complete warm/cold objects agree exactly. Each of the three parallel seed cells took about 192 seconds. No source model was refitted.

### Correct outcome under the planned condition

The T3 plan is `z=(+1,-1,+1,-1)`. Archived F05 observations48..51 instead have `z=(-1,-1,-1,-1)`. Evaluating predictions on those unmodified observations would compare different targets.

Before application scoring, `configs/grain_f16_t3_application_v1.json` fixed the evaluator. In the declared additive core simulator,

`y_plan[k] = y_archived[k] + b_true*(z_plan[k]-z_archived[k])`.

This retains the exact archived states and measurement innovations. It is valid here because z does not alter the state transition or innovation law. The hidden b value is opened only in the separate post-score evaluator, never by the scorer. This is a simulator intervention, not an adjustment available for natural human text. No model, width or threshold was changed after seeing application performance.

| Seed | Planned-outcome MSE | Predictive coverage | Wilson check | Wrong archived-target coverage, diagnostic only |
|---|---:|---:|---|---:|
| 20260915 | 0.247580 | 0.945 | PASS | 0.705 |
| 20260916 | 0.271668 | 0.940 | PASS | 0.710 |
| 20260917 | 0.256188 | 0.925 | PASS | 0.655 |

These are nominal95 predictive intervals for a realized four-step average, not confidence intervals for the conditional expectation alone. PASS means the original Wilson95 check contains .95; it does not mean exact .95 empirical coverage. The known-parameter conditional oracle has MSE .221136/.231516/.212676. The mixture's pooled excess MSE is +.036703, paired-author95 CI [.022151,.052904]. Source/person parameters therefore still cost predictive accuracy; passing the finite coverage gate is not oracle equivalence.

Every saved mixture moment and all 900 reference scores were reconstructed. For the predetermined first/last application authors and source fits0/100/199, all 3,600 parameter draws were checked by independently conditioning a joint dense past/future Gaussian covariance. Maximum discrepancy is 3.39e-15. This is not an independent optimizer refit audit.

## 2. Four-cell T1 source uncertainty

T1 remains `m(+1)-m(-1)` on the original F04 L four-cell data. New source fitting re-estimates the original four-cell noise-corrected moment estimator on 200 author-bootstrap samples per seed. Each source copy is an independent resampling unit even when the same author is drawn repeatedly. The frozen spectral floor1e-6 is retained with raw covariance eigenvalues, not silently removed or invented during scoring.

For source fit b, the four-dimensional posterior is Gaussian. With contrast c=(-1,0,0,1), its mean and variance are `m_b=c' mu_b` and `v_b=c' Sigma_b c`. The primary output integrates these analytically:

`score = mean_b(m_b)`

`variance = mean_b(v_b) + Var_b(m_b)`.

The mandated 200 joint four-dimensional conditional draws per source fit are a separately reported Monte Carlo comparator. Integration normals are shared across authors within source replicate, never keyed by identity or application order. Norms use the same primary scorer on 300 source authors, then freeze. The old fixed-source T1 bundles and results remain untouched.

| Seed | Analytic MSE | Analytic coverage | MC coverage | Source variance | Within-source variance | Wilson check |
|---|---:|---:|---:|---:|---:|---|
| 20260915 | 0.067807 | 0.910 | 0.910 | 0.0001143 | 0.0546345 | FAIL |
| 20260916 | 0.047786 | 0.955 | 0.960 | 0.0001165 | 0.0535843 | PASS |
| 20260917 | 0.046466 | 0.980 | 0.980 | 0.0001062 | 0.0536378 | PASS |

All 600 application cold/warm objects match exactly. All 600 source fits were re-created using the original row-level least-squares estimator with explicitly renamed bootstrap copies; maximum fit discrepancy6.22e-15. All 900 source and 600 application panels, including 60 million conditional four-dimensional draws, were replayed by an observation-space solve rather than the production sufficient-cell solver; maximum discrepancy2.78e-13. Source normalization differences are zero.

Spectral flooring occurred in593/600 source fits; minimum raw eigenvalue was -.014197. The underlying L world is a linear family represented in four cells, so low-rank signal and sampling noise in additional directions are relevant context. This count is reported, not treated as a numerical failure or silently upgraded to an exact full-rank population covariance.

Source variation is approximately0.208% of the total working variance. Compared with the single-fit four-cell plugin, the primary mixture changes pooled MSE by +.00001045, paired95 CI [-.00004775,.00006700]. The MC-minus-analytic difference is +.00004059 [-.00002594,.00010759]. These intervals are not equivalence tests. The failed seed remains failed; more identical source resampling is not supported as a useful next mechanism experiment.

The original smoke used12 source/6 application authors,4 source fits and10 inner draws. Its score object's generic `mc_scope` text refers to the full200-draw protocol; the smoke protocol and counts are authoritative. Smoke is interface evidence only and is not included in the full scientific tables above.

## 3. Reproducible evidence

All paths are relative to this repository.

- Independent T3: `results/grain_gptpro_foundation_v3/F16/independent_t3_v1/`; report `GPTPRO_V3_F16_INDEPENDENT_T3_BUILD_20260923.md` in this report directory.
- T3 application: `F16/t3_application_full_r1/`; evaluation `F16/t3_application_audit_r1/`, both under `results/grain_gptpro_foundation_v3/`.
- T1 source ensemble/application: `F16/t1_source_full_r1/`; evaluator `F16/t1_source_audit_r1/` under the same root.
- Paired2000-resample author comparisons: `F16/paired_target_comparisons_r1/summary.json`; `tests.xml` records the final related test suite.
- New runners: `scripts/run_grain_gptpro_f16_t3_application.py`, `scripts/run_grain_gptpro_f16_t1_source.py`; matched `audit_grain_gptpro_f16_*` scripts; `scripts/compare_grain_gptpro_f16_targets.py`.
- Public cold facade `grain_research/gptpro_foundation_v3/f16_factory.py` dispatches T1 legacy/new-source, T2 and T3 without reinterpreting target contracts. The independent T3 build can be invoked through its full brief/manifest; four-cell source-bootstrap construction currently uses its dedicated runner, so uniform checkpointed construction integration remains open.

Final verification:64 tests and11 subtests passed; new coordinator files pass Ruff. These tests exclude the old placeholder F16 test file. No parent task is promoted because scientific coverage and dependency/card packaging are separate from test success.

## 4. Next actual work

Complete eight evidence cards per target, including missing support, state/selection confounding and disqualified attribution. Connect F08 qualification/history and F12 transport only when their observed variables/target match. Complete the public construction/schema integration and canonical source requirement links. T2's previous failed calibration cell and T1's new failed cell remain visible. F14 data support and F01/F15/F17 whole-scope gaps remain, rather than being reclassified as human-only work.
