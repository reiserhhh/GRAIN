# S15: covariance needs the attention-mass control

2026-09-08. COMPLETE: all213 source N authors,3views.639 prefix calls,
0 suffix calls,0 backwards,583.01seconds total including the24-author pilot.
Same S14 common queries and S12 affine readers; no fitting or recalibration.

## Main result

The joint-covariance approximation preserves much of the exact reader's ordering
only when it retains the true attention mass assigned to each comment. It is
a useful mechanistic simplification, not yet a drop-in replacement: a systematic
offset and residual variation remain. A Gaussian approximation to comment
normalization makes the readout worse. Keep the exact common-query route active.

| Formula | r with exact reader | MSE/exact author variance | Teacher-C r | Teacher-C MSE/null | Source N r |
|---|---:|---:|---:|---:|---:|
| Exact pairing |1.0000|0|.315627|.908661|.133706|
| Uniform mass + joint covariance |.415440|1034.2793|.027614|211.8476|.108190|
| Exact mass + joint covariance |.847179|.701136|.287099|.995844|.104370|
| Gaussian mass + joint covariance |.594964|3.797855|.161011|1.685039|.063477|

The huge uniform-arm error is retained rather than repaired by rescaling.
Source N delta r for exact-mass covariance versus exact pairing is-.029336,
paired95%CI[-.100256,.043057]. There is no source-accuracy gain. Internal
approximation fidelity and human association are separate columns throughout.

## What the four arms actually changed

Write the exact head vector as sum_G P_G D_G, where P_G is the fraction of
attention assigned to comment G and D_G is its attention-tilted value mean
minus its ordinary value mean. Approximate D_G by Cov_G(V,K)q.

Uniform weighting also replaces P_G by |G|/n. Exact-mass weighting preserves
P_G. Gaussian weighting approximates log Z_G by its first two cumulants,
retaining all outside-body denominator terms exactly. These are declared
mathematical substitutions, not text cleaning or psychological noise removal.
The common query, token support, fixed reader and author folds never change.

The result localizes two effects:
- Assignment of total probability mass is indispensable for this approximation's
  scale; an unweighted joint covariance is not a sufficient replacement.
- Conditional covariance captures a substantial but incomplete part of relational
  reading. Second-cumulant normalization does not preserve the full token-tail
  competition here, and no Gaussian assumption is adopted for future scoring.

The exact-mass arm's MSE=.000034828 decomposes into squared bias=.000020761
and centered error variance=.000014067. Its mean error is+.004556. This is a
descriptive identity on frozen scores, not permission to correct the bias using
the evaluation sample. Any fitted calibration would need a new training-only
protocol. Prefer examining conditional higher moments before such recalibration.

## Next discriminating question

Attention mass can be factored as P_G=B*r_G, where B is total body attention
and r_G is allocation among body comments. S15 changes both together. A small
next experiment should retain B while replacing r_G by comment token proportions.
Compare both exact conditional reading and covariance under that operation.
This distinguishes whole-head gain from which comment is selected, then asks
whether the remaining conditional error needs a second cumulant.

Primary-source reasoning and a specific normalization mechanism are recorded in
`docs/GRAIN_ATTENTION_MASS_MECHANISM_NOTE.md`. Attention-sink research motivates
the test; S15 has not measured or identified sink tokens. No need for a new
ordinary-embedding grid or a full layer scan.

## Execution and provenance

- Fixed24-author pilot:72 calls,67.17seconds; no outcome access. It found
  exact-mass covariance r=.7730,MSE/variance=.7419, warranting the predeclared
  full cohort for a meaningful mechanism comparison. This was not a significance
  or questionnaire-driven early-stopping decision.
- Full run exact reader replay maxerror3.47e-17. Per-author scores were frozen
  and hashed before joining existing source N measurements. No questionnaire
  fitting, bridge/MEPS/lockbox, API, LoRA, or model download.
- Code/protocol committed as050c591; resource promotion recorded as3c6a5e3.
- Output: `results/grain_vector_mechanisms/conditional_moments_s15_r1/` (ignored):
  manifest, private packets/scores, pilot_024.json, run_status.json, metrics.csv,
  source_paired_intervals.csv and summary.json. All213 authors are included.
- Reproduce with `scripts/run_grain_conditional_moments.py
  prepare|run|inspect|evaluate --output-dir <new ignored output directory>`;
  run defaults to24authors, resume with `--max-authors 213`. Existing completed
  analysis is immutable. The project venv supplies NumPy/SciPy/PyTorch.

Theory: `docs/GRAIN_RELATIONAL_READOUT_THEORY_V4.md`, section8; conditional
expectation and log-partition cumulants follow
[Wainwright & Jordan2008](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/WainwrightJordan2008.pdf).
Three dedicated tests cover closed-form symmetric groups, the zero-query
derivative, and actual multihead pairing reconstruction. The current focused
mechanism suite totals67 passing tests, two existing SWIG warnings; Ruff passes.
