# F16 T1: source-frozen construction and cold execution

## Result

Replaced the legacy random-score/literal-True approach for the T1 execution subtask with an actual four-cell conditional response instrument. Three source models each contain 300 synthetic source authors; 600 application authors were scored across three seeds. Cold subprocess scores match warm scores exactly. Independent observation-space linear solves differ by at most 1.3877787807814457e-14.

This is SIM-only construction evidence, not human measurement validity or completion of F16. The cold process is a new interpreter using the same implementation, not an independent builder agent.

## Construction and equation

The brief specifies T1 = m(+1)-m(-1), with four ordered conditions (-1,-1/3,1/3,1). The constructor verifies source/model hashes and source-only record roles, preserves the original four-cell model metadata, and computes source-author contrast scores before freezing their mean and sample standard deviation. Existing source covariance flooring is retained with its raw eigenvalue metadata; it is not relabelled as a newly unconstrained covariance estimate.

For source moments (mu,V), condition incidence X and residual variance sigma2, K = X V X' + sigma2 I. The posterior working mean is mu + V X' K^-1 (y-X mu), covariance V - V X' K^-1 X V. Contrast c=(-1,0,0,1) yields score c'm and variance c'P c. Reference score is (score-reference_mu)/reference_sigma. Scores never use the application batch to compute norms.

With missing endpoints, output is explicitly model_borrowed; with no records the model supplies a prior-only contrast. This is not observed personal evidence. Numerical covariance validation also applies to the empty-record case. Bundle identity, world, reader, scorer hash, formula version, reference scale and duplicate event checks are enforced.

## Execution evidence

Artifacts: `results/grain_gptpro_foundation_v3/F16/t1_cold_full_r1/`.

Each seed folder contains the complete self-contained bundle, verified source manifest, application hash and author order, and actual cold scores. A temporary working directory supplied the subprocess only bundle and input as data; the installed source modules remain available through PYTHONPATH. This is a declared-data isolation test, not an OS filesystem sandbox. The CLI requires an externally supplied trusted bundle SHA256 and does not read source or truth files while scoring.

| Seed | Source authors | Application authors | Cold/warm maximum error | Direct-solve maximum error |
|---|---:|---:|---:|---:|
| 20260915 | 300 | 200 | 0 | 1.088019e-14 |
| 20260916 | 300 | 200 | 0 | 1.243450e-14 |
| 20260917 | 300 | 200 | 0 | 1.387779e-14 |

22 related tests pass, including trusted-hash rejection in a separate subprocess, extreme-stranger/batch invariance, missing endpoints, duplicate events and invalid covariance. The brief was written before this runner opened application data; these are previously analyzed historical simulations, not a claim of prospective preregistration.

## Remaining work

- Four-cell source estimation uncertainty is not integrated; every score says source_uncertainty_included=false and calibration_check=NOT_EVALUATED. Earlier linear-delta calibration cannot be borrowed for this different estimator.
- T2 current-state and T3 four-step mean construction remain open, including their joint innovation covariance and F13 source mixtures.
- Eight actual evidence cards per target and independent builder-agent reconstruction remain open. Same-agent cold replay does not discharge this requirement.
- Transport, text-channel integration, canonical dependency-hash acceptance and other F00-F17 requirements remain in the active goal. No parent completion claim.

No API, neural training, private data, remote job, lockbox, commit or push was used. The full T1 runner is terminal; no background process remains from this packet.
