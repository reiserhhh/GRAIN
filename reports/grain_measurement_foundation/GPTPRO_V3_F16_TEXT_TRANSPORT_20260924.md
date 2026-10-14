# F16-R03-TEXT: actual reader replacement and pilot calibration

Date: 2026-09-24 01:45 JST. **The frozen finite text-transport experiment is
executed and numerically audited.** Full F16/44-question acceptance remains open;
the next implementation is public serialization of complete calibration mixtures.

## Main finding

A better qualification classifier is not automatically a better unchanged meter.
Replacing the historical GRAIN qualification head by the fixed NLI qualification
query increases application-material accuracy from .820 to .905. Yet future-target
T3 MSE increases from .407543 to .485925 when the old observation law is retained.
Independent pilot calibration reduces it to .416966. The result separates
reader accuracy, reader-error direction and target scoring, using actual new texts
and actual local neural readouts, not numeric transformations called text transport.

T1 also benefits from calibration; T2's small MSE change has an interval spanning
zero. None of these correlations measure new human personality accuracy. Targets
are the original simulated conditional contrast/current state/planned future response.

## Frozen design and execution

- 380 new controlled English materials: 80 source calibration; 100 paired pilot,
  first60 fit/last40 validation; 200 application. Hashes/role-specific wrappers are
  disjoint, including from the old96 bank. Action/status core grammar is shared.
- Old reader: pinned NLI polarity plus frozen GRAIN comment-view qualification.
  New reader: identical NLI polarity plus its already-fixed actual-self NLI query.
  No new readout fitting, prompt tuning, API calls or neural training.
- Actual computations: 1,140 local NLI premise/hypothesis pairs and 380 frozen
  GRAIN TF-IDF/SVD24 transforms/head evaluations; reading took20.61seconds.
  This is a historical GRAIN view, not the complete GRAIN system.
- Each original seed reuses all200 frozen numeric-source models and200 application
  authors. T1 has600 application authors; T2/T3 share their600. Histories use32/48
  observations and the unchanged (+1,-1,+1,-1) future plan.
- Six arms, three targets, three seeds:10,800 prediction rows. All arms use the
  same new text addresses for a given author/target. About117seconds per full seed.
  Numeric source fitting, old bundles and old result files were not modified.
- Source/reader cut integration pairs200 source models with200 independent channel
  draws. It is Monte Carlo integration of the independent product law, **not**
  the40,000-component Cartesian integration used in numeric transport. Source and
  reader spread are combined here. Raw target units are compared; historical
  source-reference positions are not reinterpreted as freshly calibrated norms.
- The pilot consists of100 MATERIALS, not100 people. Three simulation seeds share
  this material bank. Application truth is read only by a separate evaluation
  phase after all arms finish scoring. The application-pool channel and true-label
  arms are explicitly information references, not operational calibrators.

## Results

Means below average three equal-size seeds, not pooled correlations. Intervals are
working moment-normal95 intervals under the declared source/observation model.

| Reader / observation law | T1 MSE | T2 MSE | T3 MSE |
|---|---:|---:|---:|
| Old reader / source-calibrated old law | .642989 | .352209 | .407543 |
| New reader / old law | .604832 | .347040 | .485925 |
| New reader / pilot plugin law | .583129 | .345690 | .425425 |
| New reader / pilot mixture law | .575736 | .344912 | .416966 |
| New reader / known application-pool law (reference) | .549041 | .334558 | .371666 |
| True pole and qualification (reference) | .483576 | .318189 | .340484 |

| Target | Pilot-mixture minus uncalibrated MSE [95% paired CI] | Primary r | Primary coverage / width |
|---|---|---:|---|
| T1 conditional contrast | -.029096 [-.047961,-.009210] | .655707 | .973333 /3.244678 |
| T2 current state | -.002128 [-.006878,.002028] | .361667 | .960000 /2.340981 |
| T3 planned future response | -.068959 [-.106158,-.033384] | .828552 | .960000 /2.652443 |

T1/T3 reductions are4.81%/14.19%. T2 is not promoted as improved. The2000-draw
paired bootstrap resamples authors within seed; it conditions on the reused SIM
authors, fitted sources and shared finite material pool. It does not quantify
generalization to unseen natural-language families or human populations.

The known-pool reference remains better than pilot mixture: MSE gaps .026695
[.006210,.048358], .010354 [.002294,.017903], .045301 [.026644,.062756] forT1/T2/T3.
Thus the finite pilot has not identified the application observation law exactly.
Known-pool also remains worse than perfect labels; accurate calibration cannot
reconstruct each individual item of information already lost by the reader.
These references do not prove an estimator-optimal lower bound.

## Mechanism: directional qualification error enters the likelihood

For true polarity b, qualification q and observed category c, estimate E[b,q,c].
The unchanged observation law is

```text
P(c | m) = .5 * ((1-Phi(m/sigma))*E[0,1,c] + Phi(m/sigma)*E[1,1,c])
           + .25 * (E[0,0,c] + E[1,0,c]).
```

The three categorical rows for illustration are empirical, before the Jeffreys
prior and draws actually used in scoring. For c=3 (positive and apparently own):

```text
old source reader law: .5*Phi(m/sigma) + .0625
new pilot reader law:  (11/30)*Phi(m/sigma)
```

The old reader has positive-unqualified false positives; the new reader instead
misses some positive-qualified statements. The same observed category therefore
implies a different likelihood for m. Reusing the old law is not a harmless
renaming of a probability. T3 bias moves from -.236332 uncalibrated to +.106645
calibrated; it is reduced in absolute size, not eliminated.

Application content-pole accuracy is1.0 in this literal controlled bank. New
qualification errors concentrate in14/50 positive-qualified statements and5/50
negative-unqualified statements. Pilot negative-unqualified false positives are
5/15 versus5/50 in application; positive-qualified misses are4/15 versus14/50.
The pilot law is directionally useful but finite material/family differences remain.
We have not isolated a unique linguistic cause of that difference.

All empirical4x4 channels retain full rank and the eligible polarity contrast
has total variation1. This is not total content collapse; rank alone nevertheless
does not guarantee precise individual latent recovery. Four deliberately missing
pole/qualification-anchor controls return MISSING_ANCHOR before smoothing.

Production rule: retain joint content/qualification error probabilities and assess
their effect on each declared target. Neither average classifier accuracy, raw
score correlation, nor a wider interval substitutes for the correct observation law.

## Verification and numerical scope

- 30 focused tests pass; Ruff clean; smoke and all full scoring processes finished.
- Audit r2 checks215 artifact hashes, independently reconstructs counts by
  crosstab, raw NLI ratios, GRAIN head outputs, all10,800 archived target values,
 54 metric cells and15 paired intervals. No primary result was changed.
- All600 T1 primary scores are recomputed by hidden-label enumeration and raw
  second moments; first/last dynamic authors per seed use a distinct explicit
  truncated-normal-mixture filter and future covariance calculation across all200
  source models. The check covers6 dynamic authors, not an independent full run.
- 12 fixed input objects replay in fresh processes. In total102 numerical checks,
 14,323 scalar values, maximum difference1.67e-14. This is same-agent alternate
  mathematics, not independent-human construction or psychological validation.
- Doubling quadrature at fixed first/last IDs changes scores at most .001470(T1),
 .000219(T2), .001465(T3). Independent channel-draw replacement changes them at
 most .008602/.019572/.018653; Monte Carlo error is not claimed to be zero.
 Existing ADF approximations and small fixed material scope remain explicit.
- Audit r1 was successful but narrower; r2 adds archived-truth and paired-CI
 replay. Both are retained. No failed scientific arm was overwritten or rerun
 to improve its outcome.

## Artifacts and next action

Protocol: `docs/GRAIN_F16_TEXT_TRANSPORT_PROTOCOL_V1.md`.
Module: `grain_research/gptpro_foundation_v3/f16_text_transport.py`.
Runner/audit: `scripts/run_grain_gptpro_f16_text_transport.py`,
`scripts/audit_grain_gptpro_f16_text_transport.py`.
Canonical results: `results/grain_gptpro_foundation_v3/F16/text_transport_full_r1/`.
Audit: `results/grain_gptpro_foundation_v3/F16/text_transport_audit_r2/`.
Test artifact: `results/grain_gptpro_foundation_v3/F16/text_transport_tests_r1.xml`.

Reproduce into a NEW directory using runner stages `prepare`, `read`, `calibrate`,
`smoke`, `score` (which starts separate `evaluate`). Then run the audit with
`--run <new-directory> --output <new-audit-directory>`. Use the existing shared
venv, `PYTHONPATH=.`, and single-thread BLAS. No routine rerun is needed.

Next: serialize numerical/text calibration mixtures through the public constructor
and cold interface, including versioned estimators, reference policy and reader
identity. Attach this evidence to R22/R23/R24/R25/R34 without promoting their whole
parent cards. No new prompt/reader/coverage grid; full44+6 scope remains ACTIVE.

The research payloads inherit legacy base provenance fields; current reader/cache
and calibration identity are bound by this packet's outer protocol/manifests.
They are not newly authorized public bundles merely because the internal scorer
accepts them. The next serializer must bind that new identity explicitly and must
not reuse a legacy inner reader hash as the identity of this changed observation law.
