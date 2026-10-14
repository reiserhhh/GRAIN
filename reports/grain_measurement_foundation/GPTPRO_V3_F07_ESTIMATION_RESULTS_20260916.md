# F07 full parameter recovery results

## Finding

Expression opportunity, observation error and cost jointly determine the useful measurement design. More opportunities are not universally better once their error and token cost change. This is now supported by actual parameter fits in the declared simulation, not only Fisher-information curves.

The complete run has 1,800 authors, 121 designs and 217,800 MAP fits, elapsed 178.47 seconds. There are 600 application authors across three seeds; the other 1,200 source/validation authors remain separate. This is not a human sample or a fitted personality predictor. The fixed theoretical prior is not re-estimated or selected using these cohorts.

## Application condition-slope MSE, 400-token budget

| World/channel | Level1 | Level2 | Level3 | Level4 | Level5 |
|---|---:|---:|---:|---:|---:|
| U/base | .24710 | .23735 | .23013 | .16596 | .12245 |
| D/base | .06337 | .09193 | .15586 | .20004 | .24907 |
| P/base | .20623 | .17374 | .17079 | .20004 | .25593 |
| P/MNAR | .22709 | .20641 | .18716 | .18693 | .19762 |
| flat/base | .20152 | .21298 | .21323 | .22193 | .22208 |

Paired within-seed author bootstrap (2,000 resamples) gives U level5 minus level1 -.12465 [-.14813,-.10252], D +.18570 [.16167,.21197]. P level3 minus level1 is -.03544 [-.05443,-.01608], whereas level5 is worse by +.04970 [.03498,.06561]. These unadjusted individual contrasts are descriptive finite-world evidence; they are not familywise simultaneous tests or application-selected production settings.

MNAR yields nearly tied point MSE at levels3/4; do not declare level4 an established optimum from that small difference. The full missingness law contributes information about the parameter even when observed content is noisy. This does not mean silence can be treated as a zero score: its meaning depends on the explicit opportunity/selection model.

The flat world's information was constructed equal per token at one parameter point. It does not imply equal finite-budget Bayes risk throughout the population. Its level5-minus-level1 MSE difference is +.02056 [.00393,.03885]. Integer budgets, parameter-varying information and finite-sample prior effects distinguish that risk from a local Fisher calculation. Their individual contributions have not been isolated here.

## Numerical evidence

Independent channel algebra reproduces all 217,800 count tables, objectives and squared errors. Maximum objective difference is 1.42e-14. All 217,800 covariances were subsequently reconstructed with numerical probability derivatives and explicit 2x2 inversion: maximum difference 5.71e-11, within the declared numerical-derivative tolerances. All are positive definite.

12,893 optimizer non-success flags remain in the outputs, not deleted from results. The corresponding gradient median is 2.35e-8, 95th percentile 1.08e-7 and maximum 6.43e-7. Independent Powell refits of the first three sorted authors/all121 designs give maximum absolute objective gap 2.84e-14. This sample supports numerical stopping rather than a large objective failure there; it does not replace full optimality certification.

The summary contains 242 application metric rows, 726 seed rows and 192 level1-reference paired intervals. Coverage Wilson intervals and average widths are retained. All 968 MSE values replay exactly from parameter errors; the 192 paired intervals replay through bootstrap multiplicity weights with maximum difference 2.22e-16. This same-agent alternate-formula audit is saved at `estimation_summary_audit_r1`; it does not constitute third-party replication. Wilson/width table entries have not received an additional independent aggregate replay.

## Production implications and next dependency

1. Define a target-specific observation channel before claiming one expression format is more informative. Include opportunities, reader error, missingness and cost in the same law.
2. Distinguish local information from actual finite-sample recovery. A locally flat information design is not a guarantee of identical individual estimates.
3. Use source-only selection for later deployment; this application comparison characterizes declared designs and must not become an implicit tuning set.
4. Next connect the channel to F08's actual synthetic-text reader, including qualification, polarity and source-transition filtering. The present simulator assumes known channel error and cannot substitute for that text-reading mechanism.

Artifacts: `results/grain_gptpro_foundation_v3/F07/estimation_full_r1`, `estimation_full_audit_r1`, `estimation_summary_r1`. The code/design and smoke details are in `GPTPRO_V3_F07_ESTIMATION_BUILD_20260916.md`.

F07 task-contract integration and other F00-F17 requirements remain open. No API, neural training, private raw data, MEPS fitting, remote operation, lockbox, commit or push. This packet has no active background job; the global research goal remains ACTIVE.
