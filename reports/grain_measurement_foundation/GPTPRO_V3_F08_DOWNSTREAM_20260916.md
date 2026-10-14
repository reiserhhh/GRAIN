# F08 actual downstream channel comparison

## Result and mechanism

Reading the correct class and supplying a well-calibrated probability to a measurement operator are different requirements. The revised transparent synthetic bank classifies every pole and eligibility correctly, so hard readings reproduce the metadata oracle. Soft probabilities remain underconfident and can attenuate real transition structure. In an IID world, the same attenuation can reduce overfitting instead. Neither soft nor hard is universally the correct production choice.

Application Brier losses, 600 authors per world:

| World/panel | Estimator | Oracle/hard | Soft |
|---|---|---:|---:|
| XOR/complete | B1 | .189487 | .210764 |
| XOR/complete | B2 | .210892 | .226195 |
| XOR/complete | M | .201204 | .229247 |
| XOR/mixed | B1 | .227777 | .236620 |
| XOR/mixed | B2 | .245379 | .238939 |
| XOR/mixed | M | .234121 | .239545 |
| IID/complete | B1 | .250149 | .250139 |
| IID/complete | B2 | .278779 | .258702 |
| IID/complete | M | .261010 | .252937 |

Static source-mean and personal-prefix-mean references are included in the full table. The source mean is about .25 in both worlds. Soft B2 improves some losses by shrinkage without creating new latent signal. In complete XOR, its Brier worsens while its log loss improves (.706305 to .646755); keeping both proper losses avoids a misleading single-score verdict.

## Protocol and computation

Canonical results are `results/grain_gptpro_foundation_v3/F08/downstream_full_r2`. Two frozen C2 worlds, three seeds, 300 source/100 validation/200 application per world/seed, 48 events and prefix12. Source B1/B2/M are fitted from unchanged oracle source observations and shared by all readout channels. This isolates application observation error; it is not an end-to-end source-reader-error experiment.

Two panels use the same latent histories. Complete uses only actual/corrected eligible statements and independent reader-error marginal products. Eligibility is known to be one in this panel, isolating polarity uncertainty. Mixed draws all four statuses independently; unqualified statement poles are ancillary independent bits, not hidden personal responses. Its source-transition filter retains every time position. Oracle uses metadata eligibility, soft uses q and p separately, hard uses thresholds .5.

B2 accumulates expected feature outer products and expected feature-times-response, with the original source-centered ridge penalty1. M accumulates expected four-cell counts and retains its original smoothing and source SVD projection. B1 is a frozen source transition table, not an individually fitted table. The complete oracle adapter matches the original three estimators over460 sequential prefixes; independent soft four-cell moments and missing-time semantics have explicit tests.

The complete run produced 518,400 updates, 172,800 observation rows, 54,000 author metric rows and eight actual evidence cards. Before/after card predictions refer to successive next-response targets; their difference is not a psychological treatment effect. Record identity is scoped by panel/channel, which represent alternate observations of the same simulated event.

## Preserved correction

`downstream_smoke_r1` used source filtering also in the complete panel. `downstream_full_r1` was interrupted after detecting that protocol mismatch. Both remain noncanonical evidence; no generated values were overwritten. Revision2 separates independent complete updates from mixed-history filtering. The old legacy F08 runner is also retained, not treated as a successful downstream comparison.

## Audit and remaining work

Smoke r2 replays25,920 predictions,720 author metrics and80 paired channel contrasts. Full-run audit at `downstream_full_audit_r2` completed:1,944,000 predictions,54,000 author metrics,80 paired contrasts, maximum proper-loss difference1.11e-16. Every causal evidence mass was checked. Hard/oracle scores agree exactly, not merely by a nonsignificant test; the generic bootstrap decision column labels zero intervals inconclusive, whereas the separate pointwise identity check establishes equality in this bank.

For complete XOR B1, soft-minus-oracle Brier is +.021277 [.019441,.023116]; B2 Brier is +.015303 [.012948,.017581], while its log loss improves -.059549 [-.072194,-.047179]. These are 2,000 paired author resamples within three seeds, not independent-event intervals or simultaneous familywise claims.

Full independent soft-state updates and source-fit replay, frozen output hashes, F07 reader-channel connection, second authorized reader availability, standard card/DAG artifacts and final production-rule synthesis remain executable work.

The material bank has shared semantic core phrases and only disjoint wrapper families, so this is a controlled reader-channel mechanism demonstration, not natural-language generalization or human personality accuracy. No API/neural training/private raw text/MEPS/remote/lockbox/commit/push. Global goal remains active.
