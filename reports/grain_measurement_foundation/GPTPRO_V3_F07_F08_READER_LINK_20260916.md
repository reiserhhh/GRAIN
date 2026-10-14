# F07/F08 measured observation-channel link

## What is now connected

F07 no longer only assumes a reader error rate. The frozen F08 material bank supplies a measured stochastic emission matrix E(y,s)=P(reader symbol s|personal response y), with symbols missing/positive/negative. A latent response p(theta,z)=logistic(a+bz) gives

P(s|theta,z)=(1-p)E(0,s)+pE(1,s),

dP(s)/dtheta=[E(1,s)-E(0,s)]p(1-p)[1,z].

The categorical Fisher matrix follows from sum_s dP_s dP_s'/P_s. This accommodates asymmetric qualification and polarity errors instead of forcing one symmetric flip parameter. Matrices are fitted only from source materials, with the declared status/variant design weights. Unqualified content poles are averaged independently of personal y, not relabeled as personal truth.

## Lexical reader result

Source emissions for the mixed panel are [[.5,0,.5],[.5,.5,0]]; complete emissions are [[0,0,1],[0,1,0]]. Therefore mixed information is exactly one half of complete information under this stipulated half-eligible design. At (a,b)=(0,.5), efficient slope information is .117502 vs .235004 per opportunity. Validation/application framing families reproduce these hard emissions exactly. This is not an empirical estimate of the prevalence of personal evidence in human conversations.

Six information matrices and36 three-step derivative checks are saved per reader. The independent scalar-mixture information identity is checked. Four tests cover the missing-Bernoulli reduction, false qualification of other-speaker content, asymmetric derivatives and source-only fitting.

## Offline NLI comparator

Used the already available pinned `cross-encoder/nli-deberta-v3-base`, revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`, offline FP32 CPU with four threads. Three exact hypotheses were frozen before inference for the same96 public synthetic materials: positive stance, negative stance, and own actual decision. 288 pairs completed in7.20 seconds. No API, training, private text or prompt revision occurred.

Application polarity accuracy31/32, Brier .032325; qualification accuracy19/32, Brier .353283. All8 quoted-other items are falsely eligible. Hypothetical eligibility is rejected correctly; correction eligibility is only5/8. The polarity ratio uses positive/(positive+negative) entailment, not a calibrated personal-trait posterior; raw three-class readings are retained.

NLI source mixed emissions are [[.375,0,.625],[.75,.125,.125]]. Missingness thus depends on personal response polarity in the declared synthetic channel. A single constant presence and symmetric flip parameter cannot represent it. Validation/application emission discrepancies reach .1875, while complete qualified hard emissions remain identical. Do not replace the frozen source model with application confusion to erase this shift.

This links an observed reader failure to the MNAR/asymmetric channel mechanism. It is a failure of this fixed NLI query/model combination, not proof that all semantic readers must fail or that ownership is unsolvable. No wording rescue or threshold tuning is performed.

## F08 state verification

The lexical downstream packet additionally passed a separate all-state audit: six source bundles reconstructed using augmented least squares and source table SVD;1,944,000 predictions replayed using explicit eight-state enumeration and weighted least squares, maximum difference1.58e-14. No production filter/scorer was imported by the audit. This is same-agent alternate-formula verification, not independent third-party replication.

## Artifacts and remaining work

- F07 lexical link: `results/grain_gptpro_foundation_v3/F07/reader_link_r1`.
- F07 NLI link: `results/grain_gptpro_foundation_v3/F07/nli_reader_link_r1`.
- NLI fixed inputs/raw outputs: `results/grain_gptpro_foundation_v3/F08/nli_reader_r1`.
- Lexical state audit: `results/grain_gptpro_foundation_v3/F08/state_full_audit_r2`.
- NLI downstream: `results/grain_gptpro_foundation_v3/F08/nli_downstream_full_r1`; full computation completed518,400 updates/54,000 author metrics. Both separate audits completed:1,944,000 proper losses/evidence masses replay with max2.22e-16, and six source bundles/1,944,000 state-derived scores replay with max1.61e-14. No active job remains from this packet.

NLI complete XOR B1 is .189484 soft versus .189487 oracle; mixed XOR is .250916 soft/.257020 hard versus .227777 oracle. Mixing eligibility uncertainty changes the result markedly despite strong polarity reading. This comparison does not by itself isolate polarity and qualification contributions because the panels also change material composition. The next bounded attribution should independently substitute oracle polarity and oracle qualification within the same mixed observations, without retraining or prompt tuning. Confidence intervals across these simulated author histories do not quantify new-grammar or reader-training uncertainty.

The five theoretical F07 openness/cost levels are not five measured text conditions in this bank. No fabricated per-level c/epsilon/token-cost data are supplied. The measured channel interface is established; actual channel-level design/cost mapping and F00/card integration remain separate. Full NLI downstream audit, cross-reader production-rule synthesis, F11 and other global requirements stay open. No human validity or total-theory completion is claimed.
