# HM16: Context Covectors versus Endpoint Potentials

2026-10-08. Planned, not executed. Bounded zero-forward mechanism packet
after verified HM15. Do not replay old model calls or edit frozen files.

## Scientific question

HM15 full vectors recover action, but projection/complement errors persist.
Is their remainder better represented by a receiver-dependent covector,
or by finite curvature of one endpoint potential? Preserve both possibilities
and all failures; the purpose is not to optimize a leaderboard.

## Stage0: freeze inherited states and new absolute observations

1. Verify every HM15 primary hash, its74-file alternate ledger and inherited
   HM10B3/HM12/HM13/HM14 code and material hashes. No old forward replay.
2. Preserve block16,3queries,2172addresses,724statepairs,source1080/
   validation528/application564 and family8/4/4. Keep all7 intervention arms,
   A/T,order,polarity and token opportunity, including exact noops.
3. Reconstruct absolute full position observations r,t with the original
   float32 target arithmetic. Check t-r against frozen HM15 differences;
   keep complete reference/target hashes. Property lengths1--4 only.
4. No question-endpoint states or role/polarity/family labels as model features.
   These selected prefix spans are an observed context proxy, not the entire
   transformer state. The original receiver question states remain fixed
   at each intervention and are not recomputed from a patched property.
5. Store source target and held target files separately. Freeze code,
   configuration, inputs and declared controls before source-only fits.

## Stage1: fixed mathematical candidates

Reuse alpha1,family-balanced source loss,no action intercept. Fit no hidden
neural weights. Per query all candidates use the same360source actions.

1. **Full linear:** exact HM15 source law, not refitted to get a new baseline.
2. **Receiver tensor:** d direct-sum (c_r tensor d), with c_r the frozen
   source-centered/scaled reference observation; k=k_d(1+k_c).
3. **Endpoint quadratic potential:** d direct-sum sym(m tensor d), with
   m=(r+t)/2 in the same frozen source context coordinates. Use the explicit
   two-product symmetric Gram formula; no ambient tensor allocation.
4. **Broken context control:** fixedseed1617 shuffle receiver contexts by
   distinct statepair within each role/axis/order/token-length stratum,
   never by query or arm. Reuse each pair's shuffled context across its
   arms and queries. Keep displacement, target and marginal context fixed.
   Freeze source/held mappings independently before fitting. If a stratum
   cannot change, count it and do not call it a successful permutation.
   Run the receiver tensor on this control without selecting another shuffle.

Before-fit implementation clarification: shuffle pair IDs by one fixed random
cycle per stratum, additionally respecting identical available-arm profiles.
This keeps every arm's empirical context multiset exact, including the six
selected single-token reversal controls. Singleton groups stay fixed and are
counted. This is a stratified cycle control, not an unconstrained uniform
permutation. No unavailable arm reference is invented.

Use a frozen source-family weighted mean for the context origin and one
source RMS for its scale. Keep the linear term: no automatic subtraction of
conditional signal. Scale each new combined kernel by its weighted mean
source diagonal so alpha1 has a stated average source unit. Store both raw
and normalized kernels/scales. This normalization is fixed before cold
actions, not a demand that every model have identical raw regularization.

Freeze query functions, source indices and kernel coefficients before
loading held actions. No rank/alpha/prompt/layer/degree/kernel-grid search.

## Stage2: all addressed actions and mechanism diagnostics

- Report source/validation/application separately for eachquery/model/arm,
  A/T,order,token opportunity. Keep RMSE,bias,r,sign and zero baseline.
- Paired family MSE differences against HM15 and against broken-context
  control with fixedseed1617/bootstrap10000; intervals conditional on frozen
  source functions. Source-estimation uncertainty is a separate scope.
- Compare preservation of full,projection,complement and reverse actions;
  better full action alone does not imply repaired complement calibration.
- Report tensor-versus-linear contribution and its sign at every endpoint;
  no attribution of kernel norm as a percentage of psychological information.
- Verify PSD/symmetry within floating tolerance, orthogonal coordinate
  invariance and exact identical-input zero. Test symmetric potential
  endpoint reversal and three-endpoint telescoping on synthetic vectors.
- Independent explicit tensor/augmented least squares on manageable
  synthetic data; actual source spectral dual replay and all stored actions.
- Preserve the best argument against each candidate; do not turn an observed
  proxy-context benefit into personal B_u or human personality validity.

## Decision and delivery

Matched context advantage over the fixed marginal shuffle supports context
dependence in this observation law. Endpoint potential adequacy favors
integrable finite curvature; receiver-only advantage motivates missing-context
or conditional-operator analysis, not a claim of nonconservative psychology.
No gain keeps the full linear source law and redirects to source support/new
materials rather than more hyperparameters. Mixed results remain target/arm
specific. None closes natural-author or old18/100/44 scope wholesale.

Deliver helper/runner/config/tests,sourcefunctions,cold predictions,all
diagnostics,independent arithmetic,report and resumable parent checkpoint.
Plannedroot results/grain_math_response_v1/context_covector_kernel_r1.
Expected model/API/private/neural-training calls0. No Claude/automation/
commit/push/tag/lockbox/old queues. Evaluate costs before a new forward stage.
