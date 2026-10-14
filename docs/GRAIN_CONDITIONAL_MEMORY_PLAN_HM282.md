# HM282: natural choice versus condition-qualified expression memory

2026-10-14 JST. Freeze BEFORE the one finite CPU experiment. This is a new
observation operator on existing natural events, not an accuracy/forgetting grid.

## Mainline question and competing mechanisms

HM281 found own history better than a start-matched stranger but worse than fresh
late prediction. Does the loss reside chiefly in coarse condition selection,
expression observations, or their joint/reference-dependent evidence? A
choice-dominated explanation predicts the condition-only carry resembles the
full-history loss and expression-only carry is less harmful. An expression-
dominated explanation predicts the reverse. A coupled explanation predicts that
neither marginal can reproduce the complete effect. Conditional design ratios
test a DIFFERENT reference contract, not a rescue of the natural posterior.

No outcome selects weights, priors, state count, features, seed, window length,
model or posterior temperature. All six arms are retained even if all are worse.
No held psychological labels enter construction or evaluation. New capability:
an exact condition/expression observation operator and a cold two-episode scorer
that states which observation/design prior generated its score.

## Reused evidence, not a new human sample

Same HM281386 SOURCE/120 source-disjoint historical held authors, two18-event
windows, three SOURCE references, both old32-profile assignments and identical
first-state-matched stranger mapping. Original event identities and chronology
are preserved. No raw text, private questionnaire, API, model forward, training,
download, Git or lockbox opening. These are historical numeric caches, not
untouched validation or independent human replications.

Inherited state a=4c+s: c is TOP20-venue membership0/1, s is the frozen two-bit
GRAIN coordinate partition0..3. Neither c nor s is a full semantic topic,
situation or emotional state. No semantic names are added.

## Exact observational construction

Reuse HM281 fixed SOURCE posterior-mean initial/transition rows mu_i,P_i. This
choice defines an exact tractable finite base, NOT a best-model promotion from
HM281. Integrated unknown-kernel coarsening remains a separate open problem.
Hidden paths under an integrated Dirichlet law cannot silently be processed as
an ordinary fixed HMM.

For each reference author i and early path, compute full, condition-only and
expression-only likelihoods LF_i,LC_i,LS_i using forward sum-product:

    v_i0(a)=mu_i(a)*mask_0(a)
    v_it(b)=mask_t(b)*sum_a v_i,t-1(a)*P_i(a,b)
    L_i=sum_b v_i,last(b).

Each time step is scaled and its log mass accumulated. A coarsened observer
never uses the hidden actual previous state as a covariate. This avoids a
spurious "condition-only" scorer that still receives expression history.

Six uniform-reference early likelihood contracts:

|Contract|Early likelihood|Meaning|
|---|---|---|
|none|1|No early evidence|
|full|LF|Natural complete early observation|
|condition|LC|Natural coarsened condition observation|
|expression|LS|Natural coarsened expression observation|
|expression_given_condition|LF/LC|Condition externally fixed in a hypothetical design|
|condition_given_expression|LF/LS|Symmetric fixed-expression design/control|

Natural Bayes composition is pi_i*LC_i*(LF_i/LC_i)=pi_i*LF_i, or the reverse
order with LS. Using uniform pi_i*(LF_i/LC_i) alone CHANGES the reference design;
it is neither condition deletion as noise nor identification of human do(c).
Multiplying LC*LS is generally wrong for the natural full observation; record
its discrepancy as a finite arithmetic diagnostic, not an extra winning model.

Carry only early reference-author weights into the actual late full observation.
Late starts anew from mu_i; no cross-window transition and no kernel updates.
Same late event drives every arm; do not multiply overlapping evidence twice.

## Outputs and decision

Save all likelihood prefixes, early weights/full profiles, all late predictions,
log scores, profiles, five marginals and effective-reference counts for self and
the frozen stranger controls. Compare all signed mean transition log-score
changes relative to fresh, self-minus-stranger, expression-minus-condition and
conditional-design-minus-natural-expression. Bootstrap is descriptive conditional
on SOURCE and frozen donor graph; no family-adjusted or independent-donor95%
coverage claim. Keep the start log score separately from17transition means.

The scientific decision is which finite observation/reference contract changes
the old negative carry, NOT which arm is universally best. If all remain negative,
retain this and distinguish failure of fixed reference/kernel transport from
lack of author-relevant signal. Do not automatically add decay or hyperparameters.

## Tests, execution and bounded verification

Before freeze: brute force short hidden-path likelihoods; full-view equivalence
to old fixed model; hidden-bit invariance; conditional chain reconstruction;
marginal-product counterexample; two-episode cold chronology/label rejection;
conditional initial score and no boundary edge. All tests/lint/format must pass
before prepare. One compute<=180s, no scientific retry. Runtime updates use
atomic replacement, not immutable scientific-file creation.

After compute: independent LOG-space forward recursion on all360 early histories,
direct whole-path late likelihood/profile checks for all4320 arms; cold CLI
readback; exact reproduction of old full/self/stranger/fresh fixed-row accounts.
Verification changes no experiment/model/input. Same-agent arithmetic audit,
not blind replication. Preserve failures and original seals. Original18/100/44
remains OPEN and no new job is queued at completion.

## Prior mathematics and project contribution

[Stan HMM reference](https://mc-stan.org/docs/functions-reference/hidden_markov_models.html)
documents latent-state marginalization; this is mature sum-product mathematics.
[Fortini/Petrone2015](https://arxiv.org/html/1406.5421v3) supplies the parent
Markov-mixture ancestry. Conditional probability/Bayes is established, not novel.
[OpenAI/math pinned foundations166-232](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
distinguishes conditional-independent branch composition from observation
degradation. Its tree hypotheses/threshold theorem are NOT transferred.
Project increment is the real author-history discriminator and executable
choice/conditional-expression contract, not rediscovering a Bayes identity.
