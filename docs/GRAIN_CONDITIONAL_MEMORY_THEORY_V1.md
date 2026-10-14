# Conditional memory: information, reference and observation-law mismatch

HM282,2026-10-14 JST. Canonical `conditional_memory_r1`; one frozen finite
natural-history experiment,4320 late programs on120 authors. No accuracy tuning,
new probability theorem or new human sample. Parent HM281 and its failures
remain unchanged.

## 1. What is observed and what is kept

The inherited event alphabet is a=4c+s, where c is a coarse TOP20 venue bit and
s is two anonymous GRAIN coordinate bits. The original text encoder is neural.
Both choice and expression are retained as parallel observations; neither is
declared noise. Time orders events, but the two windows have no artificial edge.

Use the exact SAME frozen SOURCE fixed posterior-mean rows mu_i,P_i. For a
coarsened view V, its probability is a sum over all compatible complete paths:

    L_i(V)=sum_(a_0:T compatible with V) mu_i(a_0)*prod_t P_i(a_t,a_t+1).

The finite forward recursion calculates this sum exactly to floating arithmetic.
The observed old hidden a_t is NOT a transition input for condition-only or
expression-only observers. Fixed-row coarsening is not integration over an
unknown shared Dirichlet kernel; that broader operator remains OPEN.

## 2. Natural conditioning and a fixed-condition design are different measures

Let LF,LC,LS denote full, condition-only and expression-only likelihoods. With
the fixed uniform SOURCE reference pi:

    w_i(F) proportional to pi_i*LF_i
    w_i(C) proportional to pi_i*LC_i
    w_i(S) proportional to pi_i*LS_i
    LF_i=LC_i*L_i(S|C)=LS_i*L_i(C|S).

Updating on C and then on S|C exactly reconstructs full natural Bayes inference.
Starting with the ORIGINAL uniform pi and using L_i(S|C) alone instead makes
a different experiment: C is fixed by a hypothetical external design and does
not reweight the reference catalogue through its natural choice likelihood.
This is a legitimate conditional observation contract, not identified human
do(C), and not "subtracting topic noise". Its symmetric C|S arm is a design
control. All six contracts have executable scores and stay available.

LC*LS does not usually equal LF: the two views of one event are dependent even
within a reference author. Treating them as independent observations loses the
condition-expression linkage. The invalid-product/full posterior TV averages
.5489/.5616/.5740 for bank0/bank1/pooled. This finite discrepancy concerns the
declared reference model, not a percentage of psychological information.

## 3. A reusable relation between information and forecasting

Let F be complete early evidence, C a deterministic coarsening, Z the first
late state, L the rest of the late episode, and P the actual joint law. Let
Q_F and Q_C be the model's conditional late predictive laws. Define

    D_F = E_(F,Z) KL(P(L|F,Z) || Q_F(L|F,Z))
    D_C = E_(C,Z) KL(P(L|C,Z) || Q_C(L|C,Z)).

Adding/subtracting the true conditional log laws gives the established identity

    E_P log[Q_F(L|F,Z)/Q_C(L|C,Z)]
      = I_P(L;F|C,Z) - D_F + D_C.

With correct predictive laws, finer observations cannot reduce expected log
prediction because conditional mutual information is nonnegative. When an
estimated SOURCE law is transported to actual authors/occasions, a coarsening
can forecast better by reducing law mismatch MORE than it loses information.
That is not proof that erased coordinates were noise. We do not know P or the
three terms separately in these natural finite data; the identity supplies a
mechanism distinction, not an estimated decomposition of the observed delta.

The same rule explains why a correct numerical Bayesian implementation can
have negative empirical history carry without Bayes' theorem failing. Sparse
SOURCE estimates, catalogue overcommitment, context-dependent response and
actual changing expression laws are competing causes, not already identified.
Do not adjust posterior temperature/decay to conceal this distinction.

## 4. Actual finite comparison

All386 SOURCE and the same120 held authors/two18-event paths; no held criterion
labels. Pooled changes in mean late TRANSITION log score, nats/transition:

|Early contract|Own minus fresh|Own minus start-matched stranger|
|---|---:|---:|
|full|-.058870|+.117598|
|condition|-.005713|+.036432|
|expression|-.015515|+.075362|
|expression_given_condition|-.056499|+.084148|
|condition_given_expression|-.042196|+.057430|

Coarse C-only/S-only losses are smaller than full in both source banks. But the
conditional-expression design remains close to the full loss in all references
and is WORSE than natural S-only by .040984 pooled [.021718,.059179] in loss
magnitude. Full versus conditional-expression differences are small and have
both signs across banks. Removing the natural condition-choice reweighting
alone therefore does not solve the actual fixed-law carry problem.

Every own-versus-stranger finite mean is positive in both banks and pooled;
coarsening does not remove all author relevance. More relevant and better
transported are separate properties. Strangers are only early-first-state
matched, not full topic/time/demographic matches; banks reuse the same held
authors. Conditional exploratory percentile ranges are not family-adjusted
confirmation or independent-donor population coverage.

Early effective SOURCE modes pooled: full4.408, condition123.514,
expression14.781, conditional-expression9.563, conditional-condition34.816.
Concentration coincides with some losses but does not establish overconfidence
as the unique cause. No norm/entropy target, arm winner or universal filter is
promoted. All32 SOURCE profile combinations and both historical anchor
assignments remain in early/late outputs. These are model-based readings,
not current human personality or emotion truth.

## 5. Connection to GRAIN's measuring architecture

The new cold chain is:

    qualified numeric events
      -> full / coarsened / explicitly conditional observation likelihood
      -> fixed-reference belief
      -> complete SOURCE-anchor profiles and next-event predictions.

This connects WHERE choice and WHAT expression to WHO-relative reference belief
and WHEN updating without forcing them into independent channels. It also
requires every score to carry its observation AND reference-design contract.
A identical target name is not enough to merge two scores from different laws.

The mainline bridge to a neural personality-query action is still separate:
HM280 uses different48 comment pairs. No combined causal chain is inferred from
these unrelated material panels. New capability here is exact masked-history
scoring and a real discriminating result, not stronger Big5 prediction.

Next inventory existing SAME-author episode/condition kernels and source
observation operators before another finite construction. A useful next
mechanism must distinguish which source-supported author property is shared
across occasions from an episode-specific expression law, with a target-linked
function retained. Not another prior/decay/weight/state/length/accuracy grid or
known-identity toy world. Original18/100/44 remains OPEN; no next job queued.

## 6. Prior mathematics and verification

[Stan's HMM reference](https://mc-stan.org/docs/functions-reference/hidden_markov_models.html)
documents discrete hidden-state marginalization. Conditional Bayes, KL and
information decomposition are standard; not novel theorems. Parent kernel
ancestry: [Fortini/Petrone2015](https://arxiv.org/html/1406.5421v3).
[Pinned OpenAI/math foundations166-232](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
motivated dependence-aware likelihood composition and observation degradation.
Its tree independence, symmetry and threshold hypotheses are NOT transferred.

One7.641913s finite computation,1080 early masked forward views/4320 late
program descriptions of120 people, no new model/API/private/train. Independent
LOG-space audit on all early views and late arms: forward4.97e-14,
late log likelihood2.93e-14, profiles1.10e-14, old full/fresh/stranger replay
1.20e-14, cold CLI0.22 focused tests(13 new) and4Python lint/format PASS.
All stages/seals pass; no scientific retries/postfreeze edits. Same-agent
alternative calculation, not third-party blind replication.
