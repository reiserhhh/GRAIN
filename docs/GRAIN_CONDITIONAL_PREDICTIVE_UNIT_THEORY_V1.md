# Conditional predictive evidence and context-carrying observation units

HM152, 2026-10-11. Actual finite observations and alternate arithmetic complete.
Not a new probability theorem or a new accuracy-optimized predictor.
The original 18/100/44 scope remains open. No human trait posterior is supplied.

## 1. The object is a conditional contrast, not vector proximity

Fix a task/style condition c and two supplied persona hypotheses H,L. The same
actual article prefix is read under both hypotheses. With final normalized
prediction states h_a(t), output covectors W[v] and any output bias b[v],

    z_a(v,t) = W[v] . h_a(t) + b[v]
    A_a(t) = log sum_v exp(z_a(v,t))
    ell_t = z_H(x_t,t) - z_L(x_t,t) - A_H(t) + A_L(t)
    L(x|c) = sum_t ell_t = log p_H(x|c) - log p_L(x|c).

This retains two accounts: action on the observed token and competition among
ALL vocabulary alternatives. A larger observed-token logit alone need not mean
stronger normalized evidence. No classifier, teacher score or personality word
proximity is needed to evaluate this contrast, but obtaining h_a still uses the
frozen neural model. This packet therefore does not achieve neural independence.

The two accounts are not a unique causal decomposition. Adding any scalar
k_a(t) to all logits changes both z_a and A_a, leaving ell_t unchanged. We retain
the native gauge and a token-uniform centering companion; neither defines a
psychological fraction. A dual change of coordinates h'=hB, W'=WB^{-T} for
invertible B also preserves logits. The normalized predictive law, not a named
coordinate axis, is the invariant object.

## 2. A precise slice composition condition

Let x = (x_1,x_2), with the same native tokens in both scoring procedures.

    L_full = L(x_1|c) + L(x_2|x_1,c)
    L_reset = L(x_1|c) + L(x_2|c)
    K_HL(x_1,x_2;c) = L(x_2|x_1,c) - L(x_2|c).

Then L_full = L_reset + K_HL exactly. The carried path obeys the probability
chain rule. Resetting a second slice starts a different observation law. K is
a difference of log likelihood ratios, in nats, not a correlation, a variance
share or a measure of human personality change.

K=0 is the exact condition for those particular reset units to preserve the
target contrast. Independence of x_2 from x_1 under BOTH hypotheses is sufficient
but not necessary: history effects may cancel in the H/L contrast. Thus generic
context sensitivity is not itself proof that target evidence changes. HM152
tests the difference that matters to the declared target, not only hidden-state
distance or two separate perplexities.

The law is hypothesis/condition dependent. Additivity for one target or suffix
does not imply additivity for another. A prefix length of 32 and split at 16
tests those finite units only. A reset in the middle of a sentence or subword
continuation is a computational intervention, not a natural clinical encounter.

## 3. Implications for the GRAIN compiler

Keep three operations distinct:

- Segment a continuous observation while carrying its required context.
- Reset and independently score a segment, declaring/calibrating that new law.
- Pool already qualified event evidence at the author level.

These are not interchangeable. The exact transformer continuation state uses
past key/value caches, positions and prompt/role bindings. A single final hidden
vector is sufficient for the immediate output projection, not automatically
for future suffixes. A smaller context representation would need its own
target-restricted sufficiency test, rather than being assumed from good fits.

GRAIN choice composition also remains: where the author chooses to speak and
how the author expresses within a condition are separate observation channels.
If a validated author model theta existed, the relevant two-channel account
would be

    log p(c,x|theta_H,o) / p(c,x|theta_L,o)
      = log p(c|theta_H,o) / p(c|theta_L,o)
        + log p(x|c,theta_H,o) / p(x|c,theta_L,o).

HM152 holds c fixed and supplies synthetic writer prompts; it estimates neither
the choice term nor a real author's theta. Event identity, factuality, author
attribution, state, reference populations and behavioral/psychological meaning
cannot be supplied by a high likelihood ratio alone. The existing GRAIN channels
are not discarded as noise or replaced with the writer likelihood.

## 4. Executed finite observation

All 702 historical synthetic articles contribute 972 hypothesis-conditioned
cases (648 matching-persona articles and 54 neutral articles scored under six
target/family hypotheses). All first 32 native tokens were available; split at
16. The two targets, three wording families, six tasks, three styles and three
generation seeds are retained. Seeds are NOT authors. Fourteen source refusals
and 519 generation-length truncations remain; no replacement text was generated.

- Full normalized and raw chosen-logit high-minus-low contrasts differ in sign
  in 18/108 condition cells. Token-uniform centering changes that count to 23/108.
  These gauge-dependent counts do not identify a unique normalizer causal share.
- Reset slice composition reverses the normalized condition contrast in 15/108
  cells, including E/named/review/neutral: +2.350826 carried versus -0.172323 reset.
- Across all 972 cases, mean K=+0.007845 while mean |K|=.661625 nats, median
  |K|=.478257 and 95th percentile |K|=2.010036. A nearly zero signed mean does
  not eliminate unit-level context contributions.
- The normalized full H-minus-L article contrast averages +5.261290, +3.931110,
  +.327350 for E/named, described, anchored respectively, and +.790009, +.928621,
  +.717685 for the three N families. These are contrasts in nats, not AUC/r,
  accuracy, posterior probability or the extent of human personality expression.

Capture completed in 709.819 seconds with 974 batched decoder forwards and 3890
logical sequences, including two causal-prefix checks. All 46,656 token accounts,
972 slice accounts and 108 condition contrasts were re-summed without importing
the scientific account module: 284,475 scalar checks, maximum table discrepancy
0. Stored-vector action replay differs from native chosen logits by at most
5.43e-6; 12 fixed full-vocabulary weight replays have maximum partition error
1.69e-6. The independent arithmetic is same-agent, not a blind replication; it
does not recompute every vocabulary partition. Both causal checks passed the
sealed 1e-4 absolute PLUS 1e-5 relative tolerance; their maximum absolute state
differences are 1.03e-4/1.11e-4, not absolute errors below 1e-4. Thirty-nine focused
tests passed with two SWIG warnings; four current Python lint/format checks pass.

Canonical records: `results/grain_math_response_v1/conditional_likelihood_paths_r1/`.
The conclusion is specific: independently resetting these units does not retain
the original diagnostic contrast. It does not identify a human trait mechanism,
or show that every natural sentence/chunk boundary needs the same correction.

## 5. What constitutes useful progress here

The scientific decision is whether selected-token action alone and reset slice
addition preserve the declared contrast. Accuracy improvement is not an endpoint.
Actual reversals require a context-carrying or separately calibrated observation
contract. Negligible effects would permit the simpler contract on this support.
Either outcome changes a measurement design, without a hyperparameter search.

This is a method-interface contribution and finite evidence, not a novel
mathematical discovery. The next substantive challenge is identifying which
part of carried evidence is target-specific and can be represented by a compact
numerical operator on new evidence. Repeating context/perplexity experiments or
fitting arbitrary high-dimensional maps would not answer that challenge.

## 6. Sources and scope of mathematical inspiration

[PICLe (Choi and Li, 2024), Section 3.1](https://arxiv.org/html/2405.02501v2)
already uses persona likelihood ratios and token-chain log probabilities to
select in-context examples, with a persona fine-tuned model. HM152 does not
invent the likelihood ratio, Bayes or the chain rule; it performs no fine-tuning
or example selection. Its local question is a paired vector/normalizer account
and compatibility of GRAIN reset observation units.

[Persona Vectors (Chen et al., 2025)](https://arxiv.org/html/2507.21509v3)
already extracts activation directions to monitor/control model character
traits. A prompt-induced model trait is not a human author's latent trait.

[OpenAI/math](https://github.com/openai/math), pinned adc7f124, the conditional
coordinate sweeps manuscript, section 08-prefix, separates an exact block
identity from distribution-dependent contraction. That is the methodological
inspiration: check composition and observation laws before asserting transport.
Texts here do not have its fair independent switches or hypercube symmetries;
its contraction theorem is NOT applied to these observations.

The captured law is frozen raw softmax. Historical articles were generated with
temperature .8/top-p .95. No automatic e-value, martingale, generative-sampler
calibration or human persona posterior follows from this different law.
