# HM152: conditional predictive paths and slice observation units

2026-10-11. Prospective after HM151,not executed. Original18/100/44 OPEN.
Goal is a mechanism/measurement connection,NOT persona classification accuracy.
No API/private text/new generation/training/Git/Claude/lockbox/oldqueue.

## Unknown, competing explanations and decisive observation

HM150 observes conditional expression;HM151 shows unrestricted vector transport
is vacuous on the high-dimensional finite support. The next identified object
is a DECLARED conditional predictive law of the frozen generator,not an arbitrary
matrix fitted between representations and not an LLM's spoken trait judgement.

Unknown:how does actual token-path evidence for a supplied persona depend on
the inherited prefix and competition among candidate expressions? In particular,
can independently rescored slices preserve that evidence without carrying the
prefix state? This directly concerns GRAIN observation-unit assembly.

Competing operational accounts:

1. Observed-token uplift:raw chosen-token logit contrast alone supplies the
   direction of evidence;the vocabulary normalizer does not change it materially.
2. Conditional predictive evidence:the chosen-token uplift and candidate
   competition jointly determine the contrast and can oppose/cancel.
3. Independently reset slices:sum of slice scores gives the same diagnostic
   contrast as the continuous prefix-conditioned path.
4. Prefix-sensitive realization:slice reset changes the contrast or sign even
   with exactly the same article tokens and persona/condition hypotheses.

These are finite descriptions to distinguish,not assertions that human traits
are binary or that one model supplies true psychological likelihoods. No new
persona polarity,rank/kernel/head/temperature/prompt fitting or AUC threshold.

## Exact account and its measurement meaning

For two fixed persona prompts H,L with SAME task/style condition c,and the SAME
article prefix x_<t,let h_a(t) be the final normalized state, W the output
embedding,and b any output bias. Define

    z_a(v,t)=W[v].h_a(t)+b[v]; logZ_a(t)=log sum_v exp(z_a(v,t)).
    ell_t=z_H(x_t,t)-z_L(x_t,t)-[logZ_H(t)-logZ_L(t)].
    L(x|c)=sum_t ell_t=log p_H(x|c)-log p_L(x|c).

The actual-token numerator is a vector action;the normalizer retains ALL
alternative next tokens,not just a personality keyword. A constant logit shift
cancels in ell_t although the two displayed accounts change. Raw-logit effects
are gauge-dependent accounts,not psychological information fractions. Keep a
token-uniform gauge companion and exact normalized contrast;do not claim the
split is uniquely causal. Rotating states and covectors together preserves it.

For the second half of a fixed article prefix:

    L_carry(x_2|x_1,c) versus L_reset(x_2|c).
    L_full=L(x_1|c)+L_carry(x_2|x_1,c) exactly.
    L_reset_sum=L(x_1|c)+L_reset(x_2|c).

The first is probability-chain composition;the second is a different observation
model. Reset may be a mid-sentence/token-boundary off-distribution probe,not a
natural psychological intervention. Keep the same native token IDs;never
decode/retokenize after splitting. Do not equate higher raw likelihood with
better personality measurement or count tokens as independent authors.

## Original support, no outcome-driven narrowing

Authenticate the completed HM150 r2 generator/material/model seals and all702
article token sequences. Use every648 persona article with its matching target
and wording family;use each54 neutral article under all2targets x3families as
nonpersona controls. Total972 hypothesis-conditioned article cases. All6tasks,
3styles,3seeds,2poles/neutral controls retained. Seeds are NOTauthors.

Keep the first32 continuation tokens,split at16. For a shorter output,retain
all tokens and report whether a second-slice comparison exists;do not replace
it or regenerate. Include stored EOS if within the prefix. Record token count,
refusal/truncation flags and the inherited task-role split. This is a new
prefix/reset computational intervention on public synthetic material,not an
untouched article test set or another generation experiment.

H/L use the parent's existing matched persona prompts,including the article's
style and task. Never prepend a trait question or ask for a score. This is
inverting KNOWN synthetic hypotheses conditionally,not inferring the unknown
writer's real psychological traits. Condition choice is held for this packet,
not declared noise or deleted from the full GRAIN construction.

## Collection, artifacts and finite budget

Local pinned Qwen2.5-1.5B only,FP32/eval/no gradients/MPS. Offline exact model
hash verification. At most4logical sequences percase:carryH/L and resetH/L.
Use a bounded batch of4 and project ONLY predicted article positions through
lm_head in small chunks,avoiding full prompt x vocabulary logits. Physical
positions use each native compact prompt;position/length effects remain part
of this conditional law. Do not force a new common position scheme.

Store final prediction states,selected output covectors/chosen logits,full-
vocabulary log normalizers,token IDs and pertoken log probabilities locally
under a NEW conditional_likelihood_paths_r1. Freeze code,plan,inputs and
prompt addresses before collection. Immutable complete batches,resumable
status;cap1200model seconds INCLUDING load and any failed started computation,no
automatic renewal/retry. Max3888logical sequences plus2 first-slice causal
checks (one pertarget),not3890 independentcases.
No partial case promoted;scientific coverage explicitly reports missingcases.

Outputs:case_manifest.json,raw captures,token_accounts.csv,slice_accounts.csv,
condition_summary.csv,coverage.json,source/input/output hashes,summary.json,
read-only alternate arithmetic audit and Chinese theory report. No leaderboard,
classifier winner,significance-only selection or native-answer imitation.

## Checks and scientific decision

Before full capture,synthetic numeric tests cover normalized/gauged equivalence,
dual coordinate invariance,chain/slice composition and a designed reversal
where numerator uplift is canceled by the normalizer. Those tests verify code,
not discovery. Actual model checks:causal prefix equality on the firstslice,
independent W.h selected-logit replay,full log-softmax account and EOS bounds.
Declared FP32 operational tolerance1e-4 pertoken absolute/1e-5 relative;failed
model checks retained without changing gates or rerunning to improve precision.

Report ALL trait/family/task/style/seed/pole cells,including negative and neutral
effects. Numerator-versus-normalized sign differences and reset/carry differences
are descriptive effects on this support,not psychological error percentages.
If normalization changes evidence direction,the vector-to-target law must be
centered against available alternatives. If reset changes target contrast,the
measurement compiler must carry context or explicitly calibrate reset units.
If both effects are negligible,retain the simpler account on this support and
move on;do not create another accuracy optimization to force a discovery.

## Prior art, differences and OpenAI/math inspiration

PICLe2024 §3.1 already derives persona likelihood ratios and the token chain:
https://arxiv.org/html/2405.02501v2 . It uses persona SFT/ICL example selection.
This packet does NOT invent the ratio,Bayes or text perplexity;no SFT or top-K
selection. The added question is the concrete vector/alternative-competition
account and compatibility of independently scored GRAIN slice units.

Project inventory:original persona_trace continuation_logprob averages token
probabilities for patch localization;VM06 studies author/context compatibility;
HM88 handles attention event/within-event normalizers;HM102 handles merging
local projective lifts;HM108 decomposes ten-digit answer readout curvature.
Those do not provide this paired whole-vocabulary persona path/reset account.
Keep their mechanisms and failures;do not relabel known context sensitivity as
new personality discovery.

OpenAI/math adc7f124 conditional-coordinate-sweeps08-prefix1--125 separates
an exact block identity from law-dependent contraction. This inspires checking
slice composition BEFORE making a population claim. Fair independent switches,
hypercube prefix symmetries and the contraction rate are not assumptions on
texts. Its advanced theorem is not applied to GRAIN by analogy.

Historical articles were generated at temperature.8/top-p.95;raw-softmax
teacher-forced likelihood is a DIFFERENT declared model. Do not call the ratio
an e-value/martingale valid under that original sampling distribution,a true
persona posterior,or a human measurement. Event qualification and independent
psychological/behavioral calibration remain separate mainline connections.
