# HM243: align the companion observation before explaining target action

Frozen before new decoder observations, 2026-10-13 JST.

## Unknown and existing knowledge

HM207/208 already implement complete-vector bilinear source relations. HM242
adds an empirical binary reference and covariance accounts, not the first such
operator. Their old native contacts queried ONLY the edited window; the
numeric scorer also used its companion. This mismatched observation contract
can neither validate nor reject an actual two-window model mechanism.
HM110/156/224 already separate encoder interaction from reader nonlinearity.
Do not rediscover generic interaction, optimize accuracy, or invent new algebra.

## Fixed finite observation

Use the SAME24 historical public PANDORA Big5-only authors and SAME two
12-comment cue-masked windows/selected and control four-event sets. Do not
read application questionnaires. Reuse their cached1024D E5 vectors and both
old203/183 SOURCE operator banks without fitting or changing them. Other
historical panels, private MEPS, APIs and training are outside scope.

For author u and edited view v, A_S removes selected events and A_C removes
control events, each leaving eight. B_own is the opposite full12-event view.
B_stranger is that view of (u+1) modulo24, fixed BEFORE outcomes. This is an
audit recombination attributed in the frozen query to one hypothetical
anonymous writer, not a new real author or human causal intervention. Retain
actual event ownership, conditions and all stranger mismatches separately.
Place A first and B second, use the old direct-trait system/question wording
and excerpt enumeration, with BOTH v choices. No truncation or length rescue.
Preflight960 prompts: min1568, median1868, max2422 tokens, all fit3072.

Run24*2*2*2 batches with all5 questions/batch =960 endpoint inquiries, each
once. Fixed local checkpoint, MPS float32,1800 model seconds including loading.
No generation, no model fitting, retries, new paid services or budget renewal.
An unfinished/failed collection remains unfinished, not a successful subset.

## Competing mechanisms and exact observable

For any frozen reader F, define:

    I = F(A_S,B_own)-F(A_C,B_own)
        -F(A_S,B_stranger)+F(A_C,B_stranger).

An additive F(A,B)=f(A)+g(B) gives I=0 exactly. Nonzero I excludes that additive
account at these endpoints; it does not isolate personality from coherence,
position, topic or any other conditional mechanism. All5x5 SOURCE-target vs
queried-target contacts, all24authors/bothviews and bothSOURCEbanks are kept.
No best-target selection, sign flipping or source fitting to new observations.

Primary SOURCE family is the inherited HM207 joint/profile/pair uniform32-
profile reference. Preserve its two frozen banks. HM242 empirical total/
within/between/common covariance is a separately labelled existing-reference
comparison, NOT a hyperparameter candidate. Exact bilinear mixed action is

    I_math = (mean(A_S)-mean(A_C))' T (mean(B_own)-mean(B_stranger)),

using oriented windows correctly when v=1. Retain allsigned cross-target
contacts and zero/negative results. Compare native own/stranger deletion
effects with the old single-window deletion effects, but do not interpret
the old one-window answer as the joint answer.

## Representation versus readout, not another precision target

Store final normalized h and10candidate logits plus full-vocabulary mass.
Native F is logsumexp(digits5..9)-logsumexp(digits0..4); also retain the
candidate-conditioned score. With h00=C/stranger, a=h10-h00, b=h01-h00 and
c=h11-h10-h01+h00, the fixed actual head f gives

    I_reader=f(h00+a+b)-f(h00+a)-f(h00+b)+f(h00)
    I_representation=f(h00+a+b+c)-f(h00+a+b)
    I_head=I_reader+I_representation.

This anchored finite identity is standard and was used in HM156. The
counterfactual sum h00+a+b is only a mathematical head observation, not a
natural input or rerun of a patched decoder. Keep the signed cross term,
native-minus-head compatibility error and centered mixed candidate logits.
Do not treat squared energy shares as psychological fractions. Check causal
evidence-prefix equality across questions, with the old3e-4 tolerance; do not
optimize it. Existing source/implementation seals stay untouched.

## Decisions

If mixed native action contacts both old source banks, the companion field
has new empirical use on aligned evidence. If native interaction is present
but source contacts disagree, unconditional covariance is not the supported
bridge: STOP its rank/kernel/precision rescue. If native interaction is small,
retain the additive account on this protocol. Representation versus reader
accounts identify which construction layer needs the joint information.
No outcome can automatically name a new trait, prove psychological validity,
or close the original18/100/44 scope. Report actual magnitudes, not a passing
accuracy threshold. Bootstrap resamples authors while keeping both views;
the cyclic foreign-companion dependencies make these intervals descriptive,
not a population-independent significance test.

## Mathematical provenance

Finite anchored decompositions and bilinear contraction are mature. See
Griebel and Holtz2010 (doi:10.1016/j.jco.2010.06.001) and the ancestor HM156
accounts. Activation patching interpretation: arXiv:2404.15255. This experiment
does not patch activations or identify attention-head mediation.
OpenAI/math pinned adc7f124, three-state02-foundations lines1-130, explicitly
defines an experiment through its observed datum before posterior reduction.
That observation-contract discipline inspires this correction; natural text
does not satisfy its symmetric tree/spin hypotheses. No theorem transferred.
