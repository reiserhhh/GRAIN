# HM165: actual conditioned-writer choice after an observed article

Frozen before new model outcomes. Original18/100/44 remains OPEN.

## New capability and competing mechanisms

HM161/162 read fictional third-person records; HM164 links article vectors to
surface counts on the SAME article. Neither measures the original conditioned
writer's subsequent self-choice. HM119/125/126/127 use constructed reading laws,
not such continuation events. Build this missing interface, NOT another patching
precision, trait-classification or inverse-fitting experiment.

Two competitors: current system policy alone drives choice; versus past article
content carries a response pattern under a neutral policy. A third possibility
is policy/history interaction which prevents simple reversal of generation.
The nine actual corners distinguish these without stipulating a latent kernel.

## Fixed materials, events and resources

All named-persona articles from HM150: two targets x two poles x six tasks x
three styles x three historical generation replicates =216 articles,108 pairs.
Named prompts are the fixed first explicit-policy test, not a winning family.
All three seeds, six tasks, refusals and truncations retained. Seeds are not
authors; assigned persona policies are not known human psychological truth.

For each actual high/low pair retain the historical user task and style. Cross
policy p in {-1,0,+1} with article history h in {-1,0,+1}; zero history is an
empty assistant message, zero policy is the original neutral system at that
style. Four new choice opportunities: optional/required social participation
and known/unknown appointment information. The actions themselves stay fixed
within each opportunity pair. Both A/B orders retained, not assumed invariant.
108 x9 x4 x2 =7776 independently registered input histories; equivalent empty
controls recur by design, not independent scientific authors/observations.

Each history is re-rendered as system/old user/old assistant/new user, not an
uninterrupted reuse of the old generation cache or a natural temporal process.
Crossed histories are counterfactual controls, not new coherent author records.
The new request asks for self-choice, never a trait diagnosis or third-person
judgment. Original style instructions may conflict with the choice format;
measure unrestricted A/B token mass instead of hiding that limitation.

Offline pinned Qwen2.5-1.5B, MPSfloat32, no gradient/training/API/network/private
data. Fixed32-row batches; full original bodies, maximum640 prompt tokens;
480 cumulative model seconds, no renewals or failed-call retry. If cap reached,
preserve partial rows, no automatic restart. No head/layer/rank/dose/prompt grid.

## Response observation, not a diagnostic answer

Compute native A/B logits and full-vocabulary normalizer. The defined binary
response protocol restricts the next token to A/B:

    q(p,h,a,o)=exp(logit_positive)/(exp(logit_A)+exp(logit_B)).

Keep full-vocabulary A/B mass and positive log odds separately. Draw exactly ONE
protocol response per request using a sealed seed+ID sampler; this is a constrained
simulated choice, not unconstrained generation, human behavior or independent
ground-truth personality. Model-law q is primary; draw realizations are descriptive.
Both label orders remain separate in all effect accounts.

At neutral/empty reference, preserve every signed term:

    P_p=q(p,0)-q(0,0); T_h=q(0,h)-q(0,0),
    I_ph=q(p,h)-q(p,0)-q(0,h)+q(0,0),
    q(p,h)=q(0,0)+P_p+T_h+I_ph.

Natural-policy/history contrast uses actual matched corners:

    q(+,+)-q(-,-) = Delta P + Delta T + (I_++-I_--).

Report opportunity shifts, history under high/low policy, full interaction and
all tasks/styles. Positive means the named action, not a universally positive
trait realization. No term is a psychological fraction or removable noise.

## Numeric author-evidence bridge

Use SAME216 article IDs and their cached GRAIN4 slices. Frozen HM147 origin and
scales produce the625 tensor features; retain its old full/single/relinked E/N
readings without refitting or selecting a winner.

New source-only response witnesses use actual neutral-policy article choices:
source diary/email/opinion108 articles, development advice36, application
review/neighbor72. Average two registered orders as an explicitly defined
balanced observation, but retain order-specific q and all effects. No persona
labels, native diagnostic answers or application choices enter the constructor:

    u_a=Cov_source(q(0,h,a), Phi(article)),
    f_a(x)=<u_a,Phi(x)-mu_source>/||u_a||.

This is a mature cross-moment witness, NOT calibrated q, a latent psychological
posterior or a new kernel theorem. The new science is its link to actual future
self-choice. Seal the source program BEFORE opening application choice outcomes
in the analyzer; execute a separate cold vector-only CLI. Test source/app roles
and all four opportunities; no r/AUC promotion, leaderboard or model selection.
Compare signed paired readings with actual history/natural response effects,
and retain their discrepancies rather than tuning to them.

## Reading and originality

- Probing Persona-Dependent Preferences (arXiv2605.13339v1), introduction,
  methodology and appendixA.2 conflict/biography passages: revealed choices and
  persona-dependent valuation already have vector/probe prior art. We do not
  claim to discover the shared preference vector or reproduce their steering.
- Personality Illusion (arXiv2509.03730v1), abstract/introduction: surface traits
  and behavioral tasks must be separated; not a proof that our text bridge fails.
- How Far Do Persona Effects Generalize (arXiv2609.32758v1), introduction:
  source reuse versus target recalibration; no target refit in this packet.
- PICLe (arXiv2405.02501v1) already covers likelihood-ratio persona elicitation;
  do not label reverse conditioning automatically Bayes or recover a prompt word.
- OpenAI/math pinnedadc7f124 Thorp section05 inverse-color compatibility and
  balanced trace moments reread: inspiration for separating observation laws.
  No permutation-independence/density/spectral theorem is applied to text.

The factorial/cross-moment math is standard. Candidate contribution: a source
observed, condition-indexed bridge distinguishing author-policy, text-history
and reader judgment, plus its empirical scope. No field-wide first claim.

## Acceptance and stopping

Authenticate original generation/observation seals and weights before capture.
Tests distinguish pure-policy, pure-history and constant worlds; verify semantic
answer order, nine-corner reconstruction, frozen-source cold inputs and unavailable
constant anchors. Independently rebuild logits/effects and pairwise covariance;
same-agent audit, not blind replication. Record real runtime/PID/batch outputs.
Preserve failures, caps and negative signs. Once the finite mechanisms are
distinguished, stop: no follow-up to raise correspondence percentages.
