# HM256: forward expression law versus reverse personality observation

2026-10-13. Freeze before any new inverse response. Prior HM150/152 and
HM255 outcomes are known; this is an exploratory NEW-endpoint experiment.

## Missing ability and non-duplication

HM152 measured text-prefix likelihoods under two supplied writer personas.
HM255 measured concrete behavioral item observations, not the inverse of those
likelihoods. HM125/148/154/221/223/232 already cover dual frames, contextual
response, lexical fields and criterion transportation. No new adjoint, metric,
reader fit, head/rank/prompt grid or generative accuracy improvement is sought.

The missing interface is a probability-compatible bridge between the SAME
forward observation law and NEW reverse judgments. It can tell whether an
arbitrary persona prior could reconcile the two directions, rather than
declaring that vector generation and trait inference must be inverses.

Competing mechanisms: (M1) instruction attribution agrees with the writer
likelihood up to a condition-specific prior; (M2) disposition judgment does so;
(M3) either reader has text-dependent departures that no prior can remove.
Both readers can have usable effects without being the same observation law.
No outcome is assumed to be a human personality posterior.

## Mathematical contract

For each fixed target/family/task/style c, X is a unique exact32-token prefix
from the already collected high, low and neutral articles at all three seeds.
Duplicate prefixes are ONE outcome, not extra observations. Keep their case
aliases. Every old case must map to an outcome; no refusal/truncation removal.

The old raw-softmax observations give a(x)=log p_H(x|c) and b(x)=log p_L(x|c).
Let L(x)=a(x)-b(x). On the declared finite support S_c, export the forward
conditional laws A_H(x)=exp(a(x))/sum_S exp(a), and A_L likewise. A cold numeric
scorer exports BOTH expit(L) (equal original priors) and
expit(L-logsumexp(a)+logsumexp(b)) (equal priors AFTER finite restriction).
Restriction changes the prior; these two scorers are not silently identified.
This is a supplied model-persona experiment, not a natural-author likelihood.

Reverse first-digit probabilities supply r_q(H|x), with native high/low log
odds R_q(x). Familiar source instructions appear as two quoted alternatives;
all common task/style instructions remain. Cross two explicit contracts:
instruction attribution versus writer-disposition attribution, and BOTH option
orders. There is no demand that either have numerically computed probabilities
or chain-of-thought, and no forward numbers are given to the reader.

Positive forward/reverse conditionals have a common joint law iff

    R_q(x) - [log A_H(x)-log A_L(x)] = gamma_c for every x in S_c.

Equivalently ALL four-cycle defects must vanish:

    C_q(x,y) = R_q(x)-R_q(y)-L(x)+L(y) = 0.

This test cancels unknown prior odds, finite-support normalizers and offsets.
It does not fit a classifier. The best possible uniform error of ANY constant
prior adjustment is exactly (max_x(R-L)-min_x(R-L))/2. Save every signed pair,
not just its RMS or favorable order. No offset is used to promote predictions.

For decoded orders R0/R1 keep their mean R and half-difference O. Save C0,C1,
Cmean and their signed order account; average is descriptive, not a repaired
reader. Also retain complete candidate mass/top-token and native levels.

## Scope and computation

All972 old cases/702 historical articles/108 conditions, at most972 unique
context-prefix outcomes and3888 NEW reverse inquiries. No new articles/people,
questionnaires, fitted model or neural training. Seeds are not authors; cases,
queries and pairs are not independent samples. No significance/accuracy claims.

Use the same pinned local Qwen model offline. Shared causal-prefix reuse is
computational caching only. One1200-model-second nonrenewable cap. Direct versus
cached logits on the first complete four-query family must differ <1e-3; this
is an instrument gate, not a precision search. No failed collection restart,
model switch, alternative wording, outcome-selected support or cap extension.

Stages: prepare (authenticate old seals, build exact-token prompts, consolidate
aliases, compile forward bundle) -> independent cold numeric scoring -> collect
-> analyze -> alternate numeric verify. All outputs are new immutable files.
Original18/100/44 stays OPEN. No API/private/Git/Claude/automation/lockbox/oldqueue.

## Prior art and use of higher mathematics

Arnold and Press1989 already study compatibility of conditional distributions:
https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478750 (abstract read).
Morton2008 sections1-3/6 gives cycle-binomial/toric relations:
https://arxiv.org/html/0808.1149v1 . Our strictly positive two-persona support
uses its elementary four-cycle specialization, not a new algebraic theorem.
Betz/Richardson2023 already test neural probabilistic coherence:
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0281372
(abstract/introduction/formal framework read, not their trained-ranker setup).
PICLe2024 already uses persona likelihood ratios (HM152's prior art).

OpenAI/math pinned adc7f124 three-state foundations98-114/166-210 was reread:
an observation law and its posterior must belong to the same statistical
experiment; dependent readers cannot be multiplied as independent branches.
No spin-tree theorem is transplanted. The NEW project ability is a native
writer/reader probability-law bridge with an outcome-free cold inverse, not
proof that LLMs are globally incoherent or a novel human-personality formula.

## Decisions after the complete finite observation

If either reader shares a prior-compatible law, develop its new-evidence
source support; don't optimize the offset. If it has text-dependent defects,
close the automatic writer-inverse interpretation for this contract. Preserve
the cold operational observer and ask what extra behavioral valuation accounts
for the departure. Do not improve accuracy/head/metric until an identified
mechanism distinction is actually obstructed by measurement error.
