# GRAIN joint target/person observation: finite HM198 result

2026-10-12 JST. Written AFTER complete collection and sealed analysis.
This is a rejected construction candidate, not a personality law or a new
theorem. Original18/100/44 remain OPEN.

## 1. Distinct objects

An evidence record H binds two behavioral attributes to persons A and B.
The four assignments x=(e,n) have the same complete whitespace word bag in
each wording/scenario bank. There are two banks and two reverse presentation
orders. e,n are designed ownership signs, not human trait parameters.

For query operator phi and requested persons a,b, define

    F_phi,ab(H) = P_model(Y | H, q_phi,ab, answer in {N,Y}).

This is a conditional answer-token reading. It is NOT automatically a
probability law over the behavioral attributes or the author's personality.
Source-supported target meaning and neural answer confidence are different
interfaces even when both use numbers between zero and one.

The declared competing composer is

    F_AND,ab^product = p_Ea * p_Nb,
    F_OR,ab^product  = p_Ea + p_Nb - p_Ea*p_Nb.

The test does not assume this independence model is true. Nor does disagreement
identify dependent human traits: probabilities from separate questions need
not be coherent marginals of any common law.

## 2. Person-qualified contrasts

For either query operator, preserve

    C_phi = (F_phi,AA - F_phi,AB - F_phi,BA + F_phi,BB)/4,
    J = (C_AND-C_OR)/2,
    U = (C_AND+C_OR)/2,
    J_product = (p_EA-p_EB)*(p_NA-p_NB)/4,
    remainder = J-J_product.

An additive person/target score has zero C_phi; a product can have nonzero
C_phi, so finding a joint contrast alone does not reject the product rival.
In the exact Boolean design, C_AND=en/4 and C_OR=-en/4, hence J=en/4 and U=0.
For actual model readings keep ALL coefficients of 1,e,n,en, not just the
one associated with the design's intended meaning.

For the two strictly positive designed laws P+/-=(1+/-rho*en)/4,

    E_P+[F]-E_P-[F] = 2*rho*F_EN.

Their separate e/n marginals agree; rho=.4 is fixed. These are designed record
mixtures, not sampled authors or independently measured personality groups.
This finite Fourier/Boolean identity is ordinary algebra, not new mathematics.

## 3. Exact vector account without an upstream linearity claim

At the final normalized query state,

    logodds_phi,ab = (W_Y-W_N)^T h(H,q_phi,ab) + (b_Y-b_N).

Therefore the actor contrast of log odds is the SAME output covector acting
on (h_AA-h_AB-h_BA+h_BB)/4. However, h depends on the requested operator and
persons. This terminal linear identity does not imply that the state is an
additive entity/attribute representation or that upstream reasoning is linear.

Block16 body endpoints were identical across all14 suffix queries in each
record (maximum difference0). The causal prefix was identical. Those are
expected compatibility properties, not evidence that body information is
unused or that a specific binding circuit has been localized.

## 4. Complete observations and rejected promotion

16 actual batches/224 enquiries, frozen cached Qwen, float32/eager MPS;
43.607877 cumulative model seconds including load. No new sampling, answer
generation, fit, API, private text, human labels or training.

| Bank/order | AND en | OR en | J en | Independent product en |
| --- | ---: | ---: | ---: | ---: |
| 0/0 | -.035074 | -.049900 | +.007413 | +.120972 |
| 0/1 | -.015599 | -.000795 | -.007402 | +.033299 |
| 1/0 | -.001437 | -.000071 | -.000683 | +.000392 |
| 1/1 | -.000397 | -.000045 | -.000176 | +.014007 |

The registered joint-interface promotion required AND/OR opposition in both
banks/orders. AND has0/4 expected signs; OR has4/4; J only1/4. The fixed
product has4/4 positive signs but does NOT reproduce J. Remainder RMS=.090627
versus native J RMS=.028222. This is an observation mismatch, not psychometric
error, accuracy, significance or an information fraction.

Three AND frames have average Y readings .980522/.996998/.998466 despite
the design supporting only one of four actor-pair conjunctions per record.
The bank0/order0 mean is .762603. Thus readings heavily favor Y. This descriptive
comparison uses the registered design, not a calibrated probabilistic truth.
AND log-odds en actor contrasts remain negative in all4 frames; failure is
not solely hidden by sigmoid saturation. Individual questions also have
en coefficients (RMS=.135869), and unchanged badge readings have en
coefficients (RMS=.016474). They cannot be treated as isolated clean marginals.

Decision: retain observations but DO NOT promote this compound prompt as a
joint-personality oracle, and DO NOT tune the product to its answers. Do not
generalize failure to all models, all bindings or all personality information.
No prompt/model-size/dose/weight grid is started to rescue the result.

## 5. Actual theory consequence and prior work

GRAIN must separate (i) empirical target/evidence observation law, (ii) the
typed target operation and person qualification, and (iii) neural answer
rendering. Source target interactions cannot be supplied simply by multiplying
independent readings or by accepting compound text-query confidence as truth.
This is an interface restriction, not a claim that joint targets are impossible.

Feng/Steinhardt, https://arxiv.org/html/2310.17191v2 , Sections3/4.1, already
tests entity/attribute substitution and additive binding functions. That does
not imply multiplicative composition of separate answer probabilities.

Ghosh et al., https://arxiv.org/html/2412.16100v2 , Introduction and Section3
through Definition1: fact-checking correctness and logical consistency are
different. Complex-query inconsistency is prior art. Their KG-based binary
fact checking is not our continuous actor-contrast measurement; their theorem
or fine-tuning guarantee is not imported. The nonexistent v3 HTML was attempted
before reading the correct v2. No claim to discover LLM logical inconsistency.

Jiang et al., https://arxiv.org/abs/2406.11050 , abstract ONLY, motivates
controlled token-bias tests. It does not establish token bias as the specific
cause of the current result. No additional probability-judgment/quantum paper
from search results was read as proof or used in the mechanism.

OpenAI/math adc7f124, Universal strong Kadison-Kastler stability,
Introduction lines77-119, distinguishes norm proximity from a multiplicative
correspondence and correction on different representation spaces. It inspired
a composition test, not a transferred von Neumann-algebra theorem. GRAIN
does not satisfy established hypotheses of that theorem.

Same-agent alternate arithmetic passed: output-head maximum6.431952e-5,
score3.552714e-15, direct mixture/actor accounts1.776357e-15. These check
implementation, not the psychological meaning. No scientific retries or edits
to sealed code/results. The report and this theory note are post-outcome.
