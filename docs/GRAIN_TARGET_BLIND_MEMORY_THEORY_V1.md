# Target-blind memory and conditional text observation

HM258, 2026-10-13. A construction within the GRAIN/vector-evidence mainline.
This document is explanatory; the scientific plan and code are separately sealed.

## 1. The distinction that matters

An observation of a target-specific effect does not imply a target-specific
storage axis. The same stored evidence can be evaluated by different functions.
Conversely, a target-specific activation does not prove that its source evidence
has that meaning for the author. These are distinct interfaces, not claims that
all targets share one trait or that all context is noise.

HM257 put a behavioral hypothesis before a real prior comment. A target could
affect both the comment's encoding and its later continuation. HM258 places ALL
eleven hypotheses after a common prior-comment prefix and actually reuses the
same native K/V. It asks whether BEFORE-context target conditioning is necessary
for a SAME-event SOURCE-linked observation. It does not estimate what percentage
of the old early-conditioned response used a particular internal path.

## 2. A declared observation operator rather than an assumed personality axis

Let a be a complete actual context comment, b a complete actual target comment,
and f an addressed matched-stranger context. The frozen autoregressive observer
has a target-independent prefix compiler C_theta(a)=KV_theta(P0,a). Let s_q^+
and s_q^- be AFTER-context high/low behavioral descriptions. For actual target
tokens b_t define

    ell_q^p(a,b) = sum_t log p_theta(b_t | C_theta(a), s_q^p, b_<t)
    L_q(a,b) = ell_q^+(a,b)-ell_q^-(a,b)
    T_q(a,f,b) = L_q(a,b)-L_q(f,b).

Equivalently T is a log cross-ratio of four strictly positive sequence
probabilities. This is standard conditional-probability algebra, not new math.
The observation includes context, target description, actual continuation and
observer version. It is not a likelihood under an identified human-trait law.

For a strictly criterion-blind additive context effect,
ell_q^p(a,b)=U(a,b)+V(q,p,b), T=0. Nonzero T rejects that strict explanation.
It does not distinguish every grammar, register, opportunity or semantic
interaction. The matched stranger is coarsely venue/length/E5 matched, not a
randomized human intervention or a complete topic control.

## 3. What the common memory guarantee does and does not establish

Causal autoregressive attention gives earlier-prefix states no incoming edge
from a future suffix. This is an architectural property, not an empirical
discovery or a general theorem about human memory. The concrete implementation
checks something additional: all eleven suffix calls receive copies of exactly
the same observed K/V, the original cache remains unchanged, and cached reads
agree with concatenated native reads under the SAME serialized observation.

After the suffix starts, its own states and subsequent target-token states are
criterion-conditioned. Thus a positive result provides a common-source-memory
construction with conditional reading, not a fully criterion-independent model.
The source encoder, suffix processing and next-token observer remain neural.
Only the exported numeric action accounts are torch-free at the cold stage.

The early and late protocols have different instruction positions, prefix
lengths, wording order and target-token positions. Their difference is not a
path-specific mediation estimate. Do not divide late/early amplitudes to report
a psychological information fraction or an upstream/downstream percentage.

## 4. Vector accounts, with vocabulary competition retained

Let h_t be the final normalized state predicting b_t, W_bt its actual token
output covector, beta_bt its fixed head bias, and A_t the full-vocabulary log
partition. At a common target token,

    log p_theta(b_t|...) = W_bt^T h_t + beta_bt - A_t
    T_q = sum_t W_bt^T Delta_context Delta_pole h_t
          - sum_t Delta_context Delta_pole A_t.

The bias cancels. W_bt changes with t. There is no assumed single global
personality-word vector. Numerator and partition can oppose each other; neither
is removed as noise. All positive and negative interactions are retained.

For a fixed observation and differentiable cache coordinate K, the local
measuring covector would be D_K L_q(K,b). That is a standard derivative and a
possible future interface, NOT a gradient measured or localized in HM258.
The present run supplies actual finite observations before selecting any local
linearization. No head/layer/gradient/accuracy sweep is hidden in the plan.

## 5. Connecting to an author-relative SOURCE meaning

The inherited SOURCE386 law defines W_p=psi_p-pi_p. Its complete score is kept
as choice mu_p(c)=E_R[W_p|c] plus within-condition W_p-mu_p(c). The same two
actual own comments give a fixed author field S_up. No current author labels or
model answers fit this field. Both directional uses produce T_uq and remain
paired at the 24-author level.

    K_pq = (1/U) sum_u (S_up-mean S_p)(T_uq-mean T_q)
    K_full = K_choice + K_within.

All 25 contacts, per-token and sum units, both directions, and all GRAIN15
coordinates are retained. K is a cross-contact, not Pearson r, prediction
accuracy, a psychological variance share, or a latent-population causal effect.
Historical representation discovery overlaps some panel authors; the source
field calibration excludes all 24. This is not a new untouched validation set.

If a common source cache supports SOURCE-linked late observations, the reusable
object is (C, observation family, event addresses, reference population), not a
label pasted onto a storage dimension. This can guide text-based assessment
construction without forcing all targets into one fixed geometric axis.

## 6. Relationship to mature mathematics and prior mechanisms

- PICLe (2024), Sec3.1 Eq3-5: persona likelihood-ratio reasoning already exists.
  https://arxiv.org/html/2405.02501v2
- Function Vectors (2024), Sec2-3: task-conditioned internal vectors and their
  reuse are prior art. This run is not the discovery of function vectors.
  https://arxiv.org/html/2310.15213v2
- Heimersheim/Nanda (2024), Sec2.2: mediation and amplification require different
  interventions. A moved instruction is not a pure path-patching experiment.
  https://arxiv.org/html/2404.15255v1
- OpenAI/math adc7f124, three-state foundations lines98-116 and152-232: declare
  the observation experiment and target-independent transformation before using
  posterior or degradation arguments. Here C is target-blind by design, but
  unknown human latent laws, independence and posterior sufficiency do not follow.
  No reconstruction-threshold, Jensen-information or tree theorem is transferred.
  https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex

The possible project contribution is an event-qualified construction linking
GRAIN SOURCE meaning to a shared evidence memory and conditional real-text
observations. It is not an assertion of global methodological priority or a
new high-mathematics theorem. The definitions above preceded the single sealed
run. The following sections record its outcomes without changing those definitions.

## 7. Actual shared-memory construction and signed results

All 96 target-blind prefix encodes and 288 suffix batches completed once:
1056 conditional sequences plus four native compatibility spots,293.269096
model seconds under the original1200-second cap. No new articles, application
questionnaire values, answer fitting, or scientific retries were used.
The original prefix K/V remained unchanged in every reuse comparison.

The SAME24 historical authors supply48 directional uses, not48 people. All
48 own full comments and92 unique comments including strangers remain.
Target lengths are44--148 native tokens; maximum new input length319.
SOURCE calibration excludes the24; historical representation discovery overlaps
17. Exact subreddit matches are0, so the foreign comparison is not full
topic/opportunity matching or an isolated personality intervention.

| SOURCE target / late criterion | Full contact | Choice | Within condition |
| --- | ---: | ---: | ---: |
| E | -.009090389 | +.001286338 | -.010376727 |
| N | -.002151087 | +.001032479 | -.003183566 |
| A | -.002657126 | -.004994512 | +.002337386 |
| C | +.008752884 | +.001016778 | +.007736106 |
| O | -.012465135 | -.010002442 | -.002462693 |

Units are SOURCE-field times nats, not r/AUC, probabilities or psychological
fractions. All25 cross-target contacts and all15 coordinate contacts remain.
C full contact is positive in both directions (+.010093462,+.007412306),
as are its choice and within parts. N/O are negative in both directions.
E/A have opposite direction signs; their negative pooled values do not imply
every author or every directional use reverses. Per-token pooled diagonal signs
are the same as the sum-unit signs, not a second independent confirmation.

Compared descriptively with HM257, pooled E/N/O change from positive to
negative and C from weakly negative to positive. Late interaction RMS is
1.406--3.064 nats, versus early .528--1.246. Stronger action has NOT preserved
the former SOURCE correspondences. Early/late serialization differs; this is
not a quantified mediation effect or evidence that human personality reversed.

The positive construction result is that a single cache formed before specific
target descriptions supports conditional real-text observations, including a
C-linked SOURCE contact. It does not require a separate target-conditioned
prefix encoder for that declared interface. Common storage does not force
common interpretation, nor does it eliminate later neural target processing.

## 8. Existing mechanisms that must not be rediscovered

| Prior interface | What it already supplies | What HM258 adds / does not add |
| --- | --- | --- |
| HM33 query/context carrier | Identical question tokens can absorb and act on author evidence | Not discovery of contextual query states |
| HM125 contextual dual frame | Query-free numeric coordinates and source-frame transport diagnosis | Not a new global author ruler |
| HM148 location/change crossing | A frozen functional reads changed vectors differently at changed backgrounds | Early/late is not this controlled four-corner experiment |
| HM154 target/wording fields | Relative lexical target organization with retained wording interaction | No new vocabulary geometry theorem |
| HM162 component/support transport | Joint source components and descendants can affect endpoint meaning | Common-cache reuse is not proof of natural causal transport |
| HM223 criterion capsule | Source evidence can act in a new excerpt-free digit reader | Full actual-comment likelihood is a different observation contract |
| HM232 source/reader crossing | Criterion discrimination and source-meaning retention are separate | HM258 connects a shared prefix to SAME-event SOURCE fields, not a faithful universal observer |

These companion documents were read during closeout. Generic vector reuse,
ordinary context dependence and the existence of a query carrier are therefore
not open novelty claims. Neither does this packet replace GRAIN's choice,
state, conditional response, path, or reference-population channels.

## 9. Next mechanism, not a larger effect or better numerical fit

The reusable object is now (qualified event, shared cache, observation operator,
SOURCE reference), rather than a fixed stored personality direction. The next
question is which use of the SAME evidence establishes a target correspondence:
context-conditioned criterion states, direct criterion-conditioned continuation
reading, or their interaction. A changed prompt/serialization cannot distinguish
those paths by itself.

Before a new finite plan, inventory existing graph cuts on identical suffix and
cache addresses, and require a new full-text observation capability. Any proposed
test must keep serialization, event identity, source fields and target token
readouts fixed; vary a declared computation edge rather than select descriptions
that increase contact. Preserve all five targets and all signed terms.
No gradient/head/layer/prompt/dose/likelihood/rank or precision grid follows.
No new path intervention has been executed by this closeout.

The re-read causal-abstraction Definition27/Section2.3 requires a declared
intervention correspondence and commuting solutions, not just score agreement:
https://arxiv.org/html/2301.04709v4 . No such full abstraction has been established
between SOURCE psychological labels and this neural observer. It supplies a
construction requirement for the next mechanism, not a license to fit one
readout until it resembles the other. OpenAI posterior sufficiency likewise
belongs to its specified experiment; it is not universal target-meaning transport.

Native cache/direct discrepancy is2.6703e-5; selected-head discrepancy4.2163e-6.
The main verifier re-sums240 four-corner readings and75 late pooled contacts;
a separate read-only scalar-loop audit covers ALL900 source contacts and150
GRAIN15 contacts, maximum1.3878e-17. All choice/within identities differ by at
most2.4286e-17. These are implementation checks, not the scientific achievement.
19 focused tests and four Python lint/format checks pass; old SWIG warnings
remain. Same-agent alternative arithmetic is not a blind independent replication.
All scientific seals are unchanged; original18/100/44 remain OPEN,Goalactive/nojob.
