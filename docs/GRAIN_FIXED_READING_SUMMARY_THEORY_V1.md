# Fixed reading laws and target-conditioned summary forcing

HM202 pre-outcome theory, 2026-10-12 JST. Empirical outcomes will be appended
separately after the fixed collection; scientific code and inputs remain sealed.

## 1. Object and coordinates

Let x be one registered natural-text evidence unit, q one of five target
questions, l a decoder layer, and r an offset in a maximal identical pre-answer
suffix. The actual prefix supplies V_lqr(x). The suffix has the same literal
tokens for all q but can encode different context. This is an instrument's
conditional summary, not a new human psychological factor.

All query-relative suffix offsets are registered before observations. Their
absolute positions can differ: this is why K and its rotary position are not
averaged. V has no explicit RoPE operation, but its contextual representation
can still depend on earlier positions and wording. These conditions remain.

Define the five-question arithmetic projection

    Vbar_lr(x) = (1/5) sum_q V_lqr(x),
    E_lqr(x) = V_lqr(x) - Vbar_lr(x),
    sum_q E_lqr(x) = 0.

In common Euclidean coordinates,

    sum_q ||V_lqr||^2 = 5||Vbar_lr||^2 + sum_q ||E_lqr||^2.

The identity is classical. It does not turn Vbar into a sufficient statistic,
E into noise, or their squared norms into psychological information fractions.
The five targets are a declared finite question family, not all possible traits.

## 2. A controlled reader, not a higher-fidelity answer fit

The native run supplies probabilities p_lqi(x) on the entire final-row support.
In replay, this law is fixed at each l and q. All earlier rows retain native
outputs. The final self value V_lqf(h_l) can depend on the changed current state;
the ordinary residual, RMS normalization and MLP continue to operate.

For a suffix donor W_lqr, the direct installation force before W_O is

    d_lq(W;x) = sum_(r in suffix) p_lqr(x) [W_lqr(x)-V_lqr(x)].

In particular,

    d_lq(common;x) = -sum_r p_lqr(x) E_lqr(x),
    d_lq(foreign;x) = sum_r p_lqr(x) [E_l,pi(q),r(x)-E_lqr(x)].

GQA repeats the immutable KV-head coordinates to query heads; all heads and
layers participate. No head is selected for an attractive effect. C is its own
foreign donor, so d_l,C(foreign;x)=0. Common C is not an identity intervention.

The recipient recurrence is

    attention_l(h_l;x,W)
      = sum_i p_lqi(x) V_lqi(h_l;x) + d_lq(W;x),
    h_(l+1) = ordinary_block_update(h_l, attention_l(h_l;x,W)).

This is a nonlinear forced recurrence with explicit forcing, not a linear
formula for the final score. Network depth is not the author's temporal change.
The identity replay checks that reconstructing the unchanged frozen contraction
matches the original instrument within its already fixed numerical tolerance.

## 3. What this can distinguish

A response difference under this construction cannot result from a change in
the final-row attention probabilities: those are fixed by the instrument. It
can result from value-carried information interacting with the remaining final
recurrence. It does not exclude generic meaning, lexical context, answer format
or position effects within those values. The block-specific installation has
no unique psychological mediation share.

Conversely, negligible effects on this suffix do not mean query relays contain
no information. Earlier question positions and direct body values remain; the
native probability bank itself can encode target-conditional body information.
Common-summary preservation is conditional on that bank, not proof that the
entire target inference is question-free.

## 4. Link to author-referenced measurement

Let phi_t(x) be a frozen SOURCE-supported text-action field and F_q(x;W,p) the
endpoint readout under the declared summary and reading law. On author u's two
registered views, define the retained paired evidence-operation contact

    C_tq(W,p) = Cov_u(Delta phi_t(x_u), Delta F_q(x_u;W,p)).

The same authors and evidence actions are used for all W. Delta is the fixed
selected-versus-control deletion contrast, not a difference of true human
personality labels. A change in C locates an interface dependency in this
finite experiment. It is not accuracy, psychological validity or an unrestricted
causal effect of a person's trait.

Both GRAIN retained and complement coordinates remain in the comparison.
An operational measurement interface must declare the evidence encoding, the
summary construction, the reading law and the author reference, rather than
attach a trait name to vector position alone. This preserves the project's
measurement spine while separating author variation from instrument behavior.

## 5. Computational contribution and next construction

The new explicit object is the per-layer forcing sequence {d_lq(x)} and its
controlled response, constructed without fitting scalar answers. After native
prefix encoding, common/foreign summaries and their forcing are ordinary tensor
operations. Native p and V still require the neural observer; the final recurrence
still uses its weights. This is not the finished LLM-free raw-text inference law.

The mechanism outcome decides whether a compiler may reuse a shared tail
summary or must retain a target-conditioned one. Construction of a SOURCE-only
reading law on new texts is a distinct next link. Replaying the native cache
more accurately does not complete it, so no fidelity optimization follows here.

## 6. Prior art and borrowed mathematics

[Todd et al.,2023/2024](https://arxiv.org/html/2310.15213v2), sections2.1-2.3,
already averages task-conditioned activations, extracts attention-output function
vectors and inserts them into new contexts. Sections3.1 and AppendixA distinguish
triggering a nonlinear computation from adding an output-vocabulary offset.
HM202 differs in its identical-surface token support, within-evidence five-query
V bank and frozen complete final-row probability law. It is not the first
task-conditioned activation experiment or proof of a new function-vector theory.

[Heimersheim/Nanda2024](https://arxiv.org/html/2404.15255v1) supplies established
path-intervention distinctions. The projection and weighted contractions are
classical linear algebra. [OpenAI/math foundations217-232](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
motivates an explicit coarsening question; its posterior/Jensen channel theorem
is not transferred to contextual V averages. Original18/100/44 remain open.

## 7. HM202 empirical consequence, appended after the sealed observation

2026-10-12: all290 batches/1450 inquiries completed in1579.120801 modelseconds;
one analysis and one alternate audit, no fit/retry/grid. Every final-row
probability and prefix V remained unchanged (maxerrors0). Identity/C logits
replay error1.907349e-5; head reconstruction4.010249e-6. All jobs terminal.

On the same24 public authors, A SOURCE/native score-action contact for the two
SOURCE views was .699/.532. Common V retained .722/.543, but its geometric
covariance amplitude fell42.386%/42.457%, with both conditional change CIs
negative. Thus higher r did not mean higher measurement accuracy or equal
action strength. Foreign A<-N summaries reduced contact to .178/.085, again
with both covariance-change CIs negative. Foreign C action change stayed at
5.945509e-7, whereas A change was .091417. Common C is not a self-control.

A complement contacts .479/.488 became common .439/.404 or foreign .177/.071;
all four corresponding covariance-change CIs were negative. N foreign contact
was .161/-.082; its historical held-author second SOURCE remained negative
(native/common/foreign -.143/-.334/-.426). All targets and signed crosscells
are retained; these are not psychological validity or human-trait accuracy.

This distinguishes value-carried sensitivity from an explanation involving
only changed final routing/normalization. It does not distinguish body-derived
summary from all lexical/task/position content, or make the native p bank
question-free. The next construction should inventory same-event body-visible
and body-blind V interfaces and isolate the body-induced innovation while
retaining its interactions. Do not seek another layer/head/r fidelity grid.

Canonical relay_value_summary_r1,45000 covariance accounts/1440 endpoint
accounts alternate-checked, same-agent not blind. Original18/100/44 stay open.
