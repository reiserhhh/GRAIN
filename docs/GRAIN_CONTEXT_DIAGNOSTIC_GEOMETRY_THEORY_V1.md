# Relative diagnostic geometry and calibration transport

HM153, 2026-10-11. Finite analysis and alternate arithmetic complete.
This develops the conditional evidence interface, not an accuracy-optimized
predictor. Mature compositional/projective geometry is used, not invented.
Original18/100/44 stays OPEN. Prior scientific code and results remain sealed.

## 1. A nonzero context score is not yet a relative diagnostic change

The HM152 prefix contribution K is the sum of per-token changes in the H/L
normalized evidence. Apply the same positive lexical multiplier g(v) to both
reset laws:

    p_Hcarry(v) = p_Hreset(v)g(v) / E_Hreset[g]
    p_Lcarry(v) = p_Lreset(v)g(v) / E_Lreset[g].

Then k(v)=log E_Lreset[g]-log E_Hreset[g] can be nonzero but is constant across
v. The vocabulary preference changed, but it did not change which word is MORE
diagnostic of H versus L compared to another word. This is a concrete rival
account, not simply an appeal to "context matters".

The relative diagnostic change is the cross-ratio

    Gamma(v,w) = log[(p_Hcarry(v)p_Lcarry(w))/(p_Lcarry(v)p_Hcarry(w))]
               - log[(p_Hreset(v)p_Lreset(w))/(p_Lreset(v)p_Hreset(w))].

For the frozen linear output head, Delta=(h_Hcarry-h_Lcarry)-(h_Hreset-h_Lreset),

    Gamma(v,w)=(W[v]-W[w]).Delta.

Constant logit gauges and normalizers cancel exactly. Gamma=0 everywhere iff
the two context tilts are proportional, equivalently a common positive
multiplicative lexical action is sufficient. Nonzero Gamma establishes a
hypothesis-dependent relative diagnostic action on this text prefix; it does
not exclude generic nonlinear grammar/prompt interactions or identify a human
trait. The distinction is lexical-relative versus calibration, NOT
"personality versus noise".

## 2. The compiler object is a relative field plus calibration

Let U(v)=W[v].Delta and DeltaA be the four-corner partition difference. Then
k(v)=U(v)-DeltaA. Specify a reference q proportional to the geometric mean of
the two reset laws; it is not an author population or psychological posterior.

    m_q=E_q U
    S_q(v)=U(v)-m_q
    a_q=m_q-DeltaA
    k(v)=S_q(v)+a_q.

Both accounts are logit-gauge invariant. They remain reference-q dependent:
changing q transfers a constant between S and a while keeping their sum and
all Gamma unchanged. This is an observational convention, not a unique causal
information split. Keep BOTH terms; do not remove calibration as noise.

The geometric mean law can be evaluated at the average reset prediction vector
because the head is linear with verified zero bias. This does not mean that
this midpoint occurs as a natural decoder state or preserves future KV caches.

Two complementary coordinate-invariant quantities are

    D(Delta)=max_v W[v].Delta-min_v W[v].Delta,
    sigma_q(Delta)^2=Delta^T G_q Delta,
    G_q=W^T(diag(q)-q q^T)W.

D is a Hilbert projective distance between normalized positive tilt measures.
It is a seminorm on hidden differences; its kernel contains directions making
every output logit shift by the same amount. The measurable relative object
therefore lives in the quotient by that kernel, not in one uniquely named
hidden axis. sigma_q retains vocabulary probability weighting rather than
allowing rare extremes alone to set the interpretation. G_q is the categorical
Fisher pullback for this output law, not the latent human-trait metric.

No extra metric fit is required. This provides a concrete GRAIN compiler
contract: actual token evidence requires the conditional relative field AND
its calibrated scalar offset. Averaged vectors or a fitted high-dimensional
transport do not automatically preserve that pair.

## 3. Actual finite observations

All972 HM152 cases/702 existing articles,15552 aligned second-token positions,
all108 persona conditions. Whole vocabulary151936. All original conditions,
14 refusals/519 generator truncations retained; generation seeds NOT authors.

| Finite quantity | Mean | Median | 95th percentile |
| --- | ---: | ---: | ---: |
| Weighted SD_q(U), nats per position | .121245 | .079826 | .371239 |
| Hilbert oscillation D, nats per position | 1.598720 | 1.180407 | 4.154505 |

The weighted variation shows that the effect is not only an extreme-vocabulary
range. No significance test treating token positions as independent people is
used. Native rounding is retained and checked, not tuned to create a threshold.

Across16-token suffixes, mean absolute shape sum is .980760nats; mean absolute
calibration sum .647988nats, while HM152 mean absolute TOTAL K is .661625nats.
Signed shape/calibration means are -.042262/+.050106, almost cancelling.
These numbers are not information percentages. Post-outcome grouping of all108
conditions finds72 opposite-signed shape/calibration contrasts and27 where the
shape sign opposes the total K sign. Dropping the constant account would change
the meaning even after correctly identifying a nonconstant field.

## 4. Multiple targets on identical evidence

All54 neutral articles are evaluated under E/N for all3 prompt families:162
pairs,2592 token positions. Common q0 is the geometric mean of allfour reset
laws in a pair. Compare q0-centered fields, not their arbitrary raw origins.

    cos_q0(S_E,S_N)=Cov_q0(U_E,U_N)/(SD_q0(U_E)SD_q0(U_N)).

Observed mean -.127250, median -.171967, range -.994809 to +.998547;
1626/2592 negative. These are conditional field angles, NOT E/N trait-score
correlations, human sample sizes, or a discovered psychological relation.

Post-outcome ALL18 task/family groups show named-family means negative in all6
tasks (-.123 to -.341), while anchored email/neighbor/opinion means are slightly
positive (+.044,+.054,+.028). Thus a single pooled "shared personality direction"
would conceal condition and hypothesis-wording dependence. This is a reason
to test target-consistent field structure across wording, not to flip signs.

The saved "shared_variance" and "specific_variance" refer ONLY to the algebraic
combinations (S_E+S_N)/2 and (S_E-S_N)/2. Their sum equals (Var_E+Var_N)/2;
they are not independent latent components, and their mutual covariance is
(Var_E-Var_N)/4. No orthogonality, psychological shared/unique fractions or
causal direction is inferred from these names.

## 5. Theory progress and next discriminating question

HM152 says how reset units change evidence. HM153 specifies WHAT changes: an
alternative-token relative field plus scalar calibration, and demonstrates
both in actual frozen observations. It does not discover a new probability
theorem or already solve human personality inference. This is a step from
scalar context effects toward a measurable conditional action object.

The next substantive unknown is whether there is a reusable TARGET-consistent
relative field across different descriptions of the same trait, rather than
only prompt-specific interactions. The54 identical neutral articles already
have both targets and all3 wording families, so a common-reference six-field
Gram can compare same-target agreement with cross-target agreement without
new decoding or unconstrained matrix fitting. Balanced target relabelings
within each wording family can separate family coherence from target grouping;
they are descriptive finite design comparisons, not an automatic randomization
p-value. Seal that new discriminator before computing its results.

Retain GRAIN choice composition, conditional response, state, factuality/event
identity/author qualification and reference populations. The present readout
still needs model-produced states and stored output weights, not raw-text
LLM independence. Personal author scoring and psychological meaning remain
separate integration tasks; no synthetic writer prompt supplies those for free.

## 6. Execution and sources

1134 output-head projection batches/38880 projected vectors,82.081seconds;
zero decoder forwards, new generation, APIs, training, private data or Git.
All selected vectors/CSV/sums/108conditions/2592 cross accounts checked:
270703 alternate comparisons, max algebra error2.66e-15. Selected native
closure max8.84e-6; fixed full-vocabulary spots projection max2.60e-5 within
sealed abs1e-4+rel1e-5.8 main and2 cross full-vocabulary spots, not every
field independently recomputed. Same-agent alternate arithmetic, NOT blind
replication.32 focused tests PASS; four prospective Python files lint/format
PASS. Analysis/audit caps600/120seconds; no cap expansion or rerun.

Canonical: results/grain_math_response_v1/context_diagnostic_shape_r1.
Prospective: docs/GRAIN_CONTEXT_DIAGNOSTIC_SHAPE_PLAN_HM153.md.
The descriptive summary script/receipt are explicitly post-outcome companions,
not retroactively added preregistered tests or changes to the scientific seal.

[Cohen and Fausti, Section4 Eq24](https://arxiv.org/html/2309.02413v2): Hilbert
distance on the finite positive cone. Positive linear contraction is NOT
applied to transformers. [Compositional geometry Section2.1](https://link.springer.com/article/10.1007/s11004-020-09895-w)
supplies perturbation/CLR precedent. HM10B1 already introduced CLR/Fisher;
HM88/102/108 handle other observation/readout normalizers. PICLe and Persona
Vectors precede persona likelihoods and activation interventions. Our finite
addition is their distinction at the GRAIN carried/reset observation interface.
OpenAI/math operation-compatible dual testing motivates choosing an invariant
test; Specht/group-representation hypotheses are not asserted for text.
