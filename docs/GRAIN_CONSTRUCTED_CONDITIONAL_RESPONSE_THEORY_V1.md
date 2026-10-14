# Constructed conditional responses and qualified author comparisons

2026-10-10. HM119 numerical construction and alternate-formula readback.
The original compatibility gate FAILED. The numeric audit passed separately;
this is not a full protocol pass or completion of the original 18/100/44 scope.

## 1. Scientific object, not another accuracy target

HM117 separated stable person readings from a focus-dependent physical edit.
HM118 showed that evidence/focus margins do not determine a measurement when
their pairing matters. HM119 constructs the conditional kernel for changed
behavior expressions, instead of fitting a predictor to application answers.

The empirical candidate is a relational response: changing who acts and whom
the instrument focuses on jointly changes the behavior response. In the
constructed observation kernel, this interaction corresponds across new
behavior instances more strongly than the focus-averaged actor response.
This is an instrument mechanism, not a newly validated human trait.

## 2. Inputs and explicit vector operations

The 288 HM115 control histories comprise three expression variants, two
opportunity wordings, six behavior families, two conditions, two behavior
polarities and two actors. Both people have the same designed opportunity.
The two current carriers differ at exactly one token: Person A versus Person B.

For each history, capture causal K/V primitives at blocks 0--16 only. Stop
before processing current tokens, layer17 attention, target questions or
answers. Let H denote this layer-indexed history, E_r the fixed focus seed,
and theta the frozen pretrained maps. A cold process constructs

\[
C_r(H)=F_{\theta,16}\circ\cdots\circ F_{\theta,0}(E_r;H).
\]

Within each block the explicit operation is

\[
\begin{aligned}
\widetilde h_\ell&=h_\ell+
 W_{O,\ell}\operatorname{Attn}_{\ell}
   (\operatorname{RMSNorm}_\ell(h_\ell);H_\ell),\\
h_{\ell+1}&=\widetilde h_\ell+W_{D,\ell}
 \left[\operatorname{SiLU}(W_{G,\ell}x_\ell)
       \odot W_{U,\ell}x_\ell\right],
\quad x_\ell=\operatorname{RMSNorm}'_\ell(\widetilde h_\ell).
\end{aligned}
\]

Attention retains the declared causal mask, excluded padding, rotary positions,
affine projection biases and grouped-query head mapping. Project the resulting
current states through the frozen layer17 K/V maps. The cold constructor uses
pretrained neural arithmetic: absence of a transformers model object or text
answers does NOT make it encoder-free or independent of LLM weights.

The six old source queries and output covectors define the observation. With
the registered scaling absorbed in q and the registered rotary coordinates,

\[
N_q^{\mathrm{weighted}}=
 \sum_h\sum_i\operatorname{softmax}_i(k_{i,\pi(h)}^\top q_{q,h})
                  v_{i,\pi(h)}^\top d_{q,h},
\qquad
N_q^{\mathrm{uniform}}=
 \sum_h\frac1L\sum_i v_{i,\pi(h)}^\top d_{q,h}.
\]

Here i indexes current-carrier tokens, not all history tokens; their states
already contain history effects. pi maps query heads to shared K/V heads.
All current constructions were sealed before opening the old focus-A states
for compatibility. No observed application current state or answer is an input.

## 3. Keep the complete actor/focus response

Suppress expression, wording, family, condition, observation and target indices.
Let b be low/high behavior polarity, o=+1 for actor A and -1 for actor B, and
r=+1 for focus A and -1 for focus B. Define

\[
G_{o,r}=\tfrac12(N_{+,o,r}-N_{-,o,r}),\quad
S(r)=\tfrac12(G_{+,r}+G_{-,r}),\quad
D(r)=\tfrac12(G_{+,r}-G_{-,r}),
\]
\[
S_\pm=\tfrac12[S(+1)\pm S(-1)],\qquad
D_\pm=\tfrac12[D(+1)\pm D(-1)],\qquad
G_{o,r}=S_++rS_-+oD_++orD_-.
\]

The subscripts on S/D denote focus parity in the second display. All four
sectors are retained. D_- is a behavior x actor x focus interaction, not an
actor-only axis. Averaging focus removes D_- by construction, whether it is
irrelevant noise or a useful relational response. Its role must be tested,
not decided from its failure to fit a single global physical direction.

The rectangular response needed by HM118 is constructible from these values:

\[
\Gamma_o=N_{-,o,+}-N_{-,o,-}-N_{+,o,+}+N_{+,o,-}
          =-4(S_-+oD_-).
\]

These finite identities are classical factorial algebra. They do not assert
that swapping name tokens is a symmetry of a pretrained neural network.

## 4. Consequence for a qualified author ledger

Keep an explicit joint evidence/focus ledger J_u and kernel N_q:

\[
S_{u,q}=\sum_{e,r}J_u(e,r)N_q(e,r).
\]

Only behavior actually assigned to the scored author is admitted as self
evidence. Other-actor responses remain instrument diagnostics. Within a family
and condition, the two count tables

\[
C^+=\begin{pmatrix}3&1\\1&3\end{pmatrix},\qquad
C^-=\begin{pmatrix}1&3\\3&1\end{pmatrix}
\]

have identical behavior and focus margins. Each total is eight. Their reading
difference is Gamma_own/4. With equal mass across six families and two
conditions, the per-family contribution is mean_c(Gamma_own)/24. Summing
families recovers the complete author comparison. These are designed numeric
ledgers, not human subjects or author-selected natural focus policies.

## 5. Actual finite findings

Comparing original wording0 with new behavior instances, weighted D_- has
mean response-vector cosine .636697 / .482938 across the two opportunity
wordings. All720 fixed family matchings have mean .229958 / .105017; the
correct matching ranks 7 / 9. Weighted D_+ instead has cosine .245677 /
.149530 and ranks 133 / 156. No metric, rotation, head, strength or rank was fit.

The result concerns six-target response vectors over six families and two
conditions. It is not Pearson r, AUC or a human p-value. Uniform observations
show the same qualitative difference. Family-specific weakness is retained,
particularly anthropomorphism and some condition-dependent A/O responses.
The qualified new-instance pairing contrasts have weighted RMS .035590 /
.022707 in the registered observation units. They are not personality scores.

## 6. Failed compatibility, successful numeric readback

Old focus-A K/V discrepancies are .000117540 / .000187635, below the frozen
.0005 limit. The reading discrepancy .000134776 exceeds the frozen .00005
limit. The original protocol therefore fails, with no rescue, threshold change
or root verification PASS. Focus-B was not independently checked against a
native reference. The findings above refer to the constructed kernel, not a
certified exact replacement for the native model.

The first cold attempt completed 576 evaluations but failed during export
because an output variable was shadowed by a history-buffer loop. Its error and
empty output remain. A separately sealed adapter repeated the same arithmetic
within the remaining original budget: 1152 actual current-only executions in
total, no repeated history capture, no new answers or fits. This repairs data
export, not scientific compatibility.

An alternate-formula readback recomputed all6912 scalar observations, all four
sectors, all28800 fixed matchings and every1728 response/432 author export.
Maximum observation error is 3.55e-15. It loaded neither torch nor transformers.
This is a same-agent numerical audit, not independent scientific replication.

## 7. Prior art and the next mechanism question

[Feng, Russell and Steinhardt](https://arxiv.org/html/2406.19501v1) already
study content/entity binding and compositional proposition readouts. HM119
does not identify their Hessian binding subspace. [Park, Choe and Veitch](https://proceedings.mlr.press/v235/park24c.html)
distinguish representation, probing and steering and formalize a causal inner
product. Our descriptive cosine is not an estimated causal metric. Attention
recurrence, parity decomposition and fixed matching tests are established tools,
not original mathematics claimed by this project.

The fixed [OpenAI/math matrix-section source](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex)
inspires complementary observations before eliminating components. Its harmonic
jet, PDE and uniformly bounded right-inverse hypotheses were not established
for text. It is an experimental-design inspiration, not a transferred theorem.

The next question is whether this qualified conditional field links to GRAIN
observations on the SAME evidence units and existing anonymous author factors.
Inventory the already available public numeric mechanisms and qualification
interfaces first. Do not launch another compiler-fidelity grid or automatically
promote a relational instrument effect to a personality factor. This question
is recorded, not executed by the HM119 closeout.

Canonical evidence: results/grain_math_response_v1/expression_response_r1,
especially 03/summary.json and numeric_readback/audit.json. Original sealed
code, failed export, compatibility failure and prior stopped budgets are intact.
