# GRAIN finite joint observation law — HM10A

Status: executed and independently recalculated by the same agent, 2026-10-08.
This is a finite observer mechanism, not a newly validated personality scale.
Canonical result: `results/grain_math_response_v1/joint_law_r1`.

## 1. Objects and units

Let C and V indicate two controlled action-condition alternatives, and I
indicate whether their nodes share the same report root or occur in separate
reports. Zero means the declared non-target alternative, not absence of text.
Every cell contains two self-state reports with four condition nodes. The
condition actors may be self or another person; these roles are retained.
Report incidence is a relation, not recoverable from a bag of condition nodes.

For fixed lexical family, state-word frame and condition actor, observe all
eight cells m=0,...,7, with C=bit0, V=bit1 and I=bit2. The reference law is the
uniform product on these experimentally crossed coordinates. Natural language
tokens, authors and conditions are not assumed independent.

Three quantities must stay distinct:

- Whole-text frozen relevance readout F_Q(m)=w_Q^T h(m)+b_Q.
- Report marginal coverage, based on the original fixed condition functions.
- Joint report coexposure, the mean over report roots of min(C_C,C_V).

The last quantity is a declared measurement of coexposure, not a discovered
psychological interaction or emotional intensity. Parser status and unresolved
scopes propagate to missing observations; declared addresses are a separate
reference audit, never substituted into the automatic scorer.

## 2. Exact interactions

For each subset S of {C,V,I}, the anchored coefficient is

\[
m_Q(S)=\sum_{T\subseteq S}(-1)^{|S|-|T|}F_Q(T),\qquad
F_Q(A)=\sum_{S\subseteq A}m_Q(S).
\]

In particular,

\[
\Delta_{CV}^{I}=F_Q(1,1,I)-F_Q(1,0,I)-F_Q(0,1,I)+F_Q(0,0,I),
\quad m_Q(CVI)=\Delta_{CV}^{1}-\Delta_{CV}^{0}.
\]

The vector contrast obeys the same inclusion-exclusion formula. Because the
readout is affine, its intercept cancels in every nonempty contrast:

\[
m_Q(S)=w_Q^T m_h(S),\quad S\ne\varnothing.
\]

Nonadditivity in this branch therefore enters through the text embedding and
input composition, not through a nonlinear fitted readout. This does not
localize an attention head or identify the psychological cause of the text.
Under an invertible coordinate map h'=hR+t, w'=R^{-1}w preserves the contrasts.
Factor-label permutations transport coefficients; they do not justify
permuting natural text tokens or merging distinct semantic targets.

Anchored coefficients are not orthogonal variance components. For a declared
Bernoulli product reference p, use

\[
\chi_S(z)=\prod_{j\in S}\frac{z_j-p_j}{\sqrt{p_j(1-p_j)}},\qquad
\widehat F(S)=E_p[F\chi_S],\qquad
\operatorname{Var}_p(F)=\sum_{S\ne\varnothing}\widehat F(S)^2.
\]

Changing the reference changes the components and their energies, not the
observed cell values. The complete table and reference must travel together.
The product basis is inspired by the pinned OpenAI math Fourier manuscript;
no hypergraph influence bound is asserted for human language.

## 3. Source support and cold completion

The monomial design [1,C,V,CV] has rank 3 on cells 00/10/01 and rank 4 after
11 is added. Repeating singleton cells does not change rank. Additive,
synergistic and cancelling functions can share every singleton observation
yet differ at 11; these three functions are an exact witness, not human data.

New source families actually include joint cells. Fit only their contrast
means, globally or under a predeclared frame/actor/incidence key. For a cold
lexical pair, mask its 11 cell and predict

\[
\widehat F(11)=F(10)+F(01)-F(00)+\widehat\Delta_{CV}^{source}.
\]

This is a masked-cell transport experiment using three observed cold cells,
not a single-text personality predictor. Source, validation and application
lexical-pair families are disjoint (8/4/4); marginal action vocabulary is
shared intentionally. Frozen HM03 target functions are never refitted.
All four arms are reported; neither validation nor application selects an arm.

Two uncertainty views are retained: the primary application-family bootstrap
with source fixed, and a result-after independent source/application-family
bootstrap. Both preserve all within-family cells and paired arms. They do not
include encoder retraining, alternate observation rules or human sampling.

## 4. Actual finding and next boundary

Joint report exposure is strongly incidence-dependent and cannot be recovered
by additive marginals. Actual whole embeddings also have nonzero joint
contrasts; a source correction improves some cold readouts. However, a wrong
conditional key can retain much of the whole-coordination gain, indicating a
common encoding/composition term rather than an identified personal response.
Do not call the known min/max coexposure formula a new discovered state law.

The next HM10B object is an individually indexed response law, distinct from
this observer law. Reconcile the existing narrow-target/native-query programs
first; use them to observe joint response content and compare shared and
target-specific action. Do not replace that step with another grammar grid
or an arbitrary simulated coefficient recovery. Natural identity, individual
response identification and the original 18/100/44 scope remain open.

## 5. Reproduction

Use the Project Persona virtualenv, cached fixed readers and Stanza inventory.
No network/model download is needed or permitted by this packet.

```bash
cd "/Volumes/mobile3/projects/Sliced Utterance In-Context Assessment"
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "/Volumes/mobile3/projects/project persona/.venv/bin/python" \
  scripts/run_grain_math_joint_law.py --device mps --output /absolute/new/output
```

Use `--packet`/`--output` on `verify_grain_math_joint_law.py` for the new packet.
The original canonical result rejects reruns after completion. The current
diagnostic and its uncertainty verifier address canonical r1 only by default.
All previous outputs and failures stay frozen. No commit/push/lockbox/API or
private questionnaire operation is part of reproduction.

## References and originality

[Grabisch, Marichal & Roubens (2000)](https://pubsonline.informs.org/doi/10.1287/moor.25.2.157.12225)
establishes equivalent set-function representations. The publisher record
was checked; this packet does not claim to have audited its complete proof.

[Lengerich et al. (2020), §§1, 3–5](https://arxiv.org/html/1911.04974v3)
motivates reference-weighted, identifiable interaction decomposition. Its
piecewise-constant purification algorithm is not reproduced here; the small
complete product design permits a direct orthogonal character calculation.

Pinned OpenAI math commit `adc7f1241b42e322a6451854ab7e4b4c146bf78a`,
`A-uniform-influence-bound-for-hypergraph-properties-October-5-2026/build/fourier.tex`:
the complete displayed section was read, including biased characters,
Parseval, difference operators and the hypercontractive argument. Only the
finite product strategy is applied. Its preprint conclusions are not assumed
validated psychology. See the parent reading ledger for the group-theory
foundations section read separately and its still-unused applicability limits.

Möbius inversion, Fourier decomposition and factorial identification are
standard mathematics. The current GRAIN contribution is their typed report-
incidence integration and actual cold text-vector measurement, not a claim
that these transformations themselves were invented in this project.
