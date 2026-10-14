# GRAIN: history-conditioned observations and readout registration

Date: 2026-10-10. Finite empirical source: HM92, not a completed personality
instrument, a new general theorem, or an accuracy-selection experiment.

## 1. What changed in the measurement definition

An event's literal behavior span is not an intrinsic vector observation.
Its carrier depends on earlier text, and a separate offline readout depends
on how a frozen query is registered relative to that carrier. Retain both:

\[
V_e=\mathcal E(e,H_{<e};\pi),\qquad
O_t(e)=\mathcal L_t(V_e,K_e,H_e;q_t^{src}(\ell);\pi).
\]

Here \(\mathcal E\) is the frozen contextual encoder, \(H_{<e}\) preceding
text, \(\pi\) the observation protocol, and \(\ell\) a query coordinate.
\(H_e\) inside the readout is a subject-boundary gate, not the history itself.
The shared letter is historical notation; implementation names separate them.
Do not attribute variation of this observer to a human response parameter
\(B_u\). That remains a different object requiring author response evidence.

HM92 freezes Qwen2.5-1.5B through blocks0--16/layer17 QKV, twelve channels,
source query seeds, source behavior covectors, and a generic system. There
is no application target question, upper answer computation or new fit.
The encoder and target-oriented source construction still exist: this is
not a neural-free transformation of arbitrary raw text.

## 2. History and normalization are different paths

For an identical action span \(j\) define

\[
p^{loc}_{t,h,i}=\frac{e^{q_{t,h}(\ell)^T k_{h,i}}}
 {\sum_{i'\in j}e^{q_{t,h}(\ell)^T k_{h,i'}}},\qquad
F^{loc}_{t,j}=\sum_{h,i\in j}p^{loc}_{t,h,i}H_{h,i,A}c_{t,h,i}.
\]

The uniform projected value control is
\(U_{t,j}=\sum_h |j|^{-1}\sum_{i\in j}c_{t,h,i}\).
Also save the unprojected mean value vector; projections alone cannot establish
that the carrier changed. No token is removed from the full-text observation.

Move negation between report and event in K1. Full token bags, K2 action
tokens and positions, body length and query endpoint are equal. With K1 before
K2, K2 mean value changes in all twelve frame/family cells (RMS .207938--.280073);
the broader 24 earlier-event controls range .203415--.280073. Uniform readouts
also change. This cannot be produced solely by whole-body softmax competition.
These are unstandardized vector units, not r-values or personality effect sizes.

For the equal-length scope flip moved AFTER the tested action, raw values and
all three local readouts have exact zero difference in all24 controls. This
fits the prefix encoder's directionality. It is not a claim that every offline
readout is unaffected by every possible future edit.

## 3. A separate coordinate path

HM92's source query is rotated at the full-body endpoint. A future edit that
changes body length can move that query even while the current carrier and
its prefix stay fixed. In a result-after diagnostic of384 K2-first pairs:

- early K/V maximum difference: 0;
- original local paired readout maximum difference: .0080327872;
- query-position-aligned difference: 0;
- uniform value difference: 0.

This is a coordinate-registration action, not information flowing backward
through the encoder and not an author-state change. It is legitimate for an
offline whole-document readout, but must not be called a causal prefix score.
The primary equal-length scope contrast keeps \(\ell\) fixed and is unchanged.
The reverse-order local paired composition fraction up to .000700505 is
therefore retained rather than described as an exact universal zero; uniform
values have only machine-scale mixed coefficients in that arm.

## 4. Exact event composition on the declared design

On the fixed uniform64-point reference \(R\), let
\(Z_1=(x,g,a)\), \(Z_2=(y,h,b)\): behavior polarity, actor and scope per event.
For any frozen readout \(F_t\),

\[
F_t=F_{0,t}+F_{1,t}(Z_1)+F_{2,t}(Z_2)+I_t(Z_1,Z_2),
\]
\[
F_{0,t}=E_RF_t,\quad F_{k,t}=E_R[F_t\mid Z_k]-F_{0,t},\quad
E_R[I_t\mid Z_1]=E_R[I_t\mid Z_2]=0.
\]

This is standard product-space ANOVA, evaluated by the complete Walsh basis
of \((C_2)^6\). No truncation, learned factor count, accuracy fit or selected
axis is involved. Mixed-event support has at least one variable from each
event group. Parseval gives its exact response energy under this design.

Full paired mixed-event fractions range .089567--.316486 across24 cells.
K2 local paired fractions with K1 first range .460104--.700351 across12 cells.
These fractions describe finite observer responses, not percentages of
psychological information. They establish that event-wise additive summaries
are incomplete for this particular observation family and reference measure.

## 5. Same event marginals, different joint evidence accounts

Use \(\sigma=a y h\) and frozen \(\epsilon=.5\):

\[
dP_{\pm}=(1\pm\epsilon\sigma)dR.
\]

Both complete event marginals agree. Summing over \(a\) preserves the entire
\((x,y,g,h,b)\) law. Because an \(a\) flip preserves bags, full token-bag laws
agree too. Every event-additive function has equal expectation in these
synthetic accounts. Actual readouts satisfy the exact identity

\[
\Delta\mu_t=E_{P_+}F_t-E_{P_-}F_t
 =2\epsilon E_R[\sigma F_t]=2\epsilon E_R[\sigma I_t].
\]

K2 local paired diagonal contrasts with K1 first have absolute range
.000210089--.007870049; reverse-order scope contrasts are exactly0. All signs
remain. Uniform and paired signs differ in5/12 cells, as do full and K2-local
paired signs. Thus detecting composition does not define a high/low trait key.
The six-by-six family/reader contrast matrices retain both their battery mean
and orthogonal reader-specific part. The latter carries61.985%--66.043% of
K2-local contrast energy in the two frames, not a universal psychological factor.

The seven product terms in changes of p, H and c integrate to the same
\(\Delta\mu_t\). These are algebraic accounts, not causal activation patches
or separately identified mediator effects.

## 6. Consequence for GRAIN construction

The observation interface should retain event identity, protocol/coordinate,
contextual carrier and joint history information before author aggregation:

\[
\mu_{u,t}^{\pi}=E_{P_u^{\pi}}O_t(e,H_{<e};\ell),\qquad
\Delta_{u,v,t}^{\pi}=\int O_t\,d(P_u^{\pi}-P_v^{\pi}).
\]

An anonymous GRAIN feature bank can carry these joint observations without
calling them personality dimensions. Source reference distributions, units,
evidence qualification and external/behavioral interpretation remain separate.
Observed history conditioning is not itself a discovered human trait dynamic.
This package defines and implements a finite observation mechanism, not an
integration of every historical GRAIN factor product or the original18/100/44.

The next useful question is whether a shared, target-indexed contextual
observation operator preserves author contrasts beyond this finite designed
support. It must distinguish encoding, coordinate registration and evidence
qualification, rather than tune a reader to be more accurate. Do not simply
add this item to an existing instrument as a named trait.

## 7. Prior art and the limited OpenAI/math connection

Coecke, Sadrzadeh and Clark2010 §§3.3--3.6/4.1 define grammatical typing and
tensor contraction for compositional meaning, including role/content coupling.
That is substantive prior art, not something HM92 invented. HM92 does not
implement their categorical functor or prove Qwen implements it.
Source: https://arxiv.org/pdf/1003.4394.

OpenAI/math pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a,
metric/unitary-connection manuscript, sections/06-matrix.tex lines146--280,
separates fixed coordinate displacement from response remainder and bounds
them using smooth PDE/Taylor assumptions. Its useful design lesson here is to
keep coordinate action and response remainder distinct. Discrete text lacks
the stated shrinking-radius assumptions: no PDE bound is transferred and no
new high-mathematical guarantee of personality recovery is claimed.
Source: https://github.com/openai/math.

## 8. Reproducibility state

Canonical results: results/grain_math_response_v1/event_history_observation_r1.
00 frozen material/source; 01 actual1536 captures/140.699147 modelseconds;
02 primary; verification.json alternate arithmetic; 03_interpretation signed
author product accounts; 04_position_diagnostic coordinate control.
Primary pending-status summary stays sealed; verification.json is final:
VERIFIED_FINITE_HM92_NOT_GLOBAL. Alternate checks1,016,752, maximum7.106e-15.
43 focused tests pass; same-agent alternate arithmetic, not a third-party audit.
No API/private data/MEPS/new neural training/Git/Claude/automation/lockbox.
