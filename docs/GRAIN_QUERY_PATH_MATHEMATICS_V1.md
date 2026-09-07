# GRAIN query-path mathematics: from evidence to a response operator

2026-09-07. Mathematical working note for the D1-D4 real-author mainline.
Equations describe the experimental functions and make testable approximations;
empirical findings are recorded separately after the fixed packet completes.

## 1. Observable chain

Let x_i be the ordered text evidence for author i, q a fixed trait query, and
E_j an evidence transformation (selected removal or a matched alternative).
The native score is

\[
s_{iq}=F_q(x_i),\qquad s_{iqj}=F_q(E_jx_i).
\]

D1 establishes usable associations between these native-query outputs and recorded
source traits. D2 now studies the computation linking the evidence to those outputs.
Personality labels are not used to find directions or tune D2 interventions.

## 2. Query-first evidence coordinates

Using a frozen encoder e and trait-pole descriptions a_q^+,a_q^-, define

\[
v_q={e(a_q^+)\over\|e(a_q^+)\|}-{e(a_q^-)\over\|e(a_q^-)\|},\quad
r_{iqk}=\left\langle {e(x_{ik})\over\|e(x_{ik})\|},v_q\right\rangle.
\]

The four largest |r| select evidence from either pole. This is a query-guided
coordinate system over existing text, not an assumption that all authors live
on one personality-word axis. Selection is evaluated by its effect on the native
query, relative to removing a similar number of tokens from other comments.

## 3. A measured vector route

Let h_{iq} be the block16 vector at the final query position. For the transformed
input obtain h_{iqj} and d_{iqj}=h_{iqj}-h_{iq}. Define g_{iq}(h) as the remaining
network computation with the ORIGINAL recipient input and all other states fixed,
but with that one vector replaced by h. We can execute this intervention.

\[
M_{iqj}(\alpha)=g_{iq}(h_{iq}+\alpha d_{iqj})-g_{iq}(h_{iq}).
\]

The natural evidence change has an exact, experimentally measurable decomposition:

\[
\Delta s_{iqj}=F_q(E_jx_i)-F_q(x_i)=M_{iqj}(1)+R_{iqj},\qquad
R_{iqj}=F_q(E_jx_i)-g_{iq}(h_{iqj}).
\]

M is the measured contribution transmitted through the chosen query-end site
under this intervention. R collects other positions, context changes and their
interactions for this reference computation. Both are useful quantities; R is
not defined as disposable noise. This is an exact add/subtract identity, not an
assumption of independent psychological components.

## 4. Local mathematical form and curvature

For a differentiable continuation g along the intervention segment, the
fundamental theorem of calculus gives

\[
M(\alpha)=\int_0^\alpha \nabla g(h+\lambda d)^\top d\,d\lambda.
\]

This is a vector displacement interacting with a query- and recipient-dependent
response field. The same displacement need not have the same consequence in a
different context. The two observed doses .5 and1 provide a finite curvature
contrast K=M(1)-2M(.5); K=0 for an exactly linear response on these sampled doses.

If the Hessian operator norm is bounded by L on the segment, Taylor's theorem gives

\[
|M(\alpha)-\alpha\nabla g(h)^\top d|\le {L\alpha^2\over2}\|d\|^2.
\]

L is a mathematical condition, not a constant estimated by the present two doses.
The finite K is directly measurable without a full Hessian sweep.

## 5. Compact shared operator and its three error sources

For an orthonormal basis U learned from discovery-author displacements, fit b_q
and predict M by \(\widehat M=\alpha b_q^\top U^\top d\). The following bound
separates the reasons that approximation can depart from an observed response:

\[
|M-\widehat M|\le
\alpha\|(I-UU^\top)\nabla g(h)\|\,\|(I-UU^\top)d\|
+\alpha\|U^\top\nabla g(h)-b_q\|\,\|U^\top d\|
+{L\alpha^2\over2}\|d\|^2.
\]

They are respectively an omitted directional component, a recipient-specific
response gain, and curvature. This motivates targeted development after an effect
is observed: expand a shared basis, condition the gain on author/context, or add
a nonlinear response term. It does not require solving all three in advance.

D4 starts with rank2/rank8 SVD plus ridge1, using only discovery authors. A fixed
held-author split tests whether the compact operator predicts new responses.
These operators approximate an observed LLM computation from its internal state;
replacing the entire encoder is a distinct future step.

## 6. The GRAIN author/group link

Let z_i be an independently scored GRAIN profile and let \(\mathcal R_i\) be the
vector of observed query/evidence/patch responses. A group hypothesis is

\[
z_i\approx z_j\quad\Longrightarrow\quad \mathcal R_i\approx\mathcal R_j.
\]

D3 tests this by predicting held authors' response profiles from discovery-author
neighbors in the frozen GRAIN geometry, with a semantic E5 geometry comparison.
This places the author/group relation inside the evidence-to-judgment mechanism,
rather than treating group structure or a high prediction score as the endpoint.

The integrated candidate is therefore a conditional response family, not a
single universal personality vector: query-specific evidence coordinates feed
internal changes, a shared operator explains transferable effects, and author/
context coordinates may organize the remaining gain and interaction structure.
