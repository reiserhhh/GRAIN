# Opportunity and conditional-response currents

2026-10-10. HM94 finite mechanism companion to HM93, not a new personality
predictor, general reconstruction theorem, or full GRAIN validation.

## 1. The observation object

HM93 separates a directed author evidence law from the contextual instrument
reading it. A current can nevertheless arise from changing observation conditions,
without an author-specific memory or dynamic rule. Neither all currents nor all
condition-related components should therefore be called personal dynamics/noise.

Let c index an explicitly declared opportunity, including expression position when
relevant. For author u, define its observation share Q_u(c), its ordered state-pair
law R_u(i,j|c), and a source-frozen common reference R_0(i,j|c). The observable law is

\[
J_u(i,j)=\sum_c Q_u(c)R_u(i,j\mid c).
\]

Retain two signed-account objects, rather than deleting the first:

\[
\begin{aligned}
J_u^{\mathrm{opp}} &= \sum_c Q_u(c)R_0(c),\\
J_u^{\mathrm{resp}} &= \sum_c Q_u(c)[R_u(c)-R_0(c)],\\
J_u &= J_u^{\mathrm{opp}}+J_u^{\mathrm{resp}}.
\end{aligned}
\]

For cells with Q_u(c)=0 the conditional law need not be estimated. The implementation
constructs counts and products directly, without assigning invented observations
to empty author cells. R_0 is learned from separate source authors. The response
is a signed reference difference, not a probability law, pure noise, or identified
personal causal response. Changing the reference changes these two accounts.

## 2. Linear graph projection, nonorthogonal opportunity accounts

On the complete four-state graph, write

\[
A(J)=\tfrac12(J-J^T),\quad
p_i=\tfrac14\sum_j A_{ij},\quad
G_{ij}=p_i-p_j,\quad C(J)=A(J)-G(J).
\]

C has zero row divergence; G records the endpoint imbalance under this edge-mass
convention. This is not a continuous-time rate generator or a physical reversible
Langevin force. Symmetric traffic, G and C use the established complete-graph Hodge
account from HM93. Since C is linear,

\[
C(J_u)=C(J_u^{\mathrm{opp}})+C(J_u^{\mathrm{resp}}).
\]

The two opportunity/response parts need not be orthogonal. For centered accounts,

\[
V_{\mathrm{raw}}=V_{\mathrm{opp}}+V_{\mathrm{resp}}
 +2\mathbb E\langle\widetilde C_{\mathrm{opp}},\widetilde C_{\mathrm{resp}}\rangle.
\]

Do not interpret either norm ratio as a percentage of personality information.
Negative cross terms allow subtracting a reference to increase variance. The
instrument kernel must be paired with the declared evidence law; increasing its
teacher-score fidelity does not by itself identify the law's psychological meaning.

## 3. What the finite toy establishes

Three condition-specific reset kernels have stationary emissions (.8,.1,.1)
up to state permutation. Each fixed-condition kernel satisfies detailed balance.
Condition-dependent independent emissions, under forward versus reverse condition
schedules of identical occupancy, produce opposite aggregate cycles without
personal memory. The full opportunity account explains the cycle: response is0.
Forgetting the conditions fabricates a response-cycle norm .1200249974.
An added conditional antisymmetric component at identical opportunity law gives
response norm .0122474487 and is retained, including its opposite sign.

Thus observable direction alone does not distinguish conditional response from
condition organization. This is a finite conditional-emission counterexample,
not a continuous-time stochastic-pump simulation or a new no-pumping theorem.

## 4. Natural numerical execution and consequence

Inherited safe caches contain276 authors/9165 comments, four ordered lexical GRAIN
slices per comment and E5 comment embeddings. Split92 source/184 application once;
source-only standardization/KMeans4 states and PCA16/KMeans6 anonymous contexts.
Four references (pooled/position/embedding/joint) are frozen, with no application
fit, model selection or neural retraining. No raw text/external label values read.
The contexts are vector clusters, not established topics or exogenous causes.
Inherited encoder/scorer/cohort dependencies remain; this is not a virgin trait test.

Technical panels0/1/2 are disjoint text views, not dates. Only within-comment slice
order is used. No cross-comment chronology or clinical change is inferred.

The joint opportunity cycle has same-author/all-stranger superiority .782858
[.759824,.807575] under the stated within-query-battery comparison. Raw .491581
and response .492751 show no corresponding superiority. This preserves author
selection/organization information, not proof that this conditional current is
a personality factor or a newly identified native LLM path.

Eight of81 FDR-adjusted links to inherited h*w query proxies occur in the
opportunity account; none in raw/response. These are not external Big5/MBTI
validity results. The strongest EI link is -.660872 (n42); openness +.534980 and
extraversion +.490052 (n142). Large proxy correlations can originate in selection
and a common reference, rather than an author-specific response operator.

## 5. The matched-stranger comparison has its own null

Selecting the nearest five strangers by Q makes comparisons harder in a way
correlated with the opportunity feature. The primary joint opportunity AUC .217572
cannot be compared with an unconditional .5 null. A post-primary, separately
registered companion retained candidate Q/features together and permuted identity
within query battery999 times. At1e-12 tie tolerance, its observed .216848 exceeds
the matched-null mean .053726 by .163122 (p=.001). The null95% range is
[.037681,.070851], not a population CI. No fits or parameter changes were made.

Two constant pooled/position opportunity accounts receive mathematical tie .5,
not rankings of floating-point residue. Primary files stay frozen; their near-.5
roundoff rankings are not scientific effects. Companion candidate-table arithmetic
was checked independently of its permutation-loop implementation by the same agent.

## 6. Ancestry and mainline boundary

- [Nartallo-Kaluarachchi et al.,2025](https://proceedings.mlr.press/v269/nartallo-kaluarachchi25a.html)
  already reconstruct graph flow from trajectories and decompose it. Their log-rate
  flow and physical interpretation are not this empirical pair-mass object.
- [Rahav,Horowitz,Jarzynski,2008](https://arxiv.org/html/0808.0015v2) distinguishes
  driven/stationary contributions and states no-pumping conditions. It motivated
  the mechanism question; its continuous-time equations were not imported.
- [OpenAI/math fixed three-state foundations](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
  removes ordering only when its conditional law is target-independent. This is a
  design lesson, not a text reconstruction threshold or proof of our method.

Conditional mixing, graph projection, clustering and permutation inference are
established tools. The contribution candidate is a useful measurement distinction
between author opportunity organization, conditional response and the instrument
reading them. HM94 changes that construction decision; it does not declare all
three newly discovered psychological mechanisms. Next question: can a frozen,
condition-indexed evidence observation connect those author accounts to target
vector action without using existing query scores as the observation's input?
Do not answer by tuning K, heads, ranks, smoothing or AUC. Original18/100/44 remain open.

## 7. Reproduction and frozen products

Canonical local output: results/grain_math_response_v1/opportunity_current_r1.
Numerical author arrays remain local derivatives; do not publish them by default.
The whitelist and all protocol/input hashes are in00/manifest.json. Source01,
cold application02, primary03, and post-primary04_matched_null are sealed.
verification.json and matched_null_verification.json use alternative arithmetic.
Plans/config/code are separate from this post-outcome interpretation.

To reproduce in a NEW output directory, use run_grain_opportunity_current_hm94.py
prepare/source/apply/evaluate and its verifier; all accept --out. The current
companion verifier is intentionally bound to the canonical run, while the companion
analysis accepts --out. Never rerun into this sealed directory or silently overwrite it.
The existing safe numerical caches and local dependencies are required; no download,
API call or neural training is implied. Focused tests:29 passed; seven new Python
files pass Ruff and formatting. Same-agent checks are not third-party replication.
