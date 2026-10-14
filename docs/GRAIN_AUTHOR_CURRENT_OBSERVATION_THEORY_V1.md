# GRAIN: directed author evidence and contextual observation

Date: 2026-10-10. Empirical package HM93, author_current_r1. This is a finite
construction of an observation object, not an accuracy optimization or a new
general Hodge theorem. Original18/100/44 requirements remain open.

## 1. Object and protocol

An author's ordered evidence ledger and the instrument reading that ledger are
different objects. On a fixed three-state vocabulary let

\[
J_u(i,j)=\frac{N_u(i\to j)}{\sum_{a,b}N_u(a\to b)},\qquad
F_t^\pi(i,j)=O_t^\pi(\text{current }j\mid\text{previous }i).
\]

The states are declared action alternatives, not discovered psychological
states. The observation protocol pi fixes contextual encoder, framing, lexical
material, target-source seeds, covectors, span and query coordinate. We impose
a memory-one observation protocol by reading separate previous/current
records. We do not assert that a natural author or the encoder is Markovian.
For this protocol, the cold numeric author score is

\[
\mu_{u,t}^\pi=\langle J_u,F_t^\pi\rangle.
\]

An arbitrary full45-event document was NOT encoded to produce this score.
The288 actual short-record captures supply F; separate numeric paths supply J.
Thus the experiment tests an executable memory-one meter, not an approximation
guaranteed to match a whole-document LLM answer.

## 2. Traffic, endpoints and circulation

Use the full-matrix Frobenius inner product, not an edge sum with an implicit
factor of two. Define

\[
S_u=(J_u+J_u^T)/2,\quad A_u=(J_u-J_u^T)/2,
\]
\[
F_t^+=(F_t+F_t^T)/2,\quad F_t^-=(F_t-F_t^T)/2.
\]

Symmetric/skew orthogonality gives the exact identity

\[
\mu_{u,t}=\langle S_u,F_t^+\rangle+\langle A_u,F_t^-\rangle.
\]

On the unweighted complete graph, project a skew matrix A onto gradient
flows and their orthogonal divergence-free complement:

\[
d_i=\sum_j A_{ij},\quad a_i=d_i/n,\quad
G_{ij}=a_i-a_j,\quad C=A-G,\quad\sum_j C_{ij}=0.
\]

These are established graph projection identities. The implementation solves
the incidence least-squares problem; alternate verification uses this closed
form. The complete observation account is

\[
\boxed{\mu_{u,t}=
\underbrace{\langle S_u,F_t^+\rangle}_{\text{symmetric traffic}}
+\underbrace{\langle G_u,F_t^-\rangle}_{\text{endpoint contribution}}
+\underbrace{\langle C_u,F_t^-\rangle}_{\text{cycle contribution}}.}
\]

All terms stay in the instrument's units. No term is automatically discarded
as noise. If b=J1-J^T1, then d=b/2. For an open path of T states,

\[
b=(e_{\mathrm{start}}-e_{\mathrm{end}})/(T-1).
\]

The gradient in THIS decomposition is an endpoint account of empirical
antisymmetric pair mass. It is not the reversible physical drift or a
person's stable set point. Ring wrapping enforces b=0 by design; real text
must not be closed merely to remove its endpoints.

## 3. Duality with the observer

Decompose the instrument's odd part too:
F_t^-=G_t+C_t. Orthogonality gives

\[
\langle C_u,F_t^-\rangle=\langle C_u,C_t\rangle.
\]

Therefore a zero dynamic score can have three distinct causes: no author
circulation, no instrument cycle sensitivity, or orthogonality between them.
In more than three states the last possibility is substantive. It cannot be
repaired merely by making a scalar approximation more accurate.

On three states, a divergence-free current has one coordinate:
chi=A01=A12=A20, with reverse edges negative. Define

\[
\kappa_t=F_{01}-F_{10}+F_{12}-F_{21}+F_{20}-F_{02}.
\]

Then the cycle contribution is chi*kappa_t. Relabeling states changes both
representations but preserves their pairing. A scalar low/high naming of the
cycle requires a separate interpretation; clockwise does not mean high trait.

Every additive observer F(i,j)=g(i)+h(j) annihilates stationary circulation.
Neutral-history reset kernels have this property. A stationary two-state
current is zero; three states can carry one. This is NOT a claim that binary
MBTI targets cannot have dynamics: persistence, conditional response and
higher-order history remain possible.

## 4. Actual finite result

Two lexical settings (workshop/everyday), two frames (direct/recorded), six
behavior families, nine ordered pairs plus three reset records per battery:
288 actual Qwen2.5-1.5B early captures,25.276531 modelseconds. Fixed blocks0--16,
layer17 QKV, all12 heads, no new fit or application question/upper answer.
Three fields retain uniform projected value, local query and subject-bound
local query; raw unprojected current mean value is also saved.

Three45-edge laws have identical one-state occupancy1/3, dwell counts and
symmetric traffic. Their counts are

\[
N_+=\begin{pmatrix}5&8&2\\2&5&8\\8&2&5\end{pmatrix},\quad
N_0=\begin{pmatrix}5&5&5\\5&5&5\\5&5&5\end{pmatrix},\quad
N_-=N_+^T.
\]

Twelve Euler orderings per law produce36 toy paths, not36 independent humans.
All ring scores within a law agree under this meter; differences in higher
orders are deliberately unobserved, not absent from the original sequences.
Cold application also evaluates actual44-edge open paths without wrapping.

All432 family/reader/field/frame/wording accounts are retained. Absolute
opposite-current contrasts range .000064109--.402912726 in native readout
units, not Pearson r or a psychological effect size. Traffic contrast is0;
reversible-author cycle is0. Reset opposite-current contrasts are machine0.
Unprojected value-vector contrast RMS ranges .027104628--.048718739 across24
batteries. Open endpoint contributions reach .112225229, and must be kept.

The two lexical settings' entire6x6 contrast matrices have cosines:

| Field | Direct | Recorded |
| --- | ---: | ---: |
| Uniform value | -.055615 | .061713 |
| Local query | .046384 | .167685 |
| Subject-bound local query | .253215 | .484963 |

Diagonal contrasts keep the same sign in27/36 frame/field/family cells, but
this does not make the whole vector law wording-invariant. Both settings
were specified before calls; neither is a human validation sample. No fitted
inverse or parameter search rescues the sign changes.

## 5. Consequence for GRAIN static/dynamic labels

Occupancy summaries of independent event coordinates cannot see these laws.
However, a mean of CONTEXTUAL event values can see them, as the uniform-value
field and raw carrier show. The encoder can write path information into the
current event before aggregation. Therefore "computed by averaging" does not
imply "static"; "uses a transition statistic" does not imply psychological
dynamics. Classification should use declared reordering/occupancy controls,
not the output statistic's name alone.

A useful measurement object is (S_u,G_u,C_u; F_t^pi), preserving composition,
endpoints and directed evidence alongside an explicit observation protocol.
This connects HM92 contextual observations to a finite dynamic author ledger.
It does not integrate all historical GRAIN factors, infer a person's B_u or
establish a new personality dimension. The raw-word mapping to states remains
declared by design; it is not a general cold natural-text state extractor.

## 6. Prior art and novelty boundary

Before capture: Jiang/Lim/Yao/Ye2009, Statistical ranking and combinatorial
Hodge theory, sections4.2--4.3/5.1, supplies gradient/divergence-free projection.
Their preference ranking is not our evidence-transition measurement.
Source: https://arxiv.org/html/0811.1067.

Before capture: OpenAI/math pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a,
three-state reconstruction manuscript, introduction1--140 and
foundations1--165. Its order-forgetting reduction is justified because the
conditional ordering law is independent of the target. This inspires asking
when forgetting author-directed order is NOT harmless. Its tree threshold
and broadcast assumptions are not transferred to language or personality.
Source: https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex.

After execution, literature review located a closer precedent:
Nartallo-Kaluarachchi et al., LoG2024/PMLR2025, Decomposing Force Fields as Flows
on Graphs Reconstructed From Stochastic Trajectories. Read arXivv2 methods2.1--2.2
and related work/discussion, and inspected their public repository inventory.
They reconstruct continuous-time rates and decompose log sqrt(qij/qji) on a
triangulated phase space. HM93 instead pairs empirical joint pair mass with
a contextual observation; its endpoint gradient is not their reversible drift.
We neither transplanted their code nor proved a language Langevin model.
Sources: https://arxiv.org/html/2409.07479v2;
https://proceedings.mlr.press/v269/nartallo-kaluarachchi25a.html;
https://github.com/rnartallo/decomposingflows.

Graph-flow decomposition and dynamic individuality are not our inventions.
Old GRAIN U1 already tested adjacency beyond exact-bag nulls; M04 already
constructed shared conditional response. The current contribution is the
explicit author/observer dual account and empirical contextual readout under
matched occupancy/traffic, not "order matters" or a claimed priority theorem.
Its next scientific use is to distinguish changes in evidence circulation
from changes in observation protocol before interpreting a dynamic feature.

## 7. Reproduction and state

Code: grain_research/construction/math_author_current.py.
Runner: scripts/run_grain_author_current_hm93.py prepare/collect/analyze/apply.
Separate evaluation: scripts/verify_grain_author_current_hm93.py.
Existing r1 stages are nonoverwriting; do not rerun prepare/evaluation over them.
Canonical output: results/grain_math_response_v1/author_current_r1.
Final verification.json: VERIFIED_FINITE_HM93_NOT_GLOBAL.
Primary02 pending status stays sealed; final04 verification is authoritative.

17 focused tests pass. Four Python files pass format check. Ruff retains one
RUF007 style preference in the frozen verifier; no scientific repair or hash
change is made to hide it.421056 alternate scalar checks have readout error
6.217e-15; float32 mean versus float64 sum differs3.546e-7 under the predeclared
1e-5 carrier tolerance. Same-agent alternate arithmetic, not third-party audit.
No API, private data, new neural training, Claude, Git or lockbox operation.
