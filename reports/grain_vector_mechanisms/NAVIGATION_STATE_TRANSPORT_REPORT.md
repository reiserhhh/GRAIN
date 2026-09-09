# GRAIN navigation: reduced state construction and its reachable space

2026-09-09. NAV goal V6 remains **ACTIVE**. Route A now has an actually executed
state constructor, not just candidate-head localization. The packet also
establishes an exact algebraic bridge from that constructor to route B.

## Findings and decision

1. Four extracted contextual blocks plus a construction-only residual operator
   reconstruct useful P/Q variation on held simulated events. Fixed-function
   fidelity is r=.737(E),.715(T/F),.830(low state) for the linear operator.
2. Real-author transport exposes a coverage problem, not a universally successful
   replacement:84%-91% of its state reconstruction error energy is outside the
   construction residual span. Within-span coefficient errors also matter to
   the score, especially for low state.
3. Projecting the generated state and compiling the same operation directly into
   effective projections are numerically equivalent: maximum score difference
   3.66e-15 across1152 mode/query/world/kernel comparisons. This is an actual
   A-to-B compilation identity, not an independent empirical replication.
4. Preserve earlier useful token-only T/F and contextual-state programs. This
   new all-state candidate does not replace them. Use the diagnosed coverage and
   coefficient gaps to develop conditional operators, not a human-label grid.

## Construction and declared runtime

Data: six construction scene skeletons, eight actor/opportunity/renderer
combinations each,48 paired snapshots per query. Development and held scene
states were excluded from operator fitting. No human texts or psychological
labels were used to fit the operator; MEPS remains external-only.

The early state x concatenates ordered query rows and the separate final input
position at layer4. The teacher target y has exactly the same coordinates at
layer17. Dimensions are95,232(E),56,832(T/F),144,384(low state). A pooled sentence
embedding is not substituted for these within-query coordinates.

Runtime loads four extracted Transformer blocks and a pretrained embedding
table (weight file841,138,320bytes), but not the original28-layer model, LM head
or actual17-layer P/Q inputs. Thus it is **extracted-neural-plus-state-operator**,
not Transformer-free or pretrained-knowledge-free. Generated P/Q is saved and
scored by the unchanged synthetic-reference function. The construction includes
288 early-state captures in27.20seconds;48 per query are used for fitting,
remaining captures are for held comparison. Natural edits cost288 four-block
calls/97.08seconds. External369 author/material units cost923calls/201.95seconds.

## Explicit mathematical program

For construction matrices X,Y, let R=Y-X. Normalize early coordinates using
construction moments only:

~~~text
s^2 = mean_i ||X_i - mean(X)||^2
U_i = (X_i - mean(X))/s
k1(u,v) = u'v
k2(u,v) = u'v + (u'v)^2/2
Kc = C K C, C = I - 11'/n
W_d = (Kc_d + I)^(-1) [R - mean(R)]
y_hat_d(x) = x + mean(R) + centered_k_d(x,U) W_d
~~~

The second kernel preserves pairwise products without materializing a
dimension-squared matrix. Both kernels, alpha=1, and the four-block cut were
fixed before any real-data result. Upper-state mean and constant residual
transport are explicit controls, not discarded because their correlation is
undefined. Effective degrees of freedom are about8.4-8.7(linear) and14.2-14.8
(quadratic); every correction lies in a construction span of at most47 dimensions.

Mathematical precedent: data-driven state reduction motivates learning a small
operator from full-system snapshots rather than blindly optimizing a downstream
label predictor. See [Peherstorfer & Willcox,2016](https://www.sciencedirect.com/science/article/pii/S0045782516301104)
and [Kramer,Peherstorfer & Willcox,2024](https://www.annualreviews.org/content/journals/10.1146/annurev-fluid-121021-025220).
Their operator-inference results concern structured dynamical systems; this
early-to-late neural-state emulator is an adaptation of the reduction idea, not
their PDE convergence theorem applied to a Transformer. Kernel regression,
snapshot spans and orthogonal projection themselves are standard methods.

## Held worlds and immediate real connections

Held-world figures below describe24 renderings of three scene skeletons.
No human questionnaire is the target in this table.

| Target | Linear function r | Quadratic function r | Linear function MAE | Linear P nRMSE | Linear Q nRMSE |
|---|---:|---:|---:|---:|---:|
| Extraversion | .7374 | .7503 | .0758 | .9599 | .7596 |
| T/F | .7150 | .7105 | .0786 | .9241 | .7478 |
| Low state | .8298 | .7433 | .1183 | 1.0135 | .8819 |

State nRMSE uses construction between-world variation, not the very large common
state offset. The method reduces some state error while retaining useful score
ordering; it does not reconstruct every hidden coordinate precisely. A higher
polynomial degree does not consistently improve the operation.

On the12-author natural intervention panel, linear function-level r is
.090(E),-.323(T/F),.127(low state); quadratic is-.001,-.284,.238.
Paired low-state endorsement responses reverse direction (linear r=-.538,
quadratic-.716); many other edit effects are attenuated. These complete negative
conditions determine the next mechanism work, rather than being averaged away.

Frozen external links:

| Material/target | n | Linear human r / AUC | Quadratic human r / AUC |
|---|---:|---:|---:|
| PANDORA Big5-only E | 213 | r=.0428 | r=.0117 |
| PANDORA MBTI-only T/F | 64 | AUC=.3043 | AUC=.2933 |
| Free-chat low state | 46 | r=.1880[-.0666,.4151] | r=.2301[-.0181,.4495] |
| Direct-answer low state | 46 | r=.1653 | r=.2090 |

The T/F polarity reversal is retained, not flipped after seeing labels. Its
linear human r is-.317[-.516,-.089]. Earlier token-only T/F AUC=.705 and shallow
Jacobian free-chat state r=.351 remain useful separate results. No candidate
selection, sign change or norm adjustment was fitted to these human outcomes.

## Reachability decomposition: where the missing computation is

Construct an orthonormal row basis V of the synthetic centered residual span.
For an evaluation pair, the actual teacher residual is used **only diagnostically**:

~~~text
b(x) = x + mean(R)
y_oracle = b(x) + V' V (y - b(x))
e_out = y - y_oracle
e_in  = y_oracle - y_hat
e_total = e_out + e_in
||e_total||^2 = ||e_out||^2 + ||e_in||^2
delta_f = L'e_out + L'e_in
~~~

The oracle requires actual later-layer states and is never an independent
candidate or a deployable upper-bound guarantee. Projection minimizes state
distance in this fixed affine subspace; it does not necessarily maximize score
correlation or minimize projected error.

| Natural target | Outside state-error fraction, linear | Actual function r | Diagnostic projected-span function r |
|---|---:|---:|---:|
| E | .8955 | .0901 | .4336 |
| T/F | .8376 | -.3228 | .4211 |
| Low state | .8913 | .1272 | .8045 |

Quadratic outside fractions are.9102,.8527,.9084. On held synthetic scenes the
same diagnostic correlations are.9992(E),.9182(T/F),.9848(state). This points
to two distinct development questions:
- Can a broader, compositional construction dictionary cover natural residual directions?
- Can the observable early state compute the useful within-span coefficients?

Full-state energy and target-directed importance differ. For natural low state,
outside score-error RMS=.0754 and linear inside RMS=.0796 even though outside
accounts for89% of total state-error energy. Large error energy therefore must
not automatically be called psychological noise, nor does small energy imply
an unimportant mechanism. Signed score accounting error is1.58e-15; absolute
orthogonal energy accounting error is1.69e-9.

## Exact A/B linkage

The fixed reader has a matrix L for mean/configuration/final contributions and
a reference offset a. Contract the generated-state operator offline:

~~~text
parts_A = L' y_hat + a
parts_B = L' x + L' mean(R) + a + centered_k(x,U) [W L]
score = anchor + sum(parts)
~~~

Exporting W L instead of W produces exactly the same three useful contributions.
This removes full-state output synthesis, not the early contextual calculation.
The compact projection files were actually exported and checked on both worlds
and natural texts. This closes one algebraic piece: once a valid upstream state
program exists, its fixed linear reader can be compiled into a smaller B program.
It does not independently solve the upstream coverage problem.

## Artifacts, tests and next work

Canonical private artifacts under results/grain_vector_mechanisms/navigation_v1:
- NAV-02/state_transport_r1:4-block weights, synthetic operators, frozen provenance.
- NAV-10/state_worlds_r1/analysis_r3: state and action evaluation.
- NAV-10/state_natural_r1/analysis_r3: actual generated states and natural-edit comparisons.
- NAV-10/state_external_r1/analysis_r3: fixed external links and intervals.
- NAV-02/state_reachability_r1: residual decomposition and compiled projection files.

External analysis first encountered an undefined constant-control correlation;
the report now preserves undefined r and chance AUC rather than crashing or
inventing zero correlation. A transient missing import in the fix was caught
by the new tests and corrected. Canonical r3 analyses use the corrected code;
earlier partial analysis folders are retained. No neural collection was rerun.
Full repository tests:2747passed,3skipped before the last four new geometry/control
tests; final focused suite12passed, including those four. Scoped lint passed.

Next maintain the NAV ledger: complete fixed-reference/group/cache/runtime cards
and develop target-conditioned relational coverage from these diagnosed gaps.
Do not widen a blind alpha/depth grid, delete failed targets, reopen an old
lockbox, fit on MEPS, or declare the whole raw-text inference goal complete.

