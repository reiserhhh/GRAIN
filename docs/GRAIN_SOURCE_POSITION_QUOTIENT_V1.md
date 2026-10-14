# Source Position Quotients and Contextual Action Fields

Empirical packets: HM27/HM28, all48 frozen HM25 controlled paths retained.
This supplies an observation-support law, not a new personality factor count.

## Fixed Observation and Covector

For T tokens, t_j=0 for T=1 and -1+2j/(T-1) otherwise. Four observed moments
are m_k=(1/T) sum_j t_j^k h_j, k=0,...,3. A shared covector lambda acts as

    A_lambda(Delta H)=sum_k lambda_k dot Delta m_k
                      =(1/T) sum_j p_lambda(t_j) dot Delta h_j,
    p_lambda(t)=sum_k lambda_k t^k.

The source constrains polynomial evaluations, not arbitrary ambient coefficients.
For source position union P, the vanishing polynomial v_P(t)=prod_{a in P}(t-a)
generates the one-variable ideal of polynomials vanishing at P. With simple
distinct real nodes, R[t]/(v_P) is isomorphic to the product of their evaluations.
For degree<=d, identifiable rank is min(d+1,|P|), regardless of sample repetition.
This is classical interpolation/Chinese-remainder algebra applied to this map.

## Exact Current Alias

HM27 source P={-1,0,1}, so v_P=t^3-t. In the cubic class only
(lambda0,lambda1+lambda3,lambda2) is identified; Delta m3=Delta m1 on these nodes.
Changing lambda by v_P(t)*b for any hidden covector b leaves source observations
and all supported cold actions unchanged. It is not a proved zero psychological
effect and must not be deleted as universally irrelevant noise.

Correct frame transport is x'=Rx,lambda'=R^-T lambda. Independent minimum-norm
extensions can change predictions after a frame change. Joint constraints
identify an equivalence class; Lagrange interpolation chooses a convenient
representative with lambda3=0, whereas HM27's Euclidean solution uses another.
Their actions agree on the actual observation support even when coefficients differ.

## Source Weights and Conditions

For family i with T_i tokens, g_ij is the gradient in raw token units. The
fixed original objective has weights1/(N*T_i) on residual p(t_ij)/T_i-g_ij.
Rewriting it as node constraints gives observation T_i*g_ij, weight1/(N*T_i^3).
Grouped weighted means at each source node give the same supported action as
the stacked least-squares solution. The scale correction cannot be dropped.

Different families/paths still have different gradients at the same positions.
The node field estimates their shared part; it does not make the actual
decoder derivative globally constant. Keep the context-dependent remainder:

    omega_q(H;c)=lambda_q o O + r_q(H;c),
    Delta F_q=lambda_q[O Delta H]+integral r_q(gamma(s);c)[gamma'(s)] ds.

For fixed lambda this identity defines the remainder exactly. Estimating it
from source data is a separate empirical problem. Integrating a source-average
gradient cannot replace the target path's own conditional field. HM26's
potential gives one learned integrable approximation, not a unique neural circuit.

## Minimal Support Witness, Not a New Decoder Result

T3 adds positions{-1,0,1}, hence no cubic support gain. T4 adds +/-1/3.
In a one-coordinate toy displacement Delta h_j=v_P(t_j), the two functions
+v_P/-v_P are source-equivalent but differ on the T4 action by64/729.
The individual null action is32/729. Coefficients on this missing direction
are unconstrained in the absence of added observations. There is no finite
general extrapolation bound without an extra coefficient/function assumption.

The current96 actual action replays have alias norm0 and maximum difference
1.73e-18 between canonical node and old joint actions. The T4 witness is exact
rational algebra, not an actual neural or human measurement. A new source
T4 gradient would open this support direction; it need not improve prediction.

## Contribution and Next Acquisition

The useful contribution is the connection between a text observation map,
an exact identifiable quotient, and a minimal source-material design. It
separates estimation precision from adding a genuinely new observation direction.
It does not claim originality for polynomial quotients or interpolation.

Next use a fixed query and ownership-controlled new property span containing
four aligned tokens. Keep context/reference and action units explicit. Test
the previously invisible polynomial direction with a true decoder derivative
and symmetric finite patches; only then learn its source-supported action.
Do not silently substitute a4-token trust span for an anxiety span, mix semantic
coordinates by token length, or call this a discovered personality dimension.

## Sources

[NIST DLMF3.3(i)](https://dlmf.nist.gov/3.3.i): Lagrange interpolation and nodal polynomial.
[OpenAI/math](https://github.com/openai/math), fixed commit
adc7f1241b42e322a6451854ab7e4b4c146bf78a: finite-field factorization section02
suggested quotient/product evaluation; rational-formula hitting-list introduction
suggested that a finite evaluation design can miss a nonzero function. Only
those sections were read here. Their advanced algorithmic theorems are not
invoked for arbitrary neural networks; the position result has an elementary
exact proof and a full numerical readback in this implementation.
