# GRAIN: shared local reading of query-conditioned transport

2026-09-10. NEXT44 develops the source-compiled block25 interface, not a new
human-trait regression or an LLM-free raw-text system.

## 1. Same material, different query-conditioned tangent

For both queries, each author's H16 BODY states, token positions and four GRAIN
directions are exactly identical. Let D_uf be that common initial direction and
T_q(H_u) the fixed model's upstream mapping to the final position before MLP25:

    d_uqf = J T_q(H_u) D_uf
    tauhat_uqf = wbar_sq^T d_uqf.

The suffix includes query/instruction tokens; it is not just the BODY. Equal
BODY inputs do not imply equal contextualized directions. Source program s
excludes the current old-source author or uses all four sources for new8.
The source vector includes both the paired gate/norm operator and output t.
Pooled t is a separate fixed mode, not an outcome-selected mixture.

Thus the interface separates **what perturbation is measured** (GRAIN D),
**how the contextualized model transports it** (d), and **how its local action
is read** (w). These are distinct computational objects, not three latent traits.

## 2. Reciprocal borrowing tests a replaceable computational component

Hold recipient d_q and its saved native action target fixed. Use both donors p:

    A_qp = wbar_sp^T d_uqf.

Keeping the target fixed matters: switching both donor reader and target would
instead assess the donor function and conceal the cost of the substitution.
The four entries are local component substitutions. They are not four new full
LLM forward interventions, nor evidence of a causal effect on the human author.

NEXT44 finds that new8 reciprocal own-t borrowing retains action r=.990935
(confirmation reader on N) and .994290 (N reader on confirmation), with
SSE/zero=.016022/.027966. Own diagonals were .996936/.998451. Reading remains
useful in both directions; query-specific amplitude differences are retained.

## 3. Exact symmetric allocation of the query contrast

Define w*=(w_N+w_C)/2, e=(w_N-w_C)/2 and d*=(d_N+d_C)/2. Then

    tau_N = w*^T d_N + e^T d_N
    tau_C = w*^T d_C - e^T d_C
    tau_N-tau_C = R + T
    R = (w_N-w_C)^T d* = 2 e^T d*
    T = w*^T (d_N-d_C).

The allocation is exact. It also equals the straight-line path integral of the
bilinear function F(w,d)=w^T d along simultaneously interpolated w and d; the
integral of each linear intermediate equals its endpoint average. Integrating
one component first instead would allocate the interaction differently.
The sum is invariant; the split is a declared symmetric convention, not a
uniquely identified psychological mediation.

[Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html)
is a methodological reference for explicit attribution axioms and paths. Our
two-factor product identity is elementary algebra, not a new attribution
theorem; here the operands are a compiled covector and a transported tangent,
not the paper's input-feature attribution problem.

With C=R+T, report both centered and uncentered accounting:

    Var(C) = Var(R) + Var(T) + 2 Cov(R,T)
    E[C^2] = E[R^2] + E[T^2] + 2 E[RT]
    pi_R = E[RC]/E[C^2]; pi_T = E[TC]/E[C^2]; pi_R+pi_T=1.

All sample moments use ddof=0 and the same observations/weights. A share may
be negative or greater than one; a vanishing denominator is left undefined.
The shares are not proportions of personality variance or causal importance.

New8 own mode has pi_T=.914272, pi_R=.085728; R/T RMS=.035711/.162922.
Pooled mode gives pi_T=1.048273, pi_R=-.048273, illustrating cancellation.
The complete compiled contrast tracks its native counterpart at r=.982463
(own) and .976153 (pooled). This is where the next development effort belongs:
query-conditioned incoming transport, not fine reader-coefficient optimization.

## 4. A useful common local reader, with a retained correction

After the fixed comparison, export w* and e without fitting. Own-mode new8
equal-reader action r=.995074/.997422 and SSE/zero=.010366/.007822. These values
are a post-outcome derivation from the same matrix, not another blind validation
or optimal blend. The original per-query readers remain exactly recoverable.

The common new8 reader pair cosine is .972757; tangent cosines average
.769-.781 by factor. Repeated source vectors for new8 are the same source
asset, not eight independent estimates of reader similarity. Neither cosine
identifies a global transport operator or a relationship between questionnaires.

## 5. Next mechanistic link

The current result connects GRAIN's common BODY directions to a largely shared
local readout via query-dependent d. NEXT45 will telescope the **difference**
in d through already saved pre-MLP25 updates, under the same w*. That distinguishes
where the query contrast is accumulated, amplified or cancelled. An MLP update
can amplify a difference inherited from attention; a large projected increment
is not automatically where query meaning originated.

Query dependence currently includes meaning, wording and instruction context.
Do not equate it with a human personality difference. Keep the existing external
criterion and interpretation layers separately linked after the mathematical
transport has been characterized.

See [results](../reports/grain_vector_mechanisms/COMPILED_QUERY_TRANSFER_REPORT_V1.md)
and [next plan](GRAIN_QUERY_TRANSPORT_FORMATION_PLAN_V1.md).
