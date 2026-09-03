# GRAIN M4-C.3 Physical-Edge Composition Audit

Date: 2026-07-31  
Status: frozen `M4_C3_NO_GO_ERROR_ATTRIBUTION`; M4-C.2 remains NO-GO

## 1. Why M4-C.3 exists

M4-C.2 V2 recovered individual physical actions but failed the composite loop:

- choice action geometry: `.7905`
- creation action geometry: `.7736`
- return geometry: `.8452`
- recovery geometry: `.9765`
- composite loop geometry: `.6519`, below the frozen `.70` gate
- complete repetition passes: `2/8`, below `6/8`

The rejected implication is:

\[
\text{each atomic action transports}
\;\not\Rightarrow\;
\text{their plug-in product transports}.
\]

M4-C.3 does not reopen or repair V2. It asks which physical edge or interaction
produced its composition loss.

## 2. Physical edges

For \(m=16\) physical conditions and a \(d=2\) response state, define:

\[
D_u\in\mathbb R^{m\times m},
\qquad
R_u\in\mathbb R^{d\times m},
\qquad
A_u\in\mathbb R^{m\times d}.
\]

\(D_u\) maps a physical menu intervention to a change in expected selected
condition. \(R_u\) maps a physical condition measure to response change.
\(A_u\) is the response derivative of the generated-opportunity hazard.

The loop is:

\[
L_u=A_uR_uD_u\in\mathbb R^{m\times m},
\qquad
\operatorname{rank}(L_u)\le d=2.
\]

The discovered chart may have rank 11 or 12, but the physical loop cannot
exceed rank 2. Additional chart directions can therefore increase estimation
variance without increasing the true loop's degrees of freedom.

## 3. Exact error expansion

Let:

\[
\widehat A=A+\Delta A,\qquad
\widehat R=R+\Delta R,\qquad
\widehat D=D+\Delta D.
\]

Then:

\[
\begin{aligned}
\widehat L-L
=\;&
\Delta A RD+A\Delta R D+AR\Delta D\\
&+\Delta A\Delta R D+\Delta A R\Delta D
+A\Delta R\Delta D\\
&+\Delta A\Delta R\Delta D.
\end{aligned}
\]

Atomic action metrics do not constrain these errors on the intermediate
subspace actually traversed by the loop.

For the fixed physical query bank \(Q\), the path-conditioned first-order
budgets are:

\[
e_A=
\frac{\|(\widehat A-A)RDQ\|_F}{\|ARDQ\|_F},
\]

\[
e_R=
\frac{\|A(\widehat R-R)DQ\|_F}{\|ARDQ\|_F},
\]

\[
e_D=
\frac{\|AR(\widehat D-D)Q\|_F}{\|ARDQ\|_F}.
\]

## 4. Eight mixed loops

Oracle and discovered edges are mixed in all \(2^3\) combinations:

\[
L_{ijk}=A_iR_jD_k,
\qquad
i,j,k\in\{O,D\}.
\]

The geometry loss of each mixed loop relative to \(L_{OOO}\) defines a
three-player cooperative game. Standard Shapley contrasts allocate the total
loop loss to creation, response, and choice while averaging over edge order.
The allocation is diagnostic; it does not make any edge causal.

## 5. Nonlinear finite intervention

The registered Jacobian loop uses a central derivative of the hazard. For
each menu intervention \(j\), M4-C.3 also computes:

\[
\mathcal L^{FD}_{u,\cdot j}
=
p_{\widehat\theta}
\left(
M^{gen}_{t+1}
\mid
\Delta x_u=R_uD_{u,\cdot j}
\right)
-
p_{\widehat\theta}
\left(
M^{gen}_{t+1}
\mid
\Delta x_u=0
\right).
\]

Comparing \(ARD\) with \(\mathcal L^{FD}\) separates edge-estimation loss from
local-linearization error.

## 6. Controls

The audit includes:

- exact reconstruction of the legacy M4-C.2 loop from physical edges;
- orthogonal chart-gauge invariance;
- one-at-a-time planted physical faults in \(A\), \(R\), and \(D\);
- a no-fault attribution control;
- rank-4, rank-8, and rank-12 paired diagnostics on the same physical worlds;
- independent train/test path views; and
- eight repetitions across five active creation worlds.

Rank comparisons are descriptive unless a paired mechanism controls every
other world property. The observed V2 correlation between transform rank and
loop geometry is not treated as causal evidence.

## 7. Frozen gates

- planted edge localization accuracy at least `.85`;
- bootstrap lower 95% bound for correct-edge attribution margin above `0`;
- null false attribution at most `.05`;
- Spearman correlation between total path error and observed loop loss at
  least `.75`;
- oracle Jacobian versus finite-intervention geometry at least `.90`;
- physical reconstruction error at most `1e-8`;
- basis-invariance maximum difference at most `1e-6`; and
- at least `6/8` repetitions pass the local diagnostic gates.

## 8. Frozen hashes

```text
2e28dfaaa3e640424e5605235114b4137666ba2570deb177a33722dc33feb2e3  configs/m4_physical_edge_audit.json
59b18b473ca37d5d56f27223b3a72aa6bfcafbd5c360275b7aeccb2a667016fc  scripts/run_suica_m4_physical_edge_audit.py
b0d6ce77ac379ed4f90bd98a1b4fb42f31bb88ea6b51ef82f38092d5bbebd96f  suica_core/m4_physical_edge_composition.py
90df68142e4dd86965aa60bf6eb6f585fae772d8af8de56429ec3f7fd96462ad  tests/test_m4_physical_edge_composition.py
```

## 9. Claim boundary

A successful audit can explain or localize finite synthetic loop loss. It
cannot:

- convert M4-C.2 V2 into a pass;
- establish that chart rank caused the loss;
- validate M4-D;
- identify a psychological construct;
- validate natural text, personality, emotion, diagnosis, or clinical use.

M4-D remains blocked regardless of the M4-C.3 diagnostic outcome.

## 10. Frozen result

The full eight-repetition discovery returned:

- exact legacy-loop reconstruction error: \(9.86\times10^{-15}\);
- orthogonal basis-invariance difference: \(4.67\times10^{-10}\);
- oracle Jacobian versus nonlinear finite intervention: `.9985`;
- discovered Jacobian versus nonlinear finite intervention: `.9983`;
- planted physical-edge localization: `1.000`;
- attribution-margin lower 95% bound: `.0293`;
- null false attribution: `0`;
- finite loop transport: `.6267`;
- Jacobian loop transport: `.6278`;
- old absolute-error-budget Spearman: `.7264 < .75`; and
- complete repetition passes: `4/8 < 6/8`.

The frozen decision is:

```text
M4_C3_NO_GO_ERROR_ATTRIBUTION
```

The Shapley allocation closes exactly:

\[
.2432_{\mathrm{creation}}
+.0769_{\mathrm{choice}}
+.0521_{\mathrm{response}}
=.3722
\approx 1-.6278.
\]

Creation receives about 65.3% of the symmetric relation-loss allocation,
choice 20.7%, and response 14.0%. This is not a causal attribution.

Rank-4, rank-8, and rank-12 loop geometry was `.382`, `.632`, and `.570`.
The nonmonotone pattern does not establish that high rank is harmful or that
rank 8 is optimal.

The failed budget averaged author-wise Frobenius errors, whereas the loop gate
compares author-to-author relation geometry. M4-C.3.1 therefore moves the
budget into centered pairwise-distance space. See
`docs/SUICA_M4_RELATIONAL_ERROR_BUDGET_THEORY.md`.
