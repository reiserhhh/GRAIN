# GRAIN: reading relative to a material's evidence coordinates

2026-09-11. NEXT85 executed; [empirical report](../reports/grain_vector_mechanisms/EVIDENCE_COORDINATE_READER_REPORT_V1.md).
This extends the same-text GRAIN/query interface, not the questionnaire ontology.

## 1. Observation, coordinate and readout are different objects

Let x denote the frozen H16 material input, d(x) the differential reader between
the two queries, d0 its source origin, and db(x) the preceding fitted reader.
Construct a five-row R(x) from the same compiled program's common evidence and
four total GRAIN response vectors. Normalize rows only by original-source RMS.
The desired observable coordinates are

```text
c(x) = R(x) [d(x) - d0]
Dc[D] = DR[D] [d(x) - d0] + R(x) Dd[D].
```

R is either source-fixed or material-dependent. The product term is required
in the latter case. BODY targets use the clamped material at which their native
derivatives were captured; context targets use original states. A token/slice
address alone is insufficient: state, query and tangent must also match.

This operationalizes the user's first idea: the same author material can be
viewed through GRAIN directions and query-dependent reading. Their relation is
a conditional action c, not an arbitrary sum of two global scores.

## 2. Exact minimum-change lift

For full row rank, Rplus=R^T(R R^T)^-1 and the unique minimum Euclidean correction
satisfying R(dtilde-d0)=chat is

```text
dtilde = db + Rplus [chat - R(db-d0)].
```

The component of db in ker(R) is preserved, not deleted as noise. Write
A=(R R^T)^-1 and e=chat-R(db-d0). Its directional derivative is

```text
D Rplus = (DR)^T A - Rplus [(DR)R^T + R(DR)^T] A
D dtilde = Ddb + (D Rplus)e
           + Rplus [Dchat - (DR)(db-d0) - R Ddb].
```

Fixed-frame and moving-frame fits are different functions, not alternative
names for the same axes. Exact constraint satisfaction establishes the lift,
not equality of chat to native c. Vector-valued RKHS methods provide precedent
for learning vector functions; this particular frame/lift is our construction.
See [Alvarez, Rosasco & Lawrence (2012)](https://arxiv.org/abs/1106.6251).

## 3. Teaching positions and teaching motion must be connected

The Gaussian derivative kernel gives a joint matrix K for source values, BODY
and context directional derivatives. With frozen group weights W and lambda,

```text
beta = W (W K W + lambda I)^-1 W Y
chat(x) = k(x)^T beta.
```

NEXT85 source losses are tiny but new8 scalar-coordinate RMS is only5.6%-6.5%
of target; action RMS in the moving frame is2.5%-4.9%. The exact group sum has
large opposing pieces. Therefore good source interpolation plus available
output directions does not determine a useful off-source extension.

An alternative account is **conditional teaching innovation**. Put
A=K+lambda W^-2. Given old observations I and added group J, block conditioning
implies

```text
S = A_JJ - A_JI A_II^-1 A_IJ
e_J = Y_J - A_JI A_II^-1 Y_I
delta c(x) = [k_J(x) - k_I(x) A_II^-1 A_IJ] S^-1 e_J
c_(I+J)(x) = c_I(x) + delta c(x).
```

For disjoint groups A_JI=K_JI. This follows from block elimination. It yields
the same full predictor regardless of assimilation order, but its intermediate
increments are order-dependent. The derivative of each increment is obtained
by differentiating k, hence remains a derivative of a single function. We use
the predetermined order value -> BODY -> context; not an outcome-selected order.

This is a standard conditional Gaussian/kernel construction, not a new theorem.
Joint function/derivative observations and distinct observation noise scales
are described in [Rasmussen & Williams (2006), section9.4](https://gaussianprocess.org/gpml/chapters/RW9.pdf).
Here lambda W^-2 is an algebraic regularization equivalent, not an empirically
calibrated noise model. Kernel Schur remainders are not human confidence limits.

## 4. What the experiment changes

It supplies a coherent moving-coordinate numerical interface and locates
amplitude compression in a jointly taught function. It does not establish that
context is useless or that more authors alone will fix it. Finite source
coverage, target scale shift, and group cancellation coexist. NEXT86 isolates
the added functions before paying for more native source captures.

The user's second idea remains an interpretation interface: slice-addressed
GRAIN/query effects can accompany independently coded relational behaviors.
A vector coordinate is not itself a psychoanalytic or transactional-analysis
construct. Such interpretations use declared coding rules and supporting text
addresses downstream, not clinical names attached to these five numerical rows.
