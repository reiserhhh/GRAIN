# HM56: Event Aggregation and Query-Specific Interpretation

2026-10-09. Finite mechanism protocol, frozen before new observer calls.

## Missing question, not a replay of HM46

HM46 already measured six content families against six queries. HM47 tested
appraisal modulation. HM55 separated event identity from narrative mentions,
but each account had only its own target's readout. The missing question is
whether one query-independent weighting of current events can reproduce the
whole six-query judgment of an account. More accurate teacher approximation
is not the criterion; no predictor, prompt, layer, rank or weight search occurs.

This tests a specific score-level aggregation model, not every possible GRAIN
pooler, Deep Sets model, hidden-state aggregation or psychological theory.

## Fixed design and cost

Reuse HM55's 24 isolated events and 192 accounts from balanced, alias_focus,
distinct_focus and other_focus. Preserve both behavior families, both
polarities, both presentation orders and all six content families. Query every
body with the same six targets (Big Five plus anthropomorphism). Bodies and
questions are unchanged. Content and query are separate software fields.

216 bodies x 6 queries = 1296 readouts; reuse 216 exact original inputs, acquire
1080 new forwards, at most 900 model seconds. No private data, generation,
API, fitting, neural training, patching or new author labels. Same frozen
Qwen2.5-1.5B, standard digit code, float32/MPS/eager/block16. Actual token
prefixes and 288 alias/distinct pairs are checked before acquisition. New
captures retain the endpoint and logits, not all intermediate tokens; this
experiment does not claim causal localization. Parent full states are retained.

## Observation and null model

For each isolated event e, collect the vector a(e) of six conditional digit
expectations in [0,1]. The whole account yields L(H), in the same six output
coordinates. The existing versioned ledger gives current own events and the
distinct descriptive mention profile; other-person behavior remains context,
not the named person's own action.

Every selected account has exactly two distinct own behavior contents, x and y.
Repeated occasions can change their weights, but not the set of these contents.
Any normalized, nonnegative, query-independent score-level weighting requires

    L(H) = t*x + (1-t)*y,  0 <= t <= 1.

Its feasible region is a line segment, not an estimated personality manifold.
Allow any t per account, including one chosen after seeing all six outputs.
Failure even with that permissive choice rules out this particular common-weight
representation on the finite inputs. Negative shared weights give the affine
line; six independent positive weights give the coordinate box. Measure all
three separately so the reason for failure is explicit.

Let d=x-y. The exact Euclidean segment projection is

    t0 = dot(L-y,d)/dot(d,d), t=clip(t0,0,1), p=y+t*d,
    w=L-p, gap=dot(w,L)-max(dot(w,x),dot(w,y)).

For the projection, gap=||w||^2. A positive gap is a separating witness for all
shared positive weights, not just failure of one fitted weight. Degenerate
x=y is handled as a singleton. Projection distance describes these fixed
readout units; it is not personality accuracy, a population p value or an
information fraction. Numerical equality uses 1e-10, not a promotion threshold.

For alias, distinct and other additions, separately project the observed
six-query update onto d. A common-weight-only update must be parallel to d;
retain the perpendicular component and all signs. Do not call this component
noise or select a new weight to erase it. Event and mention profile changes
are replayed exactly from the original ledger.

## Interpretation and advancement

If native points lie on the segment, a common weight is viable on these
materials, but neither unique nor established as a human psychological rule.
If points leave the segment but stay in the box, query-specific weighting can
explain what a common event weight cannot. If points leave the box, even such
per-query normalized positive score mixtures fail; contextual interpretation,
output calibration or a non-score-level operation is required.

Advance the author-measurement interface by separating event evidence profiles
from the context-sensitive observation operator. Keep event selection, framing,
order and other-person context available. Do not equate matching the observer
with measuring the author better or reject mean-embedding/Deep Sets theories.

## Prior mathematics and exact reading scope

- Boyd and Vandenberghe, Convex Optimization (2004), section 2.5.1 and projection
  material in section 8.1: standard separation/projection, not our new theorem.
  https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
- Muandet et al., Kernel Mean Embedding of Distributions (2017), section 3.1,
  Definition 3.1.1, Lemma 3.1.2 and its proof: a linear functional of a mixture
  has a common mixture rule. No characteristic-kernel guarantee is imported
  for these six observer coordinates.
  https://arxiv.org/html/1605.09522v4
- Zaheer et al., Deep Sets (2017), section 2.2 Theorem 2: nonlinear rho after
  pooled phi is allowed. Our convex score-level null is a narrower model and
  its failure cannot refute that result.
  https://arxiv.org/html/1703.06114v3
- OpenAI/math fixed commit adc7f1241b42e322a6451854ab7e4b4c146bf78a,
  Finite Monoid Computations, algebra.tex lines 1-140 reread: distinguish
  deterministic episode updates from output tests and respect composition
  order. Finite-monoid/F2 assumptions are not proved for the LLM. The main
  star-height theorem and other chapters are not used here.

## Acceptance and failure

Complete declared crossings, exact reused provenance, frozen parent/input/code,
all logits and endpoint captures, every scalar/profile/dual-witness replay,
unit tests with both an exact mixture world and a failing nonlinear world.
Preserve failures and STOPPED/in-flight state; never retry an uncertain forward
automatically. A full collector is not proof of the global research goal.
Original 18 protocols/100 cards/44 computational questions, natural author B_u,
psychological meaning and independent LLM-free operation stay open.
