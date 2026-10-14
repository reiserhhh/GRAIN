# HM14: Source-Only Ordered Observation and Action Compilation

2026-10-08. Planned after HM13, not executed. Finite mechanism development
on existing controlled states, not a fresh independent text replication.

## Decision and Scope

HM13 exact permutations preserve all unordered information but alter
native readouts. The next question is whether a small position-conditioned
observation can recover this missing action, rather than adding degree
to an order-blind observation. Preserve original owner, condition and query
units. Do not select a representation from application performance.

## Inputs and Fixed Arms

Use frozen HM10B3 block16 captures, source vector frame and source8/valid4/
application4 family roles. Consume actual HM10B3 full-property, HM12 mean/
centered and HM13 receiver/centered reversal outputs. No new model calls.
Check all consumed ledgers and prior code before producing a new packet.

Freeze a complete address list of reference/intervened states and their
actual readouts before fitting. Centered-reversal has its actual HM12
centered reference; keep its estimand distinct from receiver reversal.
Never pair the wrong query, receiver, arm or donor positions.

Compare four predefined action models at each original query:

1. Zero action, no fitted coefficients.
2. Original six-span source-frame means, linear action.
3. Means plus first position-weighted moment of both property spans.
4. Means plus moments1-3 of both property spans, since current largest
   declared span has4 tokens; if source metadata disagrees, fail instead
   of selecting a new order from outputs.

Retain the original rank2 vector frame; no rank search. Fit explicit
linear action to observation differences, not a new neural encoder.
Source-only feature scales, alpha=1, no intercept for action so identical
observations produce exactly0. Family-balanced loss and uncertainty.
Direct action regression is conditional on the available source states;
it is not automatically an exact derivative/operator law.

## Mathematics and Checks

Use fixed symmetric positions within each declared span. Test m_k(RH)=
(-1)^k m_k(H), Vandermonde reconstruction at all source-supported small
lengths, single-token odd-channel0, coordinate shapes, finite values and
unchanged original frame. Keep full-dimensional reconstruction distinct
from projected rank2 reconstruction.

Verify source-only fitting with an independent linear algebra solve.
For the old untagged baseline, exact reversals should change no observation
except bounded summation roundoff; record actual input differences. For
ordered models report scalar action signs, RMSE and correlation separately.
Level correlation is not an acceptance substitute. Pair every predicted
contrast with its original categorical output unit.

## Deliverables and Branch

New helper/runner/config/tests and before-fit manifest; explicit paired
state addresses, source coefficients and source-only scales; all roles/
queries/arms/token counts with family intervals; alternate numerical
checks, frozen outputs and parent checkpoint/report. Preserve old files.

If ordered channels recover placement but not owner-donor action, separate
placement from binding and inspect context-by-state response. If both
remain poor, inspect source-vector complement or fixed next-block Q/K/V
action, not degree/alpha grids. If useful, apply to separately frozen new
materials before any author-level psychological naming or external claims.

No API/private MEPS/new neural training/Claude/automation/commit/publish/
lockbox/old-queue recovery. Original18/100/44 remains open and inherited.
