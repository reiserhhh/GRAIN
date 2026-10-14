# HM71: Cold Relational Measurement on New Expressions

2026-10-09. Freeze before the first new observer call. Parent goal ACTIVE.
Question: do source-calibrated within-event relations remain useful on new
expressions of the declared events, or was HM70 only full-support interpolation?
No new predictor, calibration, rank/degree sweep or teacher-fidelity repair.

## Design and Limits

Use all six fixed targets and all 30 directed pairs. Eight cells are the HM68
declared (focal behavior x, companion behavior y, ownership g) cells. HM69's
eight paired profile cases and all 60 native-score fibers remain fixed.
These are designed probability distributions, not new human participants.

New wording is manually authored for the SAME family-0 acts, not different
family-1 acts. Each account still has K1 focal Person A, K2 companion owned by
A or B, and K3 notebook action owned by the other person. Two complete forms:

1. `paraphrase`: new wording, K1/K2/K3 presentation order.
2. `reordered`: identical new sentences in K3/K2/K1 presentation order.

No historical dates are inferred. Reordering is a controlled presentation
change, not trait change. Event equivalence is design-declared and readable in
the frozen source-to-new event map; no independent semantic equivalence claim.
All 480 bodies and 2880 target readings are acquired, including unfavorable
pairs. There is no neutral-arm acquisition or optional favorable-target subset.

Local offline Qwen2.5-1.5B, original system/query/alphabet/template, block16,
full untruncated passive forwards. No generation, interventions, neural training,
private raw/questionnaires, API, Claude, automations, Git or lockbox changes.
Bound 2880 forwards and 1200 collector wall seconds; persist each attempted unit
and actual result. Stop on uncertainty/error; never silently repeat a forward.

## Frozen Measurement

Load HM70's six-stage local charts. For every stage and pair, freeze C0, Z and
the original quadratic inverse BEFORE collection. For new observation Fnew:

    Znew = [1,Fnew] C0
    m_u = P_u^T Znew
    S_u = sum_e P_u(e) Znew(e) Znew(e)^T

No recentering, unit adjustment or calibration using the new table. Also keep
native Fnew and decoded log odds, centered logits, body endpoint and both
existing no-cross-logit controls. Exact finite rank of new tables is not a
primary success claim, and the old inverse is not refitted.

For every paired profile, store full native difference, mean difference,
same-event second-moment difference, product-of-means difference, and fixed
old-inverse readback. Compare source/cold direction and magnitude separately.
For score fibers, report both source-zero means and new means: a nonzero new
mean is observation-law change, not proof of a uniquely relational signal.

The coordinate-invariant quadratic distance is

    D2(P,Q) = ||Delta m||^2 + ||Delta S||_F^2
            = (P-Q)^T [G + G*G] (P-Q), G=Z Z^T.

This is a standard degree-two polynomial kernel, NOT globally characteristic.
Full symmetric matrices rather than unweighted upper triangles preserve the
orthogonal coordinate invariance. It is not a personality validity statistic.

Controls: independently paired events use m_u m_u^T rather than E[ZZ^T];
incorrect event correspondence uses a fixed one-cell cyclic shift of Znew.
Keep both controls even if they accidentally align. No p-value treats the
designed pairs as independently sampled people.

## Mechanism versus Calibration

Let epsilon=max_e ||Znew(e)-Z(e)|| and R=max_e ||Z(e)||. For a profile contrast
d=P-Q, the standard tensor perturbation bound is

    ||Delta m_new-Delta m|| <= ||d||_1 epsilon
    ||Delta S_new-Delta S||_F <= ||d||_1 epsilon(2R+epsilon).

Record this conservative source-to-cold certificate, actual discrepancies and
calibration coefficient norms. A failed bound is not disproof of transport;
a large cold distortion is not cured by searching a new chart. The principal
scientific decision is whether source meaning/coordinates transport or only
the existence of relational sensitivity transports. Preserve the distinction.

## Outputs and Acceptance

All raw captures, token/role/order identities, frozen code/source hashes,
full six-stage new observations and moment arrays, complete 2880 profile rows,
120 score-fiber rows, 360 pair/stage/form transport rows, summary/checkpoint.
Numerical verifier independently replays raw decoding, scalar event products,
kernel identity, control pairing, source/cold bounds, CSV summaries and seals.
Unit tests cover event identity, reorder-only text/token bags, source-only
coordinates, independent pairing, rotation invariance, bounds and validation.

Report a mechanism result rather than a PASS accuracy leaderboard. A positive
finding must specify whether exact numerical calibration, profile direction,
relational discrimination or only finite sensitivity survived. Psychological
meaning, personal response Bu, automatic natural-event identity, independently
scored unseen text, and original18/100/44 remain open.

## Reading and Attribution

Gretton et al.2012, [A Kernel Two-Sample Test](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf):
actually read Section2, definitions2 and Lemma4/6, Theorem5 and the degree-two
non-characteristic counterexample (PDF pp3-6). Kernel distance is prior art;
we use exact designed profile expectations, NOT its iid hypothesis-test theorem.
Sriperumbudur et al.2011 characteristic-embedding definitions retained from HM70.

Pinned OpenAI/math adc7f124 metric/connection introduction, all155lines reread:
common observed forms versus gauge coordinates inspires the question. No PDE,
smooth manifold, boundary continuation or uniqueness theorem transfers here.
The cold observation experiment, not the manuscript title, supplies evidence.
