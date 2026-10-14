# Source-defined evidence actions and reading-framework contacts

HM214, 2026-10-12. Post-outcome theory account, not a new preregistration.
The immutable design is [HM214](GRAIN_DIAGNOSTIC_ACTION_BRIDGE_PLAN_HM214.md).

## Finding and construction advance

A numeric scorer constructed from frozen SOURCE observations can now return
five finite evidence-edit actions on the SAME events used by the neural
mechanism probe. It uses no model answers or current-author labels. Its A
(Agreeableness-anchor) action agrees positively with the cached
recipient-framework contrast in both SOURCE banks: centered action cosines
0.697158 and 0.516715. This connects two previously separate interfaces:
SOURCE-supported score construction and actual evidence/framework intervention.
HM204/209 already provided cold numeric scores; this is not the first
LLM-free downstream scorer or a new Bayes theorem.

The remaining four targets mix signs or have weak contacts. Moreover the two
A SOURCE action arrays agree with each other only at 0.292546. The bridge
therefore does not turn source dependence into a universal trait direction.
All ten primary contacts, off-diagonals and negative accounts are retained.

## Observation and reference definition

There are 386 historical SOURCE authors, split into existing banks 203/183,
and 24 SOURCE-disjoint historical panel authors. Each has two 12-comment
windows represented by full 1024-dimensional E5 vectors. SOURCE binary
E/N/A/C/O assignments define all 32 inherited profiles. Panel questionnaire
labels and raw text are not read. Conditions and event identities remain
provenance; this does not separately identify topic or state effects.

The reference is uniform over profiles and uniform over SOURCE authors within
each profile. It is not the empirical population prevalence. The fixed
variance is the inherited label-free calibration s^2=0.05525111588037168;
it was not optimized for this result. Historical representation, cohorts,
all-SOURCE calibration and N-guided event selection are reused. This is an
exploratory construction, not an untouched test or a new nested-CV benchmark.

For SOURCE author i in profile g, let z_iA and z_iB be window means, and
K_s(x,z)=exp(-||x-z||^2/(2s^2)). The three fixed observation protocols are

    L_single,g(x_v) = mean_(i:g) K_s(x_v,z_iv)
    L_pair,g(a,b) = mean_(i:g) K_s(a,z_iA) K_s(b,z_iB)
    L_product,g(a,b) = mean_(i:g) K_s(a,z_iA) mean_(j:g) K_s(b,z_jB).

These are declared Gaussian channels, not recovered natural-language noise
laws. Gaussian constants common to every profile cancel in the odds; the
implemented potentials alone are not normalized standalone densities.
The single protocol integrates out the companion and matches the cached
native query's single-window evidence availability. The paired protocols are
additional measurement definitions: the native model did not see a companion.

## Finite edit and analytic field

For target q and protocol L define

    S_L,q(x) = log sum_(g:q=1) L_g(x) - log sum_(g:q=0) L_g(x)
    D_L,q = S_L,q(remove_selected) - S_L,q(remove_control).

Both removals address four distinct observed events within the same window.
The two sets do not overlap. The other full window is unchanged. Duplicate,
foreign or missing events are rejected rather than imputed as neutral data.
The action orientation is exactly the inherited HM199 operation.

For the single protocol its analytic gradient is

    F_q(x) = [E(z_v | x,q=1) - E(z_v | x,q=0)] / s^2.
    delta_x = x_remove_selected - x_remove_control
    m = (x_remove_selected + x_remove_control) / 2
    D_single,q = F_q(m) dot delta_x + remainder_q.

The midpoint is a numeric counterfactual coordinate, not a generated text or
human intervention. This derivative identity is standard Gaussian-mixture
calculus. On these edits the remainder RMS / action RMS is only
0.000166538-0.000532034 (0.01665%-0.05320%). It is not useful to pursue a
curvature, integration-step or floating-precision grid on this channel.

## SAME-event contacts

The primary comparator is the inherited recipient-framework contrast
foreign_increment minus foreign_wholeV, in the query-specific geometric-head
account, after averaging the two views for each author. The statistical unit
is a panel author, not a comment, head, CSV row or SOURCE-bank contact.

| Target | Bank 0 covariance | Bank 1 covariance | Bank 0 centered action cosine | Bank 1 centered action cosine |
| --- | ---: | ---: | ---: | ---: |
| E | -0.000324197 | +0.000218049 | -0.359479 | +0.159712 |
| N | +0.000661446 | -0.000150443 | +0.411612 | -0.094257 |
| A | +0.001002430 | +0.000898666 | +0.697158 | +0.516715 |
| C | +0.000081904 | +0.000015463 | +0.223817 | +0.059759 |
| O | -0.000082237 | +0.000089167 | -0.107235 | +0.101734 |

Centered one-dimensional action cosine is mathematically Pearson correlation
of the action arrays. It is NOT questionnaire-prediction Pearson r. Covariance
is a product of SOURCE log-odds and the native readout unit, not a standardized
psychological effect. No new population p-value is calculated. Both banks
read the SAME panel24 and are not independent replications or 48 people.

SOURCE-bank action cosines, without the native readout, are E -0.421925,
N -0.374236, A +0.292546, C -0.598724, O +0.440765. In particular a positive
A contact in both banks does not make its entire SOURCE operator stable.

As retained secondary accounts, A single-protocol contact with native is
+0.001045631/+0.000925508; foreign whole-V is
-0.000769145/-0.000555824. Zero-increment contact remains positive, so there
are other retained evidence routes. None of these selects a unique mediator.
All six modes, six contrasts, three accounts, four units, three view readings
and 25 cross-target comparisons are saved (64,800 contact values, not
independent experiments). The paired-view CSV contains 21,600 rows.

## Observation protocol is part of the measurement

Paired-versus-single action RMS is 10.51%-20.59% of single-action RMS;
paired-versus-product is 7.99%-18.89% on the same denominator. These are
protocol-relative score changes, not psychological-information fractions.
The paired scorer conditions on a companion that the native model never saw.
No protocol is chosen by its best native agreement.

A candidate measurement object must therefore declare at least

    (evidence/event identities, SOURCE reference, observation protocol,
     target reading, finite operation).

The neural probe is a mechanism comparator, not questionnaire truth or the
teacher used to fit this scorer. The downstream action runs mathematically
without LLM answer inference; upstream E5 vector production remains neural.
Choice, context, state, dynamics and memory remain in the parent framework.

## Mathematical ancestry and novelty boundary

Finite-mixture Bayes, gradients and covariance are mature methods.
[Kernel Bayes' Rule (2013)](https://jmlr.org/papers/v14/fukumizu13a.html)
is related source-to-posterior literature; its RKHS inverse estimator and
consistency theorem are not implemented or inherited here.
[Activation-patching guidance (2024)](https://arxiv.org/html/2404.15255v1)
already distinguishes mediation from calibration and redundant routes.

Pinned [OpenAI/math](https://github.com/openai/math) revision adc7f124, the
posterior-replicas manuscript's `08-streaming.tex` lines 1-120, supplied the
design reminder to condition on the present observation state before applying
the next law. Its spherical signals, independent Gaussian rows and memory
lower bound are not assumptions about these texts. No theorem is transferred
to personality measurement. The current contribution is an executable
same-evidence bridge and its observed scope, not new high-mathematics proofs.

## Reproduction and next decision

Canonical artifacts: `results/grain_math_response_v1/diagnostic_action_bridge_r1`.
Implementation: `math_diagnostic_action_bridge.py`; fresh-process scorer:
`scripts/score_grain_diagnostic_action_cold.py`. It accepts only the frozen
SOURCE bank and addressed vector/edit JSON, not native output or panel labels.
The executed prepare/compute/verify/cold stages each ran once. Literal direct
Gaussian distances plus inherited HM209 paired actions checked 2,410 values,
max error 1.043999955929742e-14. The fresh-process cold action agrees exactly.
Ten new tests plus HM209/HM213 tests pass (33 total); audits are same-agent,
not blind third-party replications. All scientific files and outputs stay sealed.

Next inventory HM191/193/195/208/210/211 source-condition/author/observation
interfaces before a new finite design. Specify what source-qualified evidence
unit or observation rule can transfer, and its sparse-profile/fingerprint/
context competitor. Do not repeat Gaussian curvature, posterior means,
head/layer/scale/rank/accuracy grids. Original18/100/44 remain OPEN; this packet
is complete, goal ACTIVE, with no running job or automatic next packet here.
