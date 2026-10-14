# F16 evidence history and target-specific action

## Result and decision

The numerical part of F16-R02 is implemented and executed: eight actual evidence
cards for each of T1/T2/T3, with original material addresses, before/after frozen
scores, support, intervals and interpretation/action. A versioned ledger binds
the same observation law to explicit eligibility, alias, summary, revision and
retraction operations. It preserves both event time and report availability and
uses the already-verified target-clock adapter. Old frozen scorers and norms are
unchanged. This is not merely a24-file/count check.

The useful production finding is **target-relative sensitivity**. A balanced
recent offset can almost cancel in a conditional contrast while affecting a
current-state or future-expression score. Thus the same report change cannot be
declared universally irrelevant noise. This observation is explained below by
ordinary Gaussian conditioning, not presented as a newly invented theorem.

Scope: SIM numeric observations and supplied eligibility metadata. No natural
language classifier inferred these qualifications. F08 text-law matching,
F14 corpus/target support, F12 matched transport, public construction and canonical
card/dependency acceptance remain open. Full goal ACTIVE; F16 parent not promoted.

## Fixed design and implementation

Protocol before application: `configs/grain_f16_evidence_v1.json`. Seed20260915,
original first/last application authors per target, source ensemble200 and
personal integration200; no source refit. T1 uses its original32-event F04L
four-cell panel; T2/T3 use the same48-event F05core panel. There are **four distinct
simulated authors**, not24 independent people. Thirty numeric objects are the
six original target/author baselines plus24 after-operation objects.

Each target has two full-support, two missing-condition, two state/selection and
two ineligible-evidence cards. The cases are fixed in the config:

| Case | Actual operation |
|---|---|
| full_plain | Original observations, no edit |
| full_history | Two aliases and a two-record summary; final report corrected by+.25 |
| missing_plus / missing_minus | Explicitly withdraw all +1 / -1 reports |
| selection_high | Withdraw y<0, retaining original opportunity addresses |
| late_offset | Revise each last4 value by+1 as an observation-channel edit |
| quoted_tail | Reclassify last4 as quoted/non-self; omit from self likelihood |
| hypothetical_all | Reclassify all as hypothetical; use explicitly prior-only output |

`negated` follows F08's designed *explicitly corrected actual report* meaning:
the supplied corrected numeric value is eligible and is not sign-flipped. It is
not a rule that arbitrary negative sentences constitute observed behavior.
`unknown` remains unobserved, not a zero. Aliases/summaries add no likelihood;
their references become stale if their parent is replaced. New independent
noisy reports of an already observed event are not aliases: they require a
report-noise model and are rejected rather than counted as independent events.

Revisions preserve event identity, condition and time. Availability after the
requested cutoff is rejected. Reports have the measured author's ID, not an
assertion that the subject of every quotation is that author; supplied
qualification specifies whether the numeric payload is eligible for self-scoring.
Event matching and qualification are metadata inputs, not discovered semantics.

## Observed finite effects

Raw score changes relative to the same author's full numeric prefix:

| Operation | T1 conditional contrast | T2 current state | T3 future mean |
|---|---:|---:|---:|
| Unedited | 0 | 0 | 0 |
| Aliases/summary plus final+.25 correction | +.027380 | +.149810 | +.018878 |
| Withdraw +1 condition | +.067101 | +.832957 | -.272184 |
| Withdraw -1 condition | +.039419 | -.003467 | +.073042 |
| Keep y>=0 | 0 | +.767139 | +.801359 |
| Last4 values+1 | +.000397 | +.593504 | +.171856 |
| Last4 reclassified quoted | -.058910 | +.835850 | +.186380 |
| All hypothetical | -1.359611 | +.111917 | +2.412865 |

These are actual finite changes in declared simulation response units, not
accuracy gains or population effects. T1 and T2/T3 use different worlds/authors;
the magnitudes are not a controlled cross-method superiority comparison. Removing
one condition changes information and source borrowing; it does not identify the
cause of a person's underlying response. Target intervals appear on each card;
they are not independent-error confidence intervals for these paired edits.

The T1 y>=0 case retained32/32 observations: all were already nonnegative. Its
zero effect is deliberately retained. This card establishes the no-op boundary,
not an active T1 selection-bias effect; thresholds/authors were not changed after
seeing this. T2/T3 retain22/48 under the same rule. The resulting working scores
do not repair outcome-dependent selection. No hidden truth was loaded to assess
bias or improved prediction here.

All-hypothetical results are source priors, not measured neutral people. Their
large changes illustrate lost self evidence, not genuine psychological change.
Quoted-tail T2/T3 still refer to cutoff47 and future48..51, not retained endpoint43.

## Exact explanation of the T1 cancellation

This section is a separately saved **post-result algebraic explanation**, not a
predeclared new empirical test. For fixed eligible rows, design X, source model b,
prior mean mu_b/covariance Sigma_b and observation noise R_b, define

`K_b = X Sigma_b X' + R_b`

`w_b = K_b^(-1) X Sigma_b c`, with `c=(-1,0,0,1)'`

`a_b = c' mu_b - w_b' X mu_b`.

The frozen contrast score is exactly `mean_b(a_b) + mean_b(w_b)' y`. Therefore
an edit d on the **same observations/support** changes the score by
`mean_b(w_b)' d`. The four last records here span all four conditions. Their
absolute weighted contributions sum to .280596, but signed contributions sum
to .000397: cancellation fraction .998584 (99.86%). The final single-condition
correction+.25 yields .027380. All200 source scores before/after are reconstructed
within3.55e-15.

This supplies an inspectable local influence rule for this target: how common
versus differential edits act depends on the target contrast and observation
design. It does not establish cancellation for every context/panel, invariance
of uncertainty, or a general linear attribution for T2/T3. With deletions X/K
change, so fixed-support weights cannot be reused. T2/T3 also refit personal
parameters from the retained history under the frozen source law. A channel
offset versus real state change requires additional identifying information;
these numeric cards do not separate them simply because a score changed.

## Verification and retained failure

| Check | Result |
|---|---:|
| Actual cards, original categories | 24, 8 per target |
| Complete numeric outputs identical in cold process | 30/30 |
| T1 independent full-observation covariance solves | 2,000 |
| Maximum T1 conditional mean/variance discrepancy | 8.44e-15 |
| Dynamic personal draws checked against dense covariance | 12,000 |
| Maximum dynamic discrepancy | 6.66e-16 |
| All-source saved mixture/interval resummation maximum error | 2.89e-15 |
| Archived full-input score/variance/reference discrepancy | 0 |
| Input/run files hashed by audit | 109 |
| Tests, final suite | 131 tests and11 subtests passed |

Tests cover copy invariance, replacement/restoration, withdrawal, prior-only,
invalid ancestry/times/roles, unauthorized truth fields, all eight case definitions,
the response kernel, and detecting a fabricated score despite correct counts.
All new Python files pass Ruff. Dynamic covariance checks use predetermined
source indices0/100/199 and all200 parameter draws; this is not an independent
MAP/Hessian refit. T1 uses all200 source models. Same-agent alternate-formula
checks are not independent-human H06 reconstruction.

First smoke r1 stopped because T3's archived bundle hash was in
`protocol_before_application.json`, unlike the T1/T2 application manifest.
The interface was corrected, preserving r1; smoke r2 then passed6 cards and12
cold objects, including800 T1 and4,800 dynamic checks. Full target cells took
about0.95/17.58/9.86 seconds with at most3 CPU workers. All jobs are terminal.
No neural training/API, remote execution, private raw/MEPS/labels, source fit,
lockbox, commit or publication was used. Previous failed calibration cells remain.

## Artifacts and next execution

- Ledger: `grain_research/gptpro_foundation_v3/f16_evidence_ledger.py`.
- Runner: `scripts/run_grain_gptpro_f16_evidence.py`.
- Audit: `scripts/audit_grain_gptpro_f16_evidence.py`.
- Weight explanation: `scripts/analyze_grain_f16_evidence_actions.py`.
- Test: `tests/test_grain_gptpro_f16_evidence_ledger.py`.
- Canonical cards: `results/grain_gptpro_foundation_v3/F16/evidence_full_r1/`.
- Audit and reconstructed effect table: `F16/evidence_audit_r1/` under the same root.
- Algebraic companion: `F16/evidence_action_explanation_r1/`.
- Final test XML: `F16/evidence_tests_r2.xml`.

From this repository (use new output paths; no rerun is necessary for acceptance):

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. '../project persona/.venv/bin/python' scripts/run_grain_gptpro_f16_evidence.py --output results/grain_gptpro_foundation_v3/F16/evidence_full_r2
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. '../project persona/.venv/bin/python' scripts/audit_grain_gptpro_f16_evidence.py --run results/grain_gptpro_foundation_v3/F16/evidence_full_r2 --output results/grain_gptpro_foundation_v3/F16/evidence_audit_r2
```

`--smoke` runs6 cards with the full source/personal integration budget. `--resume`
requires unchanged config/code/input hashes; do not overwrite failed/frozen runs.

Next is the already-prioritized F14-R01 support/observation-law inventory and
target-compatible textual qualification/history integration. A binary F08 pole
is not a continuous Gaussian state. Retain this completed numerical interface,
then construct or explicitly delimit the lawful text observation map. Follow
with F16-R03 matched independent-pilot transport and public build/schema, plus
remaining F01/F15/F10/original-protocol gaps and F17 whole44+6 acceptance.
