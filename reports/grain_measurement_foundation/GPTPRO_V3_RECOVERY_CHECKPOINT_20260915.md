# GPT Pro V3 Recovery: Execution and First Restored Mechanisms

Date: 2026-09-15. Goal remains **ACTIVE**. This is a checkpoint, not full-plan acceptance.

## Scope and Actual Work

The authorized scope remains the original 100 atomic cards, 44 computational
requirements, and computational preparation for six human-validation designs.
The previous 100/100 declaration remains withdrawn. No parent card is promoted
solely because the repairs below pass tests.

Implemented and executed:

- Requirement-level acceptance rejects an empty ledger and 100 self-reported PASS
  cards. A source crosswalk, explicit scientific decisions, task coverage, review
  rationale, and hash-bound evidence are required. This checks provenance and
  explicit review; it cannot itself determine whether a theorem is correct.
- Revision-aware dependency requests check the frozen protocol, exact DAG
  dependency IDs, manifests, file hashes, and explicit dependency acceptance.
  F17 CLI now uses this loader and does not manufacture its input manifest.
  Adoption in the other historical CLIs is still incomplete.
- F00 null time is tested as an accepted record with a distinct event ID. Role
  conflict is no longer accidentally tested as a duplicate-ID conflict.
- F15's LP actually enforces `B = A K`, `K >= 0`, and `K 1 = 1`.
  An infeasible order returns a checked Farkas vector, not just a solver message.
- F02's C3 generator now contains both declared `(pi,q)` pairs, not just one.
  C1 types are balanced, and roles and author/material/noise/render streams use
  the specified SeedSequence structure.
- The full F02 numerical battery runs C1, C2-XOR, C2-IID and C3, across all three
  seeds, with 300 source / 100 validation / 200 application instances per world
  and 48 events per instance. Source fitting, prefix inference and truth scoring
  are separate subprocesses. The first 12 observations initialize each person;
  the next 36 are sequentially predicted before they are appended.

No API calls, new neural training, private raw text, lockbox operation, commit or
push occurred. Counts below are simulated instances, not human participants.

## Restored Scientific Results

### Conditions Determine Whether Personalization Helps

Application Brier losses, equally averaged over three seeds (lower is better):

| World | Shared source rule | Personal conditional/B2 | Personal mean/M | Oracle, scoring only |
|---|---:|---:|---:|---:|
| C1: individual condition responses | 0.250025 | 0.199210 | 0.258340 | 0.187477 |
| C2-XOR: shared second-order rule | 0.189487 | 0.210892 | 0.201204 | 0.189444 |
| C2-IID: shared random process | 0.250149 | 0.278779 | 0.261010 | 0.250000 |

In C1 the personal conditional estimate improves Brier by **-0.050816**,
paired-author bootstrap 95% CI **[-0.053641, -0.048087]** (600 application
instances, three seeds, 2,000 resamples). Ignoring the condition and estimating
a personal overall mean instead worsens loss by +0.008315.

In homogeneous C2 worlds there is no simulated personal transition difference
to recover. B2 and M fit finite-prefix variation and perform worse than the shared
source table. This is a production-rule result: distinguish a common process
from personal response variation before adding individual parameters.

### One-Step Similarity Is Not Path Equivalence

The XOR histories `00` and `11` have identical next-observation laws (TV=0),
but their two-step future laws differ by TV=0.5; the difference remains 0.5
through the enumerated horizon six. All 96 history/horizon/next-observation
state-update checks pass at the declared exact tolerance.

Thus a measured object must declare the future horizon it preserves. A shared
next response is insufficient for treating two expression histories as the same
dynamic state.

### A Smaller Relational Representation Can Preserve the Whole Experiment

The exact F04 sign-response subcase uses the same binary slope target and the
same paired observations in three representations. Full condition cells and a
three-state response difference each contain **0.398499991 nats**, whereas the
overall mean contains **0 nats**. A stochastic reverse map reconstructs the full
observation channel from the minimal difference channel; no such map exists
from the mean, as certified by a Farkas witness (`d'y=-1`, `E'y>=0`).

This supports preserving target-relevant relations, not maximal dimension or
unconditional averaging. It is standard finite information-order mathematics
applied to the declared GRAIN target, not a claim to a new Blackwell theorem.
The F09 process-family application of this certificate remains open.

### Opportunity and Choice Remain Partially Identified

C3 implements `(pi,q)=(.8,.5)` and `(.4,1)`, both with publication rate .4.
Visible opportunity permits separate estimates; masked opportunity gives the
population bound `q in [.4,1]`; published-only content has no total-opportunity
denominator. The sample plug-in interval is **not** a confidence interval:
its empirical inclusion is .925 at prefix12 and .975 at prefix48 in this run.
The exact population identification statement and sample uncertainty stay separate.

## Evidence and Verification

- `results/grain_gptpro_foundation_v3/repair_audit/r1/`: actual regression output,
  corrected null-record case, corrected garbling subcase, and OPEN final acceptance.
- `results/grain_gptpro_foundation_v3/F02/recovery_smoke_r2/`: latest smoke battery.
- `results/grain_gptpro_foundation_v3/F02/recovery_full_r2/`: canonical restored
  numerical results, source archives, models, predictions, author errors, paired
  CIs, exact future-law checks and manifests.
- Full run: **12 world/seed runs, 7,200 simulated author instances, 345,600 records**.
- **331 output hashes verified**. All paired comparison rows replay byte-for-byte
  against recovery_full_r1; r2 adds oracle-only scoring and plug-in-bound diagnostics.
- **59 tests passed in 1.61s**. These include the 34 earlier tests and 25 new
  adverse/math tests; they do not validate the still-unrepaired work packages.
- Ruff passes on new recovery modules/runners/tests with EXE002 ignored because
  this volume reports source files executable. Formatting and diff checks pass.
- Earlier source versions are preserved under `recovery_source_archive/r1/`;
  full_r2 also embeds the exact sources it ran.

## Next Executable Work, Not Human Pending

1. Finish F00's reusable execution/record/model isolation and adopt its request
   contract beyond F17. Review historical root artifacts before accepting DAG edges.
2. Complete F02 card-level replay/documentation review, rather than treating this
   numerical recovery runner as automatic acceptance of every F02 task.
3. Restore F03's paired designs, independent random streams, source-only shared
   selection slopes and application-prefix MAP intercept, all three observation
   views and all three prefix budgets. Do not reinterpret MNAR as repaired by IPW.
4. Resume F01's actual primary-source crosswalk and F14's real metadata audit;
   neither is a recruitment requirement. Continue remaining F04-F16 protocols in
   dependency order, including independent cold construction and real uncertainty
   propagation, not JSON descriptions of those operations.

No claim is made that these restored results close all 44 requirements, or that
the human personality interpretation is proven by synthetic target recovery.
