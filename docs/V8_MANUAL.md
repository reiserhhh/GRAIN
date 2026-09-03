# GRAIN V8 Manual — Working With the System

Status: living practitioner manual (2026-08-03). The sealed v0.2.1 worked
example (`docs/WORKED_EXAMPLE_MANUAL.md`) remains the frozen construction
record for the V7-era battery; this manual covers the V8-era repository as it
is actually operated. Wherever this manual and `docs/CLAIMS_LEDGER.md`
disagree, the ledger controls.

Audience: a researcher or engineer who wants to (a) verify the repository,
(b) run and reproduce protocols, (c) register a new experiment without
breaking governance, or (d) understand what a number in a report is allowed
to mean.

---

## 1. Install and verify (no data required)

```bash
pip install -r requirements.txt        # or the CI pin set: requirements-lock-main.txt
python -m pytest -q tests/test_suica.py                 # 39 passed
python -m pytest -q -p no:cacheprovider                 # 970 passed (2026-08-03)
python scripts/verify_grain_v021_lockbox.py             # portable v0.2.1 seal check
```

Expected deviations you should NOT be surprised by:

- On a **clean checkout** (fresh clone, CI), 2 tests skip with reason:
  they hash run artifacts under `results/`, which is untracked by design.
  Tracked-file hash checks still run. (Process rule 4,
  `docs/V7_PROCESS_AUDIT_20260717.md`.)
- `requirements-lock-v0.2.0.txt` is the frozen 2026-07-15 audit environment
  and is never edited; CI installs `requirements-lock-main.txt` (same pins
  plus the V8-tree dependencies).

CI on every push runs the full suite on Python 3.12 and 3.14 and verifies the
sealed v0.2.1 tag tree in a separate worktree. The `release-lockbox` job runs
only on `v*` tags — seeing it "skipped" on branch pushes is correct behavior.

## 2. Repository layout

| Path | What it is |
|---|---|
| `docs/` | Theory canon, protocols/plans, seals, ledgers, governance notes |
| `reports/` | One report per executed study (tracked; numbers must trace to artifacts) |
| `scripts/run_suica_*.py` | One runner per protocol; deterministic, config-driven |
| `configs/*.json` | Frozen run configurations, incl. source-lock hash blocks |
| `suica_core/`, `suica_sim/` | Library code: estimators, gates, bridges, synthetic worlds |
| `tests/` | Full regression + protocol/lock tests (the 970) |
| `results/<protocol>/<run>/` | UNTRACKED run artifacts: `decision.json`, CSV/parquet rows |

Entry points for reading: `README.md` → `docs/SUICA_THEORY_ROUTE_INDEX.md`
(map) → `docs/SUICA_UNIFIED_THEORY_SYSTEM_V8.md` (canon) →
`docs/CLAIMS_LEDGER.md` (what is actually claimed).

## 3. The system in ten minutes

- **Chain.** (H,F)→B→X→Z→𝔄→𝔐→Θ→D. Everything upstream of 𝔐 is technical;
  Θ (named constructs) and D (decisions) are INACTIVE — nothing in this
  repository names a personality factor or produces a diagnosis.
- **Output types.** τ ∈ {V, R, P}: individual vector candidates, relations,
  population fields. They do not convert silently; the typed R→V bridge
  (`suica_core/m4_relation_bridge.py`) is the only promotion path and it
  refuses group-only structure (designed-null record 200/200).
- **Operators.** Every observation is indexed by a slicing/representation
  operator; a changed implementation is a NEW operator (F16). Results carry
  their operator; comparing across operators is a licensed act, not a default.
- **Reference-relative scores.** A score is a position relative to a declared
  reference population/opportunity/calibration; batch-standardized
  exploratory scores are not reusable individual scores.
- **Refusal.** Out-of-license comparisons raise; gates refuse when support,
  comparability, or rigidity conditions fail. A refusal is a result.

## 4. Anatomy of a protocol run

Every executed study follows the same shape:

1. **Config** — `configs/<protocol>.json`, often containing a
   `source_lock`/`detector_lock` block: SHA-256 of the exact code files the
   protocol froze. Tests re-verify these hashes forever after.
2. **Runner** — `scripts/run_suica_<protocol>.py`, deterministic given config
   + seeds; long batteries run in registered chunks.
3. **Artifacts** — `results/<protocol>/<run_tag>/`: always a `decision.json`
   (machine-readable verdict: status string, gate booleans, headline
   estimates, provenance) plus row-level CSV/parquet. Artifacts are
   immutable once written; corrections happen in NEW files or dated notes,
   never by editing a run's outputs.
4. **Report** — `reports/<PROTOCOL>_REPORT.md`: human-readable, every number
   traceable to the artifacts, with a Reproduction section giving the exact
   command.
5. **Ledger row** — `docs/CLAIMS_LEDGER.md`: claim, artifacts, STATUS (the
   controlling tier), result-and-boundary text.

To audit any headline number: report → artifact path → recompute from rows.
The 2026-08-01 review demonstrated this end-to-end (15/15 headlines traced,
9 recomputed exactly).

## 5. Running the standard verifications

```bash
# Estimator layer against planted ground truth (synthetic, no data):
python scripts/run_grain_synthetic_ground_truth_v2.py
python scripts/run_grain_p0b_thin_cell_regime_v3.py

# Typed R->V bridge unit + calibration surface:
python -m pytest -q tests/test_m4_relation_bridge.py tests/test_m4_bridge_hetero.py

# A full M4-D leg (example: two-stage construction, Leg 5):
python scripts/run_suica_m4_d_two_stage_leg5.py
```

M4-D/E/F runners assert bit-exact reproduction of previously persisted
anchors (at the 1e-16 level) before computing anything new — if you see an
anchor assertion failure, your environment or checkout differs from the
recorded one; do not "fix" the anchor.

The PANDORA external-connection script
(`scripts/run_suica_v8_pandora_external_connection.py`) reproduces the
adjudicated campaign; if you touch it, read the report's "Post-hoc
adjudication (2026-08-03)" section FIRST — the bridge headline may only be
cited together with its same-day null and the operator-selection order.

## 6. Registering a new experiment (the process contract)

Binding rules (V7 process audit + 2026-08-03 appendix):

1. **Register before you run.** The registration — question, design, leans
   (directional expectations), pivot conditions (what outcome forces what
   interpretation) — goes into the plan document and its own commit BEFORE
   execution. Results land in a separate, later commit. A seal that shares a
   commit with its outcome is EXPLORATORY by construction.
2. **Append-only adjudication.** Never rewrite a conclusion in place: add a
   dated note/addendum; originals stay. Never `git commit --amend` shared
   history.
3. **Run-directory naming.** No date-like suffixes that are not execution
   dates.
4. **Clean-checkout discipline.** Tests may depend on `results/` artifacts
   only via skip-with-reason guards; verify a fresh clone (or at minimum a
   fresh venv from the lock) before pushing anything that touches
   dependencies or artifact-reading code.
5. **Labels and holdouts.** Any label join is a consumption event: it must be
   registered before the join, recorded in the ledger, and the operator must
   be fixed BEFORE labels are seen. Every operator run on a question is
   reported together. Text-blindness claims must check the consumption
   record first (the Essays confirm-half TEXT is spent as of 2026-07-29..30;
   its labels are not).
6. **Tier language.** Prose never exceeds ledger status. If the ledger says
   EXPLORATORY, the abstract says exploratory.

## 7. What a number is allowed to mean

- **Tiers.** T3 = holdout-confirmed; T4 = sealed confirmatory (lockbox
  opening); EXPLORATORY = pattern worth registering, not evidence;
  POST-HOC = selected after seeing outcomes — citable only with its
  selection history. Synthetic results are labeled finite-synthetic and
  never imply human prevalence.
- **Budget/operator indexing.** Many quantities are budget- or
  operator-indexed. Example: the V37F dense-tail residual AUC is .8635 at a
  256-event budget and .5393 at 512 — quoting it without the budget is a
  reporting error (it was one, and was corrected).
- **Author AUC ≠ personality validity.** Worlds with zero individual
  structure yield author AUC .864. Identification results never license
  measurement claims here.
- **Nulls are results.** The program's real-text direct-prediction answer is
  currently a null (V8 invariant score, Big5 mean r=−.005, below nuisance)
  and is reported as such.
- **Forbidden outputs.** Personality/state naming, individual diagnosis,
  clinical or forensic use, cross-regime level comparisons outside license,
  promoting P-level structure to individuals without the bridge.

## 8. Verifying integrity end to end

```bash
# Sealed release (content + identity when on the tag):
python scripts/verify_grain_v021_lockbox.py            # --content-only for CI/archives

# Source locks (protocol code frozen at run time):
python -m pytest -q tests/test_v8_w0_calibration_v37h4c.py \
                    tests/test_v8_permutation_orbit_frontier_v37h4d_r2c.py
```

The F16 rule: a changed implementation is a new operator. If a lock fails,
you either reconstruct the recorded source state or register a NEW study ID —
re-baking hashes to make a lock pass is prohibited (the 2026-08-03 F16
reconciliation note documents a three-way verification that the committed
tree equals the originally locked sources).

## 9. Pitfalls (each one happened)

| Pitfall | What actually happened | Rule it produced |
|---|---|---|
| Trusting local green | jsonschema/persim/cryptography missing from the CI lock; artifact-reading lock tests failed on clean clones | lock-main + process rule 4 |
| Outcome-contingent operator choice | bridge r=−.103 null → operator switch → r=+.498 shipped alone | remediation item 1 forward rule |
| Penalty artifacts read as physics | ridge bias manufactured a "budget-invariant floor" | Leg 4b retroactive correction; de-bias before interpreting floors |
| Metric structure read as world structure | paired diagnostics rewarded shared shrinkage; cleaning LOWERED the score | common-mode disclosure requirement |
| Believing author separation | AUC .864 with zero individual structure | T8′ deflator, cited program-wide |
| Fixing an anchor to make a test pass | (prevented) | anchors are evidence; environment must reproduce them |

## 10. Where to go next

- Theory questions → `docs/SUICA_UNIFIED_THEORY_SYSTEM_V8.md`, then the
  route index for which experiment established what.
- "Can I claim X?" → `docs/CLAIMS_LEDGER.md` +
  `docs/SUICA_FOUNDATION_GAP_LEDGER.md` (which gap your claim crosses, and
  its refusal while open).
- Open work → D3 panel composition redesign; discovery-objective redesign
  (lead suspect: freeze-whitening scale family); bridge debiased cross-half
  floor; v0.3.0 numeric lockbox.
- History and lessons (Japanese) → `docs/V8_DEVELOPMENT_REPORT_JA.md`.
- Paper draft → `docs/V8_THEORY_PAPER.md`.

## 11. The identity-era instruments (dated supplement, 2026-08-10)

Four instruments joined the toolbox; each has a certificate and a
misuse mode. Controlling documents: `docs/SUICA_IDENTITY_THEORY_V1.md`,
`docs/SUICA_V8_IDT_INTEGRATION.md`.

- **Frame-refreshed discriminator (T6″ v2).** Use for ANY
  reproducibility-as-identity reading. Rule of thumb: if the frame
  (norms, issuer sample) is shared across your replicates, your
  "stable individual signal" may be a forgery — refresh the frame and
  keep only what survives; a reader that PROFITS from frame error is
  reading a forgery.
- **Two-split probe (T6′).** ρ_interleaved − ρ_contiguous estimates the
  state share at card level, reader-free. Read the GAP, never one half
  alone.
- **Taxometer (η̂).** Reads whether a typed world is
  dimension-like (identity off the type axes) or type-discretized
  (identity on them), ±0.125 with a known +O(0.05) bias at the η=0
  pole; requires state-shape whitening. Decide whether
  identity-removal-then-cluster can work BEFORE trying it.
- **Completeness meter.** A typology's surviving-identity share — the
  "error = 0" audit. Certified on true/given partitions with the
  partition-propagation correction; noise floor ~0.005; always
  cross-fit.

### Pitfalls added by the identity era (each one happened)

1. Shared-frame reproducibility read as identity (it was issuer error —
   the forged card improved re-identification).
2. De-framing used as preprocessing (it zeroed trait recovery — the
   frame is the trait's scaffold in this family; diagnostics only).
3. A difference-law and a pooled curve glued into a level formula (it
   missed a sealed band by 5× and under-predicted its own training
   arms).
4. Bars on the wrong quantity (three separate defects: energy vs ARI,
   ARI vs overlap, zero-band vs honest penalty) — rule 19: the bar
   shadows the theorem's OWN quantity.
5. Hand-typed numbers in reports (14 wrong cells caught by mechanical
   re-read) — generate tables from artifacts; rule 24 extends artifact
   verification to prose.

For any future real-text work, `docs/SUICA_REALTEXT_GOVERNANCE.md`
(R-G1..R-G8) is binding.
