# SUICA v0.2.1 Release Notes

Release date: 2026-07-17

> Historical release under the former project name. The current method is
> GRAIN; the release identity and scientific record remain unchanged.

## Ruling

v0.2.1 is a **remediation release**. It contains no new scientific results.
It corrects the process defects found by the adversarial audit of the V7
evolution and the v0.2.0 release, and supersedes v0.2.0 as the current
verified state. v0.2.0 remains the frozen historical release: its tag is
immutable as-is, and nothing under `release/v0.2.0/` has been modified.

## Audit verdict being remediated

Substance: TRUST — the V7 numbers reproduce and the key machinery (max-T
family, W10-downgrade grounds) was independently verified. Process: multiple
governance violations, inventoried in full in
`docs/V7_PROCESS_AUDIT_20260717.md`:

- the v0.2.0 tag was deleted/re-created six times in 25 minutes on
  2026-07-15, and `LOCKBOX_MANIFEST.json` was regenerated five times
  post-"freeze";
- E4's own-vs-stranger alignment AUC family (.719--.805) was designed and run
  after the 8/10 transport failures were on disk yet shipped labeled
  "registered";
- "DOWNGRADED 2026-07-12" markers on the V6-W10/V6-E4 ledger rows were
  backdated (the downgrades were made 2026-07-15);
- the W10 downgrade cited a "Later audit" that existed nowhere as a document;
- TGEO-P9 carried three inconsistent statuses across documents;
- a dated audit-log entry was edited in place (df3fdae), and the freeze
  commit (5259f1b) deleted recorded caveats from the V5PORT ledger row;
- RELEASE_NOTES_V020 misstated the W4b effective-rank range (22.8--22.9; the
  WORD12 artifact value 22.7496 rounds to 22.7).

## Documentation fixes in this release

- **E4 relabel** (`docs/V7_DISCOVERY_LEDGER.md`,
  `docs/V7_OPERATOR_INDEXED_MULTIVIEW_V71.md`): the alignment family is now
  POST-HOC EXPLORATORY with the full 2026-07-15 timeline appended; the result
  stands as computed; a genuinely registered fresh-cohort confirmation is the
  listed follow-up.
- **Backdating corrections** (`docs/CLAIMS_LEDGER.md`,
  `docs/THEORY_FORMAL_NOTES_V3.md`): downgrade markers now carry the true
  date (2026-07-15 V7 audit) with the backdating disclosed; the in-place edit
  of the 2026-07-12 audit-log entry is disclosed in a dated appended note and
  its content adopted as controlling.
- **`docs/V7_W10_AUDIT.md` created**: the previously phantom "Later audit" is
  now a real audit record with the four verified grounds (cid-parity
  split-half + OP-15 ~2.5x precedent; same-sample axis fit, script lines
  ~407-417; selection-optimism-uncorrected null; missing
  opportunity/format/matched-stranger controls), the preserved numbers
  (r=.441, ICC-share .381, n=81), and the registered rescue-or-burial path.
  All citations updated.
- **TGEO-P9 reconciliation** (`docs/CLAIMS_LEDGER.md`,
  `docs/PAPER2_MOTION_SKELETON.md`, `docs/THEORY_V6.md`): one framing
  everywhere — pre-V7-standard status stands as recorded; V7-standard
  cross-fit re-audit PENDING; T8-prime's "instruments do not certify kernel
  rank" demonstrated for 2/3 instruments (W10, E4), TGEO-P9 the open third.
- **Rank corrections** (`docs/RELEASE_NOTES_V020.md`, appended Corrections
  section): 22.8--22.9 -> 22.7--22.9; plus the TGEO-P3 rank-4 vs W4b
  effective-rank bridge paragraph (different objects in different feature
  spaces; neither overturns the other; TGEO-P3's row stands unedited).
- **V5PORT caveat restoration** (`docs/CLAIMS_LEDGER.md`): the convergence
  numbers, the laugh_rate-"www" scorer-defect caveat, and the unregistered
  opening-#2 candidates note restored verbatim from the research repo as a
  dated appended restoration.
- **`docs/V7_PROCESS_AUDIT_20260717.md` created**: the permanent process
  record, including the six dangling tag objects and the standing rules.
- **Research-repo reconciliation** (historical twin): dated appended notes in
  `SUICA_CLAIMS_LEDGER.md`, `SUICA_THEORY_V6.md` (v6.13), and
  `SUICA_THEORY_FORMAL_NOTES_V3.md` adopt the V7 downgrades as controlling
  without editing the original sections; mechanical doc sync between the
  repos for those sections is prohibited.

## Code fixes in this release (companion changes)

- Geometry support-refusal is now the default (rows outside the declared
  support envelope are refused rather than scored).
- FactorBundle artifacts carry tamper binding (content hashes bound to the
  bundle identity).
- The consensus estimator docstring corrected to state what is actually
  computed.
- W4b gains a simultaneous-envelope option for the effective-rank confidence
  statement.
- CI now includes a release-identity check (tag/manifest/content-hash
  consistency), so a repeat of the v0.2.0 tag-churn pattern fails the build.

## Standing rules (binding from this release)

1. Release tags are immutable — bump the version instead of re-pointing.
2. Adjudications and dated log entries are append-only with true dates.
3. Registration commits must verifiably precede runs; families added after
   results exist ship labeled POST-HOC EXPLORATORY.

## Release engineering

- `release/v0.2.1/` carries this release's evidence bundle: a fresh
  `LOCKBOX_MANIFEST.json` (172 files) over the corrected tree,
  `TEST_REPORT.md` (full suite 318 passed), and the six deidentified evidence
  snapshots regenerated from the SAME frozen 2026-07-15 source runs as v0.2.0
  (no evidence artifact was recomputed).
- New scripts: `scripts/build_suica_v021_release_bundle.py` and
  `scripts/verify_suica_v021_lockbox.py` (minimal adaptations of the v0.2.0
  pair; content plus tag-identity verification against `v0.2.1`).
- `release/v0.2.0/` is untouched and remains verifiable at the v0.2.0 tag
  checkout; its manifest, test report, and evidence snapshots are additionally
  sealed inside the v0.2.1 manifest, so any future modification of the
  historical record fails v0.2.1 verification.
- Per-push CI now runs the v0.2.1 content-only verifier; the tag-triggered job
  runs the full v0.2.1 verifier (identity check) plus the theory closure
  audit.

## Compatibility

No result numbers changed in this release. Evidentiary TIERS changed for:
E4's alignment family (registered -> post-hoc exploratory), V6-W10 and V6-E4
(dates and citations corrected; downgrades unchanged), and TGEO-P9 (V7
cross-fit re-audit recorded as pending). The v0.1 P5 lockbox status and all
v0.2.0 sealed evidence are untouched.
