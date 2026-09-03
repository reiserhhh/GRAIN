# SUICA Freeze & Independent Release Plan v1

> Historical v0.1 plan. The realized v0.2.0 freeze is specified in
> `V7_LOCKBOX_V020.md` and `RELEASE_NOTES_V020.md`. Opening #1 of the legacy P5
> lockbox was spent after this plan was written; this text is retained for
> provenance rather than rewritten as if it had predicted the result.
> SUICA is the historical name used by this frozen plan; the current method is
> GRAIN.

Created: 2026-07-05. Status: PLAN (execution gated on the reviewer-readiness
sweep and user go-ahead; precedes OP-3 lockbox opening by design).

## Purpose

Freeze the SUICA method at a reproducible, self-contained state and publish it
as an INDEPENDENT worktree/repository, separate from project persona, so that
reviewers and other researchers can reproduce the validation chain from zero.
Per the user's decision: **v0.1.0 is frozen with the lockbox UNOPENED** — the
strongest possible reproducibility statement, because external-validity
confirmation (P5) can then be performed by anyone, including reviewers,
exactly once, under the committed preregistration.

## Repository shape (proposed name: `suica-method`)

```text
suica-method/
├── README.md                  <- method summary + quickstart + claims table
├── LICENSE                    <- (user choice; MIT or CC-BY-4.0 for docs)
├── CITATION.cff
├── requirements.txt           <- pinned: numpy/pandas/scikit-learn/scipy/
│                                 statsmodels/pyarrow (+ optional: empath,
│                                 sentence-transformers for OP-9)
├── docs/
│   ├── THEORY.md              <- from SUICA_RIND_THEORY_BASE_V3
│   ├── RULEBOOK.md            <- from SUICA_FIXATION_RULEBOOK_V1
│   ├── CLAIMS_LEDGER.md       <- full audit history (6 rounds, retractions)
│   ├── OPEN_PROBLEMS.md
│   ├── VALIDATION_PLAN.md     <- P0-P5 framework
│   ├── PREREGISTRATION.md     <- lockbox opening #1, UNOPENED, frozen
│   └── DATA_ACCESS.md         <- how to obtain PANDORA/Essays (see below)
├── suica/                     <- package: suica.py core + v2_lib estimators
├── scripts/                   <- the validation pipeline (P0, P0B, tiers,
│                                 prep, P1-P4, E1-E7, OP5, OP9)
├── tests/                     <- test_suica.py (39 tests, no data needed)
└── examples/
    └── synthetic_demo.py      <- P0 + P0B run out of the box (no real data)
```

## Data policy (hard constraints)

1. **PANDORA cannot be redistributed.** The original dataset is
   access-controlled by its authors (Gjurkovic et al.); DATA_ACCESS.md ships
   the request procedure + the exact prepared-file schema + SHA-256 manifests
   of our prepared inputs so others can verify byte-identical preparation.
2. **X-market data is excluded entirely** (proprietary collection; only its
   aggregate numbers appear in reports).
3. **Essays**: same policy — schema + acquisition pointer, no raw text.
4. **What DOES ship executable from zero**: the synthetic ground-truth
   harnesses (P0, P0B). A reviewer with no data can verify the estimator
   layer in minutes; with PANDORA access they can rerun the full chain via
   the frozen tier-split script (stable-hash membership is reproducible from
   the raw dataset without our files).
5. No user text, no user IDs, no per-user score tables in the public repo.
   Reports containing aggregate statistics only.

## Freeze mechanics

1. Complete the reviewer-readiness fixes (this session's sweep).
2. `git worktree add` / new repo initialized from an export of the frozen
   file set (NOT a fork of project persona history — the parent repo contains
   unrelated market-agent material and early exploratory noise). Provenance:
   README records the parent commits (154822a, 05be394, cad83d5, + final
   freeze commit) for auditability.
3. Tag `v0.1.0-prereg-sealed`. The preregistration file's commit hash is the
   seal; opening the lockbox later = new tag `v0.2.0` with results appended.
4. CI (optional, later): run tests + P0 + P0B on push.

## Definition of "clean" for v0.1.0

- [ ] Every headline number in docs traces to a script + report in the repo.
- [ ] No hardcoded result strings; reports regenerate from artifacts.
- [ ] Deprecated/duplicated code paths removed or marked (e.g., the historical
      anchor-weighted objective in suica.py carries a deprecation note; naive
      centering marked falsified-do-not-use).
- [ ] Construct registry with operational names (incl. wcl_60 renamed to
      apostrophe-omission register; wcl_02 flagged choice-coupled).
- [ ] Reviewer-readiness sweep items resolved or explicitly listed as
      limitations in README.
- [ ] Lockbox unopened; opening budget statement in PREREGISTRATION.md.
```
