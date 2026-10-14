# F01 recovery protocol v1 (replaces the rejected 8-row r1)

Frozen 2026-09-26 before the canonical run. Cards F01-02, F01-03, F01-04 here;
F01-01 (mechanism graph/object dictionary) and F01-05 (50-requirement map) are
produced with the requirement ledger. The rejected `F01/r1` is retained.

- F01-02: scan all five frozen sources (P01–P04 extracted PDF text with page
  markers; P05 summary sections) with `f01_sources.extract`; deduplicate by DOI >
  arXiv id > specific URL > normalized title; classify by a declared keyword rule
  (load_bearing / context / tool / unavailable_locator) with card-match overrides.
  Primary cards come from `configs/grain_f01_primary_cards_v1.json`; every
  locator must come from a cached primary file (hash recorded) or an explicit
  web fetch; unavailable primaries are recorded with the reason and never
  reconstructed. Bot-check/reCAPTCHA pages are not bypassed.
- F01-03: adoption ledger for every predecessor method family named by the
  protocol (adopt / adopt_with_difference / finite_certificate /
  not_applicable / not_executed, reason, responsible card).
- F01-04: identification contract plus four minimal counterexamples computed
  from scratch (unknown exposure, state-shift gauge, shared method, equivalent
  rival): equal observable law within 1e-12, different target, and the added
  observation that breaks the equivalence (difference > 1e-6).
