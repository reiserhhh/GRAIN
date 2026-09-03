# SUICA v0.2.0 V7 Technical Lockbox

> Historical lockbox under the former project name. The current method is
> GRAIN; this lockbox retains its original identity and estimands.

Status: sealed by the annotated `v0.2.0` tag and
`release/v0.2.0/LOCKBOX_MANIFEST.json`.

## What is frozen

Version 0.2.0 freezes the executable V7 technical object

\[
\mathcal G=(R,O,E,A,M,L,Q,V,\nu),
\]

including the reference population, observation operator, representation,
regularization, metric, tie-safe landmark orbit, refusal rule, profile map,
and uncertainty contract. It also freezes the corrected W2, W3, W4, W4b,
W7, and G1 aggregate decisions; their source-artifact hashes; the V7 code,
configs, schemas, protocols, tests, and the evidence-status lattice.

The highest supported release claim is:

`OPERATOR_INDEXED_RELATIVE_TEXT_GEOMETRY_WITHIN_DECLARED_DOMAIN`

The tracked release evidence contains aggregate rows only. Raw text, raw
identifiers, per-person scores, and restricted corpora are not included.

## What is not established

This release does not establish named personality factors, reliability or a
G coefficient, minimum detectable difference, state-trait decomposition,
clinical validity, cultural or language invariance, complete OOD detection,
universal factor count, market prediction, or transport of source norms,
landmarks, coordinates, and labels into another domain.

Unknown empirical relations are frozen as `UNTESTED/NO CLAIM`, not as null
findings. PANDORA and the MEPS feasibility corpus were development or
fixed-condition feasibility inputs; neither is a new V7 personality-validity
confirmation set.

## Independent future gates

1. **Repeated measurement**: crossed person/condition/occasion data must
   estimate reliability, state-trait variance, scoring uncertainty, and MDD.
2. **External construct validity**: cohorts, anchors, reliability correction,
   effect floors, artifacts, incremental validity, and replication must be
   frozen before labels are opened.
3. **Domain transport**: target-local reference fitting is mandatory unless a
   paired bridge identifies an alignment. Source coordinates cannot be reused.

Measurement-only use on a new corpus with target-local references does not
open an outcome lockbox. The first analysis against market outcomes, clinical
criteria, or unseen psychological labels is a new opening and requires a
separate preregistration.

## Verification

```bash
python scripts/verify_suica_v020_lockbox.py
sha256sum release/v0.2.0/LOCKBOX_MANIFEST.json
git rev-parse v0.2.0^{commit}
```

The annotated tag message records the manifest SHA-256. A verifier can compare
that value with the checked-out manifest and then verify every critical file.
The strict verifier requires Git metadata, an annotated `v0.2.0` tag at
`HEAD`, and the matching manifest hash. `--content-only` checks file integrity
for CI or source archives but does not establish release identity or closure.

## Relation to the v0.1 preregistration

The v0.1 P5 preregistration is a separate historical lockbox. Opening #1 was
spent on 2026-07-07: the overall rule failed (2/7; required at least 4/7), H2
and H6 individually passed, one legacy opening remains, and the Essays
confirm-half remains unopened. Version 0.2.0 does not rewrite that frozen
record or reset its budget.

## 2026-08-03 note: Essays confirm-half text consumption

The 2026-07-07 record above concerns the LABEL budget and remains intact: the
Essays confirm-half labels have never been read. Separately, during the V8
realtext runs of 2026-07-29..30 (`v8_corpus_local_composition_residual`,
`v8_conditional_concordance_spectrum`, `v8_exchangeable_background_audit`),
`load_essays_events` (scripts/run_suica_v8_realtext_relation_field.py) read
the TEXT of every Essays row (`usecols=[user_id, text]`) and ordered authors
by a fresh salt (`"v8rt-essays"`), bypassing the frozen 50/50 dev/confirm
split. Consequence: the confirm-half is no longer a text-untouched cohort.
Any future design that needs text-blindness must exclude the Essays
confirm-half or obtain fresh data. The label lockbox itself is unchanged.
Recorded by `docs/V8_PUSH_REMEDIATION_20260803.md` (item 2); ledger row
V8-ESSAYS-TEXT1.
