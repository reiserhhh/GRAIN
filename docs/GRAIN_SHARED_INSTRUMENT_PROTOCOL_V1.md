# GRAIN shared instrument protocol V1 (R14; P05 S2; P03 pp.25-28; P01 pp.9-14)

Written 2026-09-26 JST. This is one protocol that every reader of open text must pass
before its scores are interpreted. It combines the three components P05 S2 names. The
machine-readable form is `configs/grain_shared_instrument_protocol_v1.json`, and the
checker is `scripts/check_grain_shared_instrument_protocol.py`.

## Component S: stress test (counterfactual sensitivity; P01 pp.11-14, 23)

- **S1 nuisance battery.** Source-declared g_N edits: style, length, paraphrase, order,
  first/third person, language, politeness, expressive ability. Each is applied to real
  text with content-retention validation, and the result is reported as edit distance
  over within-person distance, per coordinate.
  - Pass: every upper 95% bound is below .5, AND the supervised readout shift is below
    .10 SD per edited unit.
- **S2 construct battery.** Construct-changing edits under a construct contract, with
  human confirmation. Pass: CSNR > 1 on the target coordinate, and the eigenproblem
  Sigma_P v = lambda Sigma_N v yields a selective direction.
- **S3 edit battery of P03 p.26**: delete, counterfactual insert, pragmatics-only,
  ownership, query, model, language. Each must give a repeatable change in the readout
  along a documented path.

## Component R: Reader-MTMM (P01 pp.14-15; P03 pp.18-19)

- **R1.** At least 2 readers x 2 traits x 2 occasions; shared method and trait
  identified with an anchor.
- **R2.** Reader-variance ratio, from reader x item, reported per target.
- **R3.** Out-of-reader transfer. A map fitted on one reader preserves the other reader's
  intervention signatures on held-out items.
  - Pass: cosine above .7 for every declared intervention and no absent channel.

## Component M: model as instrument (P03 pp.26-27; P04 pp.15-16)

- **M1.** Cross-reader intervention-effect concordance and the single-law test.
  - Pass: single-law compatible, no absent channel.
- **M2.** Internal path. The same edit moves the reader along a repeatable internal
  direction (vector-path audit). It needs native forwards or backward passes.
- **M3.** Shortcut exclusion: topic, occupation and stereotype shortcuts are controlled
  before interpretation.

A reader passes only when every component passes. Otherwise its score remains a
computational output.
