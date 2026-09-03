# GRAIN Japanese Scorer — Equivalence Protocol v1 (OP-8)

Created: 2026-07-07. Governs `scripts/suica_ja_scorer_v1.py`.
Rulebook anchor: D1 (population- and language-relative measurement; no
cross-language numeric comparison, ever).

## Three-stage protocol (status per stage)

| Stage | Content | Acceptance rule | Status |
|---|---|---|---|
| 1. Forward translation | EN anchor lexicons for the frozen v3 battery (7 lexicons: self_focus, second_person, directive, negative_affect, conflict_threat, uncertainty, novelty_play) translated to Japanese; composition formulas carried over UNCHANGED; machinery validated on synthetic planted truth | P0-style self-test: planted-order Spearman >= 0.90/construct at 120 sentences; determinism exact | **DONE 2026-07-07** — recovery 0.914-0.952 (40-sentence attenuation 0.868-0.902 disclosed); results/suica_ja_scorer_v1/ja_scorer_selftest.json |
| 2. Independent back-check | DeepSeek (different family) blind classification of every JP token into EN category names | >= 80% blind back-match per lexicon | **DONE 2026-07-07 (5/7 clean, overall 0.833)**: 5 categories 1.00; directive 0.00 = scope artifact (token lexicon = advice-nouns; force in patterns — back-check patterns next); conflict_threat 0.385 = EN-shared boundary fuzziness, battery-invariant (tension sums both bins). results/suica_op8_stage2_backcheck_v1/ |
| 3. Within-language re-validation | The full P0 -> P1 chain ON A JAPANESE CORPUS: synthetic ground truth with JP text statistics, then disjoint-occasion retest on real JP writers | Same bars as the EN chain (P1 SB targets, volume curve) | PENDING — needs a Japanese longitudinal text corpus; the X market corpus in this repo is Japanese but too small/topic-locked (n~31 accounts) for retest |

Until stage 3 closes, JP scores are MACHINERY-GRADE ONLY: usable for
pipeline tests and demos, not for any measurement claim about a person.

## Structural translation decisions (disclosed, stage-3 sensitivity targets)

- **T1 tokens**: fugashi/MeCab (unidic-lite) morphemes; denominator excludes
  pure punctuation; rates per 100 tokens as in EN.
- **T2 patterns**: Japanese realizes directives/hedges morphosyntactically
  (すべき, してください, かもしれない); these are counted as raw-string regex
  hits, a structural deviation from the token-only EN scorer.
- **T3 pro-drop**: Japanese omits subjects; first-person RATE levels are not
  comparable to EN levels. The measurement object is within-JP-population
  between-person variance (D1).
- **T4 formulas**: identical to `suica_v2_lib.score_slices_v2` — tension =
  .40*(neg+conf+unc) + .35*unc + .25*conf; directive = sqrt(directive_rate x
  second_person_rate); first_person = self_focus rate; novelty = novelty rate.

## Known open risks for stage 3

1. Keigo/register: formal Japanese suppresses direct second-person pronouns
   (directive blend may under-fire in polite registers) — the OP-7a register
   lesson predicts per-register norms will be needed.
2. Orthographic variants (漢字/かな/カナ) are only partially enumerated;
   stage 2 must sweep variants systematically.
3. The negative_affect list skews clinical (不安/鬱); colloquial negativity
   (むかつく, うざい, しんどい) is deliberately deferred to stage 2 to keep
   stage 1 a clean forward translation of the EN source words.

## Real-text smoke observation (T1, 2026-07-07)

On the repo's Japanese X market corpus (12 accounts >= 3,000 chars, topic =
semiconductor stocks), rates are non-degenerate but SPARSE: means 0.005-0.098
per 100 tokens, first-person nonzero for only 2/12 accounts
(results/suica_ja_scorer_v1/ja_scorer_x_smoke.json). Reading: topic-locked
telegraphic register (rind regime: domain-fixed) x pro-drop (T3) x stage-1
lexicon narrowness. Not a validity datum; it sets the expectation that
stage 3 needs a diary/dialogue-register JP corpus, not market chatter.
