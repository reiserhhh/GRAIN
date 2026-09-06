# VM06: current Essays and X material inventory

2026-09-06. **Material/tokenizer preflight only; new neural domain tests pending.**

The current script parses Essays `user_id,text` and X post/author/text/time/language/
collector metadata only. Source containers are fully hashed, including the
Essays CSV which also contains questionnaire columns; those outcome values are
not parsed. No market price/return/engagement outcome values are used. Private
IDs and text hashes remain in ignored results. Historical test results do not
count as this program's domain-mechanism comparison.

## Material

| Corpus | Rows / authors | Relevant constraint |
|---|---:|---|
| Essays | 2,467 /2,467 | Exactly one document each; within-document pieces are not independent sessions |
| X market snapshot | 37,435 /19,954 | Median one post per author;17 symbols,257 collector query groups |

Essays median3,164 characters, no duplicated author IDs or exact cross-author
copied text. X median165 characters;84 distinct exact texts are shared across
authors. There are no duplicate/conflicting post IDs in this snapshot. All
37,435 timestamps parse and explicitly state a timezone, spanning
2026-03-17T16:52:52Z through2026-06-17T05:59:59Z.

| X source language | Posts | Authors | Authors >=6 posts | Authors >=12 posts |
|---|---:|---:|---:|---:|
| en | 14,253 | 7,597 | 303 | 91 |
| ja | 12,893 | 7,972 | 211 | 52 |
| ko | 4,717 | 1,685 | 139 | 46 |
| zh | 3,496 | 1,645 | 90 | 24 |

These counts precede length/copy/retweet and common-condition eligibility.
Language is source metadata, not independently verified linguistic ground truth.
Query groups are collector-mediated labels, not experimentally fixed situations
or direct measurements of authors' free topic choices.

## Tokenizer feasibility

Fixed sampling before token inspection: SHA256-ranked record IDs, at most200
per source-language cell, including200 Essays. Total2,708 sampled records and
8,124 tokenizer calls; zero neural forwards. The same cached tokenizer revisions
as the earlier preflight are used, without downloading models.

| Sample | BGE unknown fraction | English E5 unknown fraction | Multilingual E5 unknown fraction | Multilingual E5 complete <=512 |
|---|---:|---:|---:|---:|
| Essays,200 records | .000006 | .000006 | 0 | 36/200 |
| X en,200 records | .00897 | .00872 | .00296 | 200/200 |
| X ja,200 records | .23469 | .22904 | .00161 | 200/200 |

English BGE/E5 would discard substantial Japanese token identity through UNK.
An embedding-only downstream algorithm does not remove this upstream issue.
Low UNK for multilingual E5 permits testing, but does not prove measurement
invariance, language equivalence or personality validity. Essays need explicit
lossless windows; silent512-token truncation would drop most sampled documents.

## Next bounded domain experiment

1. Freeze common lossless observer windows and language/length/copy rules before
   extracting new vectors. Reuse multilingual E5 as the Japanese-capable observer;
   English BGE/E5 comparisons belong in the English-only compatibility stratum.
2. Essays: test within-document representation structure/order sensitivity,
   labelled as technical split behavior, not author retest or personal history.
3. X: use multiple distinct posts per author with source/copy checks, and
   language-/collector-stratified other-author controls. Report eligible N anew;
   do not interpret raw >=12-post counts as final cohort counts.
4. Keep corpora separate. There is no known common person across Essays/PANDORA/X;
   compare scoped effect estimates, never invent row-pairing or X personality labels.
5. Preserve the frozen source-only PANDORA comparison in parallel. No domain result
   is grounds to select a bridge model or claim cross-cultural psychological invariance.

Private inventory: `results/grain_vector_mechanisms/vm06_domain_inventory_r1`;
identity`d60f6acd95eba9124bf497b5009e6143832710639cdda73747eed7bfb909b12f`.

```bash
python scripts/prepare_grain_vector_domains.py \
  --data-root "$PERSONA_DATA" \
  --output-dir results/grain_vector_mechanisms/FRESH_DOMAIN_INVENTORY
```
