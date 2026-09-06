# VM06 domain context function: frozen V1

2026-09-06. Prospective extension of the completed RC context/order instruments,
not a new psychological score or an expansion on the same44 MEPS responses.

## Question and scope

Does the frozen Qwen observer use correct source context when predicting later
observed text in Essays and X, and does an exact-token order disruption alter
that advantage? High vector similarity alone does not establish equal function.
Compare three separate strata: Essays English, X source-en, X source-ja.
No personality, mood, price, return, or engagement outcomes. No invented cross-
corpus author links. Essays parts measure within-document continuity, not retest.
X posts measure source-account correspondence, not known human identity or trait.

## Source selection before neural results

Use the exact source container hashes in `vm06_domain_inventory_r1`; only its
allowlisted text/structural columns are parsed. Fixed salt `vm06-domain-context-v1`.
All raw text/tokens/vectors/IDs and row-level results stay ignored and local.

- Essays: preserve original characters. Divide each document into six successive
  spans at character sextiles moved to the next whitespace boundary. Context is
  the first three spans verbatim; current is the sixth span verbatim. The intervening
  two spans are a declared gap, not silently treated as observed. No second
  independent session is inferred. At most96 eligible authors, sorted by salted ID.
- X: require nonempty author/post IDs, valid zoned timestamp, en/ja source label,
  at least40 text characters, no leading `RT @` retweet marker. Exclude all exact
  texts shared between different accounts; deduplicate exact within-account text
  by earliest time. Missing/ambiguous IDs fail or are explicitly excluded.
- Assign each account at most one (language,symbol) cell, the cell with most eligible
  posts (ties resolved by salt). Require >=6 distinct posts; take six evenly spread
  chronology indices. Context concatenates the first three complete posts with
  newlines; current is the sixth complete post, retaining the two-post gap.
- After token-budget checks, retain symbol/language cells with >=4 accounts;
  at most12 accounts/cell and96/language, selecting whole cells in salted order
  while respecting the cap. Report exclusions and final cell Ns. Collector symbol
  is an observational control, not randomized topic or unconfounded choice.
- Qwen current must have16-256 complete native content tokens; context16-640
  ordinary tokens and complete prefix<=768. Reject an entire candidate on failure,
  never truncate to rescue it. Current tokens must match no-context tokenization.

## Observer and conditions

Same pinned Qwen2.5-1.5B-Instruct weight/tokenizer hashes as RC; FP32 eager MPS,
no generation, training, or KV cache. Context is an earlier **user** turn and
current is another user turn. Do not mislabel source authors as assistants.
The repeated-user template is a diagnostic likelihood design, not a natural
assistant-generation use case. Body origin768, physical length1024, explicit
position IDs, layers input/10/16/final norm. Current and boundaries stay fixed.

Within each frozen cell, Hungarian log-context-length matching provides a
deranged donor bijection; a second fixed random derangement is also retained.
Donors never cross account, symbol, language, or corpus. Essays is one prompt cell.
The same six conditions run for every retained author:

1. Correct context, native order.
2. Matched other-author context, native order.
3. Random other-author context, native order.
4. No context.
5. Correct context, deterministic eight-token-block permutation (seed731).
6. Matched context, the same donor-specific permutation rule.

Use the existing exact-token `permute_prefix` operator on content tokens only;
do not decode/retokenize after permutation. Preserve outside-span tokens and
multiset, verify inverse and current-input state identity. Shuffling is an
off-distribution intervention and cannot yield a personality-information fraction.

## Endpoints and uncertainty

Primary: per-author mean log probability of observed current tokens2..T
(first token excluded), correct minus length-matched context. All-token mean
is secondary. Also report correct-random, correct-none, shuffled correct-matched,
and order interaction: native advantage minus shuffled advantage.
Layer-wise mean-vector cosine/displacement accompanies these functional contrasts;
input-embedding equality is an instrument control, not a useful predictor.

Compute1,000 case bootstrap draws (seed731) separately per corpus/language,
conditional on selected texts and frozen donor assignments. These are not CIs
for a newly sampled social population or donor assignment; texts and account
ownership/selection remain observational. Do not count tokens, seeds or layers
as independent subjects. Report positive-case fraction and heterogeneity. No
universal psychological r/AUC cutoff or significant-only reporting.

## Gates and deliverables

Before private neural forwards, public EN/JA instances under the new role template
must pass repeated/self-patch exact logp replay, no-context body identity,
fixed/compact mean logp difference<=.02 and minimum site cosine>=.999, matching
the established RC numerical gate. Exact bag, inverse and role-boundary tests
must pass. Do not weaken gates after seeing domain results.

At most288 authors x6 conditions plus public/replay controls; max2h per neural
run, sampled MPS<=9 GiB. Save complete case checkpoints and a terminal state;
do not promote an incomplete stratum. One public observer pass followed by the
frozen full cohort is the plan, not adaptive selection on early effects.

Private: material manifest/cases/exclusions/donor mapping, source snapshot,
model manifest, public checks, per-token likelihoods, mean-state vectors,
case effects, summaries and independent replay. Public: domain report with
counts, all effects, limitations and implications for the observer specification.
This complements the full-source representation/head pipeline rather than
replacing it. The mE5/BGE/E5 domain representation contrast is a separate pending
comparison; no geometry-based winner or bridge choice is made here.
