# VM05 source-safe same-current history counterfactual

2026-09-06. New material/instrument protocol, before any new neural history
features are compared with source outcomes. This implements the missing context
object from GRAIN_VECTOR_INFERENCE_MECHANISM_PLAN_V1, not another task-difference
or author-matching substitute. The existing observer/source jobs remain unchanged.

## Object and question

For each of the three original12-comment views, keep the latest comment as
current x and all11 earlier comments as own history H. Require strict time order,
distinct discussions and no normalized duplicate bodies. All36 own comments are
preserved across the three views; reject an unsupported author rather than
changing the input budget or replacing an awkward comment.

Compare current-token representations and decision positions under own history,
one matched other-author history and a history-invisible control:

```
delta_history(x,q) = h(x | H_own,q) - h(x | H_matched,q)
delta_visibility(x,q) = h(x | H_own,q) - h(x | history invisible,q)
task_interaction = delta_history(x,q_disposition) - delta_history(x,q_behavior)
```

The first is relative to a particular matched donor, not an exactly evaluated
population expectation. Matching conditions on topic/time/length proxies and may
remove meaningful choice variation. Preserve all arms; never interpret a residual
as pure personality, or a common component as disposable noise.

## Reference and support rules

The99 historical geometry-reference authors in the original common-material
budget table are a **fixed, label-unused history dictionary**. Exclude them from
all source prediction/evaluation rows for this experiment. They are MBTI-only and
not culturally/personality representative; report this reference dependence.
No source/test author donates to another test/train author. No bridge donor,
questionnaire value, source label, newly learned embedding or outer-CV score is
used to select histories. Existing reference exposure remains disclosed.

Candidate histories consist of the11 prior comments of each reference view.
They must predate the target current, share no discussion/comment/normalized body
with any of the target's36 comments, and have a different author. Reuse the
earlier fixed matching rule: absolute log token-length ratio plus .15 times
log-time-gap difference plus .5 times one-minus-subreddit-Jaccard. Break ties
deterministically. Choose once on raw input, then reuse the same donor under
raw/cue/placebo; no outcome- or mask-dependent rematching. Report reuse, support,
length/time/topic mismatch rather than claiming these proxies perfectly match.

## Token and numerical controls

Build own and matched prompts with identical system/task/current. Tokenize the
actual pinned Qwen chat template, locate the full history block via character
offsets and split the token stream into prefix/history/suffix. Include boundary-
straddling history tokens in the history block. Prefix/suffix and all current
token IDs must match exactly; otherwise reject the view before forwards.

Reserve the longer history block's token count as one fixed slot. Shorter arms
use invisible masked padding inside that slot. The no-history arm is **the same
layout with history keys masked**, not a newly tokenized shorter prompt and not
the instruction "No history". All visible current/suffix positions are identical
within the comparison. Use explicit absolute position IDs, no KV cache, fixed
3072 execution length and the original16-token safety margin. Never silently
truncate. This is an attention-mask intervention, not ordinary conversational
memory removal; record both the mask and the retained token positions.

Keep all four originally declared Qwen sites in the protocol; block10 remains
primary. Capture input-embedding equality as an instrument, not an author score.
Pre-question current states must be question-invariant under the same layout.
Qualification must test replay, changing only masked token IDs, exact current
token/position equality, all-visible agreement with the existing observer when
layouts coincide, and masking/restoring history. Tiny synthetic models can test
plumbing but cannot replace pinned-real-checkpoint qualification.

## Execution phases and comparisons

1. Label-free complete-source material audit, including support in each group,
   reference exclusion, official folds, same-ready overlap and token budget.
   No weights are loaded during this phase. Save source hashes and private
   selection/layout commitments. Unavailable cases remain explicit.
2. Synthetic instrument controls, then a fixed technical panel on the pinned
   Qwen checkpoint after the current GPU collector reaches its terminal state.
   Do not resync a running remote repository. Budget actual forwards/cache bytes
   before committing a full source queue; each invocation remains bounded.
3. Collect the complete supported source/bridge material without target values.
   Compare original observers restricted/refitted to the **same** supported
   authors and36-comment budget. Source heads: Big5-only and MBTI-only separately,
   official outer folds/inner source selection, same frozen readout family.
4. Compare GRAIN, history effect and their combination on the matching accepted
   GRAIN subset. A support restriction is not a whole-population improvement.
   Freeze all source-side choices before bridge outcomes. Direct LLM Big5/MBTI
   scoring is a separate still-open research control, not these two initiation
   questions. No new LLM weight training or paid API use.

This protocol addresses model-side context effects, not human trait causality,
temporal test-retest or clinical assessment validity. A null or support failure
is evidence about this explicitly defined candidate, not the whole GRAIN theory.

## Additional material safety gate, before any real history forward

The original normalized-text check rejects exact duplicates, not near copies.
Keep the initial material registry intact and independently audit all selected
histories. Also flag raw-text word-four-gram overlap for own-current versus its
history, and matched-history versus any target comment. A pair is flagged when
both texts have at least12 words, share at least8 distinct four-grams, and either
Jaccard is at least.8 or containment of the shorter four-gram set is at least.9.
These are declared heuristic copy controls, not a semantic paraphrase detector.
They can flag shared boilerplate as well as quotation; preserve their separate
diagnostic records and make no claim that all paraphrases have been excluded.

Create a strict registry excluding whole flagged authors without rematching or
changing their texts. No new source result chooses this threshold. Subsequent
history qualification and comparisons must identify whether they use the original
supported registry or this strict registry; primary extraction uses the strict
registry. All baseline/source/readiness intersections must be recalculated on the
same retained authors, not borrowed from a larger population. Independent lineage
and matching reconstruction is distinct from full independent tokenizer replay;
report coverage of each audit explicitly, then test token layouts on the pinned
technical panel before scaling.

## Token-boundary instrument repair, 2026-09-06

The first full material pass (r1, body-only history interval) supported4551 of7167
target authors;2569 were rejected by prefix/suffix/current-token equality. An
outcome-blind hashed panel of64 rejected authors per source group (192 total)
identified a boundary-bookkeeping problem: punctuation can merge with the two
newlines after history, so one native token crosses the interval while another
prompt leaves the separator in its suffix. Enclosing the **exact existing two
newlines** in both history intervals restored all192 authors/3456 arm-task layouts,
with zero native token changes. This is not a prompt rewrite or a loosened
equality gate. The current excerpt, template and matching rule remain unchanged.

Retain r1 and its independent audit. The next complete preparation uses an
explicit `--history-envelope body_and_separator`, recorded in the manifest, in a
new r2 directory. Its invisible history mask includes these same separator tokens;
this convention must be kept in qualification and every real forward. Repeat the
complete support and independent matching/copy audit; panel recovery alone does
not establish full support. No labels or real model responses chose this repair.
