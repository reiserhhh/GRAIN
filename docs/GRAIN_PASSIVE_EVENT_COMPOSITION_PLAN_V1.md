# HM55: Passive Event Composition

## Scientific Decision

Stop improving injected source-vector fidelity. Determine whether a frozen
reader distinguishes repeated descriptions of one occasion from descriptions of
distinct occasions. Connect passive reading to GRAIN's existing versioned event
ledger, while preserving narrative emphasis rather than discarding duplicates.

This is not a predictor contest, a new deduplication algorithm, or evidence that
the model's judgments are psychologically correct.

## Frozen Design

- Observer: unchanged local Qwen2.5-1.5B, float32, MPS, eager attention, block 16.
- Six queries: Big Five plus anthropomorphism, unchanged standard digit code.
- Fresh controlled accounts: two behavioral families per query, focal polarity
  low/high, both display orders, eight history regimes.
- 24 isolated-event reference calls, then 384 complete-account calls: 408 total.
- Maximum 600 acquisition seconds. No fitting, patching, generation, API,
  private data, model selection, or additional parameter search.
- Full block states, raw candidate logits, exact token addresses and records are
  retained. All materials/code/specification are frozen before acquisition.

Regimes: balanced; alias_focus (same focal occasion described three times);
distinct_focus (three focal occasions); other_focus (two additional occasions
belong to B); revision; revision_alias; retracted; other_only.

The balanced account contains focal behavior of polarity p and other-family
behavior of polarity 1-p. Alias and distinct versions have identical action
words and report counts: only the occasion identifiers differ. Test actual
token IDs/offsets instead of presuming matched lengths. Order reversal changes
display order, not declared occasion identities. Corrections/retractions always
follow their parents. Explicit identities are design metadata, not automatically
discovered events in real text.

## Two Different Objects

Let H be the report history, E(H) the current qualified A-event ledger, and
s(e) the frozen isolated-event reader score for the same family and polarity.
Define a controlled reference, not a personality truth or Bayesian posterior:

    event_profile(H) = sum_{e in E(H)} s(e) / |E(H)|
    mention_profile(H) = sum_{m in A-action mentions of H} s(m) / |mentions|

Aliases add a narrative mention but no independent event. Revision replaces one
claim; old mentions remain in the narrative history, explicitly marked stale.
Retraction removes the active claim, not the historical text. B's actions remain
in the account but are not A's behavioral events. No A evidence returns null,
not zero. The second channel is a descriptive mention profile, not valid current
action evidence; all discourse, qualification and ordering remain available.

Exact interface checks: event_profile(alias)=event_profile(balanced),
event_profile(other)=event_profile(balanced), and
event_profile(revision_alias)=event_profile(revision).
With focal f and other o, distinct minus balanced is (f-o)/4;
alias minus balanced has the same change only in mention_profile.

Native whole-account readouts are *measured*, not constrained to equal either
profile. Compare score and high/low log odds, paired alias-minus-balanced,
distinct-minus-alias, other-minus-balanced, revision_alias-minus-revision,
retracted-minus-balanced, and both-order differences. Report all targets and
polarities, including weak/reversed isolated evidence. Hidden-state displacement
at the query endpoint is descriptive; cosine overlap is not a causal circuit.

## Interpretation and Stop Rule

If alias and distinct act differently, occasion identity participates in passive
judgment beyond identical behavior word multiplicity. If they act similarly,
report multiplicity may dominate this reader/protocol. Either answer determines
whether GRAIN must separate event evidence from narrative emphasis. Do not tune
the prompt, model, layer or weights to rescue the answer. Complete this finite
battery and stop; no accuracy promotion, human validity, or global completion.

Verification: focused synthetic ledger tests; exact material/token checks;
408 saved actual forwards; scalar and ledger recomputation via a separate
verifier with zero neural calls. Same-agent arithmetic verification is not
independent experimental replication.

## Prior Art and Mathematical Inspiration

- Mergeable Model-Side Aggregation States (Song et al., 2026):
  https://arxiv.org/html/2607.26448v1 . Established identity-aware set/frequency
  operators; its HLL is explicitly not new. We do not implement or claim HLL.
- BayesBench (Samanta et al., 2026):
  https://arxiv.org/html/2606.30850v1 . Evidence trajectories and dependency
  caution, not a proof of our behavioral score or a benchmark we reproduce.
- OpenAI/math fixed commit adc7f1241b42e322a6451854ab7e4b4c146bf78a,
  Finite Monoid Computations, sections algebra and compilation: deterministic
  episode updates and output tests are separable; assumptions include a finite
  monoid/state system. Those assumptions are not established for LLM states.
  This supplies a design analogy, not an applied new theorem.
- Existing GRAIN evidence_history.active_records already implements aliases,
  revisions and retractions. Existing P1N tests two-attribute interactions.
  Neither is renamed as this packet's discovery.

Candidate contribution is the measured passive-reader event/mention distinction
and its connection to a measurement interface, not any constituent algebra.
