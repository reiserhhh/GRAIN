# Explicit reporting-source sieve: diagnostic follow-up

2026-09-13. Frozen after the initial NLI-only classification was observed, before
parsing this packet or constructing the sieve. This is a developmental repair,
not a new independent validation sample; original NLI results stay unchanged.

The435-record NLI run completed870 pairs. Among400 authored quotes, original
wrappers yielded172correct/28unresolved; written wrappers50correct/34unresolved/
16wrong; explained wrappers100unresolved. Thus prompt paraphrases expose a real
failure of the generic NLI quote-source proxy. The500 native calls have now
completed the frozen NLI-only score route; do not rewrite that baseline.

## Fixed additional rule

Use existing Stanza1.11 EWT parser and source_event_graph. Identify a report verb
outside the quote but governing a quoted predicate, or the nearest preceding
report verb in the first quote sentence. Inspect that verb's source subject,
not the first-person pronouns inside the quoted body. Use the existing REPORT
vocabulary. A direct first-person-singular reporting subject gives self, a
direct other/generic subject gives other. Retain cue/source character addresses.
Reject locally negated, modal, embedded-source or passive/ambiguous reporting
cases rather than invent an actual event or unique source. Unresolved explicit
cues stay unresolved; no explicit cue falls back to the original frozen NLI
classification. Never consult designed owner, target identity or native score.

Run on all435 original records. Compare sieve versus NLI source routes, then
replay the same300 mixture scores from the existing native/PQ endpoints. No new
native model calls, no labels/MEPS fitting, no changes to the old function.
Every omitted record remains in the source ledger. No selected self evidence
means null observed score, never0. Natural35 source labels are descriptive only.

## Predecessor alignment

-Pareti2016 PARC3.0 §3.1 separates source/cue/content rather than treating a quote
 as the document author's belief: https://aclanthology.org/L16-1619.pdf .
-Muzny et al.2017 §5.1 uses speech verbs and dependent subjects in a staged
 quote-to-mention then mention-to-speaker procedure:
 https://aclanthology.org/E17-1044.pdf .

This is a narrow source-adapter borrowing the explicit-cue principle, not a
replication of those systems, a full coreference resolver or their novel-corpus
evaluation. No paper code or restricted corpus is copied. Keep self/other source,
experiencer, endorsement and factuality distinct. Follow-up success decides
whether the explicit-source adapter is useful, not whether all source attribution
has been solved. Files go under ownership_scoring_r1/sieve_v1/.
