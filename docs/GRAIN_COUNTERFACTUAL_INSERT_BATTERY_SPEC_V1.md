# Counterfactual-insert reader battery: specification V1 (PREPARED, NOT AUTHORIZED, NOT RUN)

Date 2026-09-26 JST. Prepared under seminar-06 item 3: "Prepare an exact question, existing material/reader
compatibility, finite call table and decision rule if it would unblock a source demand." No forward was run.
Running it needs a separate supervisor packet.

## Source demand it would unblock

- **P03 p.26.** Vector-path reader audit: "insert counterfactual evidence ... ask whether ΔY_reader changes along a
  repeatable internal path." It is one of seven reader edits.
- **Ledger.**
  - R14 unmet: "counterfactual insert BLOCKED; no repeatable internal path".
  - R33 unmet: "counterfactual insert (needs forwards)".
  - F14/content_coverage_r1 lists it as the only reader edit that is **absent but executable**. Language is
    human-stage, H03.

## Exact question

Take real-text units. Append ONE declared evidence sentence at the end of each unit.
- **Q1 (direction).** Does the reader's E readout move toward the inserted pole, relative to a length-matched
  neutral insert?
- **Q2 (ownership).** Is the move smaller when the same content is attributed to someone else?
- **Q3 (dominance).** Is a single sentence able to move the readout by ≥ .5 of the natural between-author SD? This
  would be a construction vulnerability.
- **Q4 (internal path).** For the Qwen observer, is the move carried by a repeatable site, L12.H0 (the reverse-trace
  V2 reader)?

## Material and reader compatibility (existing, local, no download)

- **Units.** 40 of the R17 natural units: the first 20 PANDORA and the first 20 Essays by the frozen R17 manifest
  order (`R17/natural_sensitivity_r1/manifest.csv`). Held-split authors; access is already logged in F00, and no
  new label read is needed.
- **Readers.**
  - E5-large-v2 and BGE-large, with the R17 IPIP-30 and GRAIN-24 heads. The heads are cached, and R17's base-vector
    replay matched within FP32.
  - `cross-encoder/nli-deberta-v3-base`, the F08/F16 NLI reader with the E pole query.
  - The local Qwen2.5-1.5B-Instruct observer (`load_observer`, offline, MPS FP32) with the persona E question and
    L12.H0 hooks.
  - All are in the local cache.
- **Insert bank** (frozen before any forward, 2 paraphrases each):
  - P+ own ("I really enjoy meeting lots of new people at parties.");
  - P− own ("I prefer spending my evenings alone and quiet.");
  - N own (length-matched, trait-irrelevant);
  - P+ other ("My friend says she really enjoys meeting lots of new people at parties.").
- **Natural scale.** The R17 between-author SD per reader, already computed.

## Finite call table (cap enforced before calls; no retry after a failure)

| Tier | Reader | Units × conditions | Forwards |
|---|---|---|---|
| 1 | Qwen observer (readout + L12 head activations captured in the same pass) | 40 × (original + 4 inserts) | 200 |
| 1 | NLI DeBERTa (E pole query) | 40 × 5 | 200 |
| 1b | E5 + BGE encoders (originals cached from R17) | 40 × 4 × 2 | 320 |
| 2 (only if Q1 passes for Qwen) | Qwen causal patch of L12.H0 (insert → original) for P+ and P− | 40 × 2 | 80 |

The Tier 1 load-bearing budget is 200 Qwen forwards, the "~200-forward" battery. NLI and encoder forwards are small
local encoder calls, which R17 counted separately. The total hard cap is 800. Estimated wall time is under 10 min
locally.

## Decision rule (declared)

- **Q1.** Paired over units: ΔY(P+) − ΔY(N) > 0 and ΔY(P−) − ΔY(N) < 0, with 95% unit-bootstrap CIs excluding 0.
  If both hold, the reader is insert-sensitive in the declared direction. If either CI includes 0, the result is
  inconclusive, not insensitive.
- **Q2.** ΔY(P+ other) < ΔY(P+ own), with the CI excluding 0. If so, the reader qualifies ownership. Otherwise
  ownership is not qualified, and a construction rule requires ownership filtering before the reader.
- **Q3.** Median |ΔY(P+) − ΔY(N)| ≥ .5 natural SD means single-sentence dominance. Construction then requires a
  per-sentence influence cap, or evidence aggregation before reading.
- **Q4.**
  - Tier 1: the L12.H0 linear attribution explains ≥ .5 of ΔY with a consistent sign in ≥ 80% of units.
  - Tier 2: patching L12.H0 transfers ≥ .5 of ΔY. If so, "repeatable internal path".
  - Otherwise "not localized", which does not mean absent.
- **What each outcome changes.**
  - R14/R33 "counterfactual insert" moves from ABSENT to EXECUTED at the scope of this finite panel.
  - Q2/Q3 failures become construction rules.
  - Q4 decides whether R14's "repeatable internal path" can be claimed for one reader.
- **Out of scope.** No psychological validity claim follows either way. The inserts are declared lexical evidence,
  not person behaviour.
