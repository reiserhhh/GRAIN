# VM06: same-person fixed/free text and AI-role separation

2026-09-06. **Complete frozen comparison; no psychological-label inference.**

## What was measured

The [frozen design](../../docs/GRAIN_VM06_FIXED_FREE_V1.md) was committed as
966d4ce; ea553c1 verified the legacy nonlearned position buffer before private
extraction. The complete material is46 people,138 direct answers and126 chat
views, not the RC packet's44 last replies. All1,600 original turns and366,385
characters are covered by1,609 existing lossless windows. Local multilingual
E5 uses fixed512-position float32 forwards, `query: `, masked mean and L2
normalization. Each window is independently encoded; this is not full-dialogue
contextual attention, an LLM personality judgment, or a new GRAIN factor fit.

User/assistant speakers remain separate. Primary aggregation weights normalized
window vectors by own-text token counts. Sensitivity pools original turns
equally. The three direct answers are also pooled with equal question weights.
No MEPS psychological values are parsed; mixed Markdown containers are scanned
and hashed for allowed direct-answer sections. Text and vectors remain local.

For each condition pair, cosine S_ij compares person i with person j.
The cosine gap is mean same-person minus mean other-person similarity.
The percentile reports how often the same person is closer than other people,
averaged over both query directions. **It is not Big5 r or MBTI AUC.**
Intervals resample46 participant nodes1,000 times, not46x45 independent edges.
P-values permute complete author links1,999 times; Holm covers all seven primaries.

## Primary results

| User-only condition pair | Cosine gap [95% node CI] | Same-person percentile [95% CI] | Holm p | Length-matched gap |
|---|---:|---:|---:|---:|
| Q1 / Q2 | .003368 [.001360,.005632] | .5713 [.5202,.6306] | .0035 | .001618 |
| Q1 / Q3 | .004221 [.002356,.006438] | .5802 [.5368,.6303] | .0035 | .001920 |
| Q2 / Q3 | .006017 [.003379,.008815] | .6297 [.5741,.6946] | .0035 | .003005 |
| Q1 / free | .002480 [.001038,.003960] | .5565 [.5202,.5969] | .0045 | .001456 |
| Q2 / free | .001984 [-.000018,.004182] | .5570 [.5028,.6182] | .0250 | .000980 |
| Q3 / free | .002625 [.001108,.004204] | .5536 [.5077,.5963] | .0050 | .001405 |
| All fixed answers / free | .002465 [.001287,.003730] | .5908 [.5474,.6384] | .0035 | .001593 |

The differences are small. All seven one-sided permutation tests pass the frozen
Holm family, but Q2/free's two-sided bootstrap gap interval crosses zero. These
are different inferential procedures, not seven proven personality effects.
Five-neighbor length matching reduces the gaps, often substantially. Its entries
are descriptive point estimates, not separately confirmed effects. Length,
writing habits, repeated subject matter and AI assistance remain explanations.
Equal-turn all-fixed/free percentile is .5937 [.5499,.6402], close to .5908.

## Why role handling matters

| Comparison, token pooling unless noted | Same-person percentile [95% CI] |
|---|---:|
| Fixed user answers / free user text | .5908 [.5474,.6384] |
| Fixed user answers / free assistant text | .5225 [.4775,.5642] |
| Fixed user answers / naive mixed free text | .5336 [.4871,.5781] |
| Free user / its own assistant text | .9942 [.9839,.9995] |
| Fixed user answers / equal-turn mixed free text | .5787 [.5258,.6308] |

The near-perfect within-conversation match most plausibly reflects shared
conversation content and feedback. It cannot support personality measurement.
Associated MEPS answer/assistant matches are also .6983-.7246 on25-26 people;
these answers may incorporate AI help and must not be called independent retests.

Assistant text contributes a mean87.04% of content tokens in free conversations.
With token pooling, mixed-vector cosine is .99931 with assistant-only, versus
.96854 with user-only. This closeness is partly a predictable property of the
weighted mean itself, not a discovered causal mechanism. Equal-turn weighting
reduces this mechanical imbalance (.99115 versus .98857), but does not establish
that mixed authorship is valid. The lower cross-task mixed percentile is an
observed contrast, not a separately tested superiority claim.

## Contribution and boundary

This gives an operational separation of **cross-task author correspondence**
from **within-conversation content correspondence**. The former is modest; the
latter can be nearly perfect without measuring stable human psychology. GRAIN
must retain role and condition provenance when forming author observations.
Do not remove topic or AI context wholesale: retain them in separately named
context channels, while the author feature channel stays speaker-explicit.

This result complements RC1-RC5: context has a measurable computational effect,
but context-driven matching and author validity are not interchangeable. No
external-label association, clinical inference, same-day/cross-day trait
stability, cross-language invariance or personality-specific direction is shown.

## Reproduction and audit

Private run: `results/grain_vector_mechanisms/vm06_fixed_free_r1`, identity
`ac077428fbd6d8b73c5c39f934c955b7d4b86cda24aa739077bcefccdf64e51b`.
1,617 forwards total (8 public +1,609 private),269.33 seconds, sampled MPS2.242GB.
Public repeat/padding replacement errors0; compact/fixed max component error
1.30e-7 (threshold1e-5). The only legacy loading discrepancy is a verified
integer-arange position buffer, not a missing learned weight.

`audit_grain_meps_fixed_free.py` independently replays1,394 pooled view vectors,
32 condition matrices, seven primary permutations (13,993 draws) and Holm.
Max pooling error8.33e-17; matrix9.99e-16; point statistic3.33e-16; p error0.
It verifies original window/checksum lineage; it does not independently repeat
neural extraction or all bootstrap draws. The bootstrap has synthetic node tests.
14 focused tests passed; the combined MEPS and RC5 regression selection passed34
tests. Neither is the entire historical repository test suite.

```bash
python scripts/run_grain_meps_fixed_free.py \
  --input-root "$PERSONA_DATA/MEPS+AI_conv_experiment" \
  --preflight-dir results/grain_vector_mechanisms/vm06_meps_preflight_r3 \
  --windows-dir results/grain_vector_mechanisms/vm06_meps_windows_r3 \
  --output-dir results/grain_vector_mechanisms/FRESH_MEPS_RUN
python scripts/audit_grain_meps_fixed_free.py \
  --run-dir results/grain_vector_mechanisms/FRESH_MEPS_RUN \
  --windows-dir results/grain_vector_mechanisms/vm06_meps_windows_r3 \
  --output-dir results/grain_vector_mechanisms/FRESH_MEPS_AUDIT
```

No new layer/prompt search is authorized by this result. Proceed to the current
Essays/X domain packet and complete source-head comparisons; bridge and VM07 remain open.
