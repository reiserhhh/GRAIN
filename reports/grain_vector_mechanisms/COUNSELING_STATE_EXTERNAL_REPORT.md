# Counseling state connection: M30C

2026-09-09. Complete. The all-factor extension finds a clear state-aligned
external signal in the existing46-person free AI conversation data. This packet
uses a frozen local observer with no model training or questionnaire calibration.

## Main result

| Actual adapted composite | Free chat vs post-state r [Fisher95% CI] | Endpoint BH q | Direct MEPS answers vs mid-state r |
|---|---:|---:|---:|
| Negative affect,8 selected items | .427 [.156,.638] | .0031 | .184 |
| Dejected feelings,5 selected items | .536 [.291,.715] | .0003 | .005 |
| Combined negative feelings,13 items | .518 [.268,.703] | .0003 | .047 |

Free-chat Spearman rho=.514/.595/.590 in the same row order. Participant-bootstrap
intervals are [.236,.607], [.363,.677] and [.320,.681]. These are actual participant
questionnaire links, not agreement between two machine readouts. The three
composites overlap, so this is not three independent replications of one claim.

Direct answers are written for assigned scenarios, with AI assistance allowed;
their implied feelings need not be the author's actual state. The contrasting
result helps locate evidence opportunity. The direct and free-chat primary
criteria are different occasions (mid versus post), not an interchangeable
accuracy contest on one identical target. All cross-occasion links are retained.

## Does this reflect the current occasion?

For the SAME free-chat readout, relation to the preceding mid-state is much
weaker than relation to the following post-state:

| Composite | Mid-state r | Post-state r | Paired difference95% bootstrap CI | Partial r(post, readout | mid) |
|---|---:|---:|---:|---:|
| Adapted NA8 | .166 | .427 | [.015,.512] | .408 |
| Adapted dejected5 | .204 | .536 | [.062,.626] | .512 |
| Combined negative13 | .145 | .518 | [.139,.631] | .526 |

The three partial correlations retain BH q=.0054/.0005/.0005 within this follow-up.
Partial r uses one covariate and43 residual degrees of freedom. It is a
descriptive association, not a new fitted personality predictor.

Readout vs post-minus-mid change also correlates positively: r=.319/.382/.437,
BH q=.0307/.0132/.0072. These are observational within-session changes, not
causal effects of AI counseling and not predictions of future mood. There is
one free-chat episode per person here, not a longitudinal clinical forecast.

The mechanistically useful interpretation is **occasion-sensitive evidence
use**: the same scoring computation follows the state near the text more than
the person's earlier score. It expands the theory beyond stable author traits
without needing to decide that every detected signal is a new personality factor.

## Construction and scoring

- Use only the participant's own text; no assistant replies, translations,
  questionnaire responses, future ratings or fictional-character attribution.
- All46 complete texts in each view fit the observer context. No clipping or
  topic optimization;3 fixed queries x46 authors x2 views =276 calls,188.97seconds.
- Questions describe the administered item content and ask about expressed
  state, not diagnosis or a lifelong trait. Native score is the fixed0-9
  conditional digit expectation divided by9, with candidate mass preserved.
- Human scores are strict means on the actual1-6 response scale. Item1-8 are
  the local PANAS_NA selection; item9-13 the local POMS_SHORT selection. These
  are **adapted composites**, not canonical scale norms or separately validated
  latent factors. No item reversals in this administered selection.
- Baseline is before worry recall; pre follows recall; mid follows MEPS; post
  follows free chat. All46 x4 x13 responses are present. Descriptive alpha across
  occasions ranges .843-.903 for NA8, .803-.837 for dejected5 and .888-.933 for
  combined13. Alpha does not establish construct validity by itself.
- Complete matrix:3 queries x12 phase/composite targets x2 views =72 links,
  plus6 intended change links. Frozen prediction bytes precede association joins.

## Equations and next mechanism step

Let s_uqc be the frozen native query score and Y_uk,t the actual mean of the
administered items at phase t. The measured objects are

```
L_qk(c,t) = Corr(s_uqc,Y_uk,t)
D_qk(c) = Corr(s_uqc,Y_uk,after - Y_uk,before)
P_qk(c) = Corr(s_uqc,Y_uk,after | Y_uk,before).
```

These do not require retraining and do not conflate state and trait. M30B is
constructing all25 narrower-target affine functions (22 relational/session plus
these3) on the same two independent PANDORA reference authors. Its next test is
whether the frozen P/Q functions retain these state associations in the cached
MEPS vectors, and which mean/configuration/final terms carry them. No46-person
model or human-outcome coefficient will be fitted to rescue that transport.

## Files

Plan: `docs/GRAIN_MEPS_MOOD_EXTENSION_M30C.md`.
Runner: `scripts/run_grain_mood_external.py`; scoring:
`grain_research/vector_mechanisms/mood_catalog.py`.
Timing comparison: `scripts/analyze_grain_mood_timing.py`.
Local outputs: `results/grain_vector_mechanisms/counseling_mood_m30c_v1/` and
`results/grain_vector_mechanisms/counseling_mood_timing_m30c_v1/`.
Only aggregate findings enter tracked documentation. Participant rows, latent
states and study source files stay local and ignored.
