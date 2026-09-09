# GRAIN navigation round 3: evidence operation and query identity

2026-09-09. NAV goal remains **ACTIVE**. These packets locate which parts of the
text-to-judgment computation already transport and which the shallow programs lose.

**Finding:** a function constructed on synthetic events follows natural-author
evidence changes across twelve held/evaluation-role PANDORA authors. Extraversion
response r=.928-.971; low-state r=.860-.990; T/F r=.646-.912. These are correlations
of paired **model responses**, not human personality-prediction accuracy. The
function still receives actual P/Q from the full prefix. This is evidence for a
portable local reader and a concrete target for the independent upstream program.

## 1. Natural evidence response: NAV-04

The twelve authors came from an existing exposed evaluation-role source, not a
new lockbox or construction set. Each has twelve excerpts. A frozen lexical rule
selected the first first-person proposition, without model attribution or labels.
The original excerpt body is retained inside explicit wrappers: my experience vs
a friend's, endorse vs do not endorse, today vs several years ago. Removal drops
the same selected excerpt. Other excerpts remain unchanged. These are controlled
counterfactual contexts, not verified facts about the author; endorsement wrapping
is not identical to a linguistic negation transformation.

288 native/PQ collections took432.94seconds. Three targets, eight conditions and
twelve authors were retained without success-based filtering. Synthetic/real
functions were unchanged. Independent candidates made288 shallow calls in
133.95seconds. Bootstrap resamples the12 authors, not individual edits.

| query | axis | r | r_lower | r_upper | mae | mean_abs_native_delta | mean_abs_readout_delta | sign_agreement |
|---|---|---|---|---|---|---|---|---|
| TF_cont | actor | 0.8216 | 0.4570 | 0.9549 | 0.0128 | 0.0175 | 0.0222 | 0.7500 |
| TF_cont | endorsement | 0.9124 | 0.7655 | 0.9736 | 0.0115 | 0.0164 | 0.0263 | 0.9167 |
| TF_cont | removal | 0.6462 | 0.0570 | 0.9681 | 0.0392 | 0.0726 | 0.0589 | 0.8333 |
| TF_cont | time | 0.8143 | 0.4343 | 0.9553 | 0.0050 | 0.0076 | 0.0092 | 0.7500 |
| adapted_dejected5 | actor | 0.8666 | 0.7533 | 0.9627 | 0.0054 | 0.0139 | 0.0115 | 1.0000 |
| adapted_dejected5 | endorsement | 0.8601 | 0.7485 | 0.9630 | 0.0055 | 0.0070 | 0.0062 | 0.6667 |
| adapted_dejected5 | removal | 0.9074 | 0.6277 | 0.9795 | 0.0145 | 0.0271 | 0.0269 | 0.7500 |
| adapted_dejected5 | time | 0.9898 | 0.9787 | 0.9973 | 0.0019 | 0.0054 | 0.0037 | 0.9167 |
| extraversion | actor | 0.9336 | 0.7299 | 0.9847 | 0.0023 | 0.0062 | 0.0070 | 0.6667 |
| extraversion | endorsement | 0.9614 | 0.9007 | 0.9953 | 0.0013 | 0.0134 | 0.0134 | 1.0000 |
| extraversion | removal | 0.9276 | 0.7380 | 0.9850 | 0.0084 | 0.0212 | 0.0209 | 0.7500 |
| extraversion | time | 0.9711 | 0.9144 | 0.9947 | 0.0012 | 0.0061 | 0.0063 | 1.0000 |

The strongest gain in knowledge is not just level agreement: changing evidence
changes the fixed function in the same direction/pattern as the native observer.
Keep magnitude and sign columns: low-state endorsement r=.860 has only8/12
matching signs and its signed mean differs; correlation alone would hide this.

## 2. What the independent program loses

The token-only and four-block contextual operators are frozen round1/2
candidates, not retrained on these edits. All1/4/8-depth results remain in the
artifacts; the four-block comparison below was fixed before this packet.

| query | axis | r | mean_abs_native_delta | mean_abs_readout_delta | sign_agreement |
|---|---|---|---|---|---|
| TF_cont | actor | 0.3457 | 0.0175 | 0.0072 | 0.5833 |
| TF_cont | endorsement | 0.2000 | 0.0164 | 0.0022 | 0.7500 |
| TF_cont | removal | 0.0084 | 0.0726 | 0.0110 | 0.6667 |
| TF_cont | time | 0.0655 | 0.0076 | 0.0008 | 0.7500 |
| adapted_dejected5 | actor | 0.2302 | 0.0139 | 0.0005 | 0.6667 |
| adapted_dejected5 | endorsement | 0.6996 | 0.0070 | 0.0005 | 0.5833 |
| adapted_dejected5 | removal | -0.1529 | 0.0271 | 0.0022 | 0.5000 |
| adapted_dejected5 | time | 0.6329 | 0.0054 | 0.0005 | 0.5000 |
| extraversion | actor | -0.4440 | 0.0062 | 0.0015 | 0.3333 |
| extraversion | endorsement | 0.0266 | 0.0134 | 0.0008 | 0.9167 |
| extraversion | removal | 0.0874 | 0.0212 | 0.0047 | 0.5000 |
| extraversion | time | -0.3254 | 0.0061 | 0.0006 | 0.7500 |

Most shallow effects have too little amplitude. For low-state endorsement/time,
four-block response r=.700/.633 is present, but absolute effects are about.0005
against native.0070/.0054. T/F's useful static human association from round1 does
not imply evidence reasoning equivalence. Next develop the early-to-late
conditional computation, rather than a better human-label regressor.

## 3. Query wording and response coding: NAV-05

A genuine2x2 instrument factorial used the same96 worlds: binary Big5-E vs binary
MBTI-E/I wording, each answered with0/1 or A/B (positive=1/B). Four question-token
regions each received **their own** two synthetic references.456 full-model calls
(including reference gradients) took107.92seconds. Native and own-reference
predictions were scored without reusing another question's G coordinates.
Original continuous Big5 and original MBTI instruments remain unchanged.

| query_a | query_b | axis | r | mae |
|---|---|---|---|---|
| big5_binary_digits | big5_binary_letters | levels | 0.9107 | 0.1874 |
| mbti_binary_digits | mbti_binary_letters | levels | 0.3986 | 0.6650 |
| big5_binary_digits | big5_binary_letters | actor | 0.8476 | 0.0306 |
| mbti_binary_digits | mbti_binary_letters | actor | -0.1161 | 0.0215 |
| big5_binary_digits | big5_binary_letters | opportunity | 0.8487 | 0.0607 |
| mbti_binary_digits | mbti_binary_letters | opportunity | 0.8206 | 0.0297 |
| big5_binary_digits | big5_binary_letters | realization | 0.6287 | 0.0506 |
| mbti_binary_digits | mbti_binary_letters | realization | 0.6647 | 0.0650 |

Held MBTI candidate mass stays essentially1, yet letter scores are about.718-.919
and digit scores about.088-.280. This is an output-coding effect, not parsing loss.
Construction-only calibration in logit coordinates separates a prior from evidence:

~~~text
d[q,c](x) = logit_positive(x) - logit_negative(x)
p[q,c](x) = sigmoid(d[q,c](x))
offset[q] = mean_construction(d[q,letters] - d[q,digits])
p_cal[q,letters](x) = sigmoid(d[q,letters](x) - offset[q])
~~~

Canonical decomposition: logodds_analysis_r2. Fixed offsets are1.9828(Big5) and
3.1654(MBTI). Held probability MAE between codes falls .1874 -> .0509 and
.6650 -> .0530. This calibrates one instrument to another, not to human truth.
Big5 logit-level code agreement is r=.9702; MBTI is .3904.

The residual is informative: MBTI actor-edit logit-response agreement is r=-.171,
while opportunity-response agreement is .885. A constant output prior explains
much of the score shift but not all query-specific evidence computation.
Target/coding identity therefore belongs inside the mathematical function.

Prompt-format sensitivity motivates this separation; see
[Sclar et al.](https://arxiv.org/abs/2310.11324). The numeric decomposition is our
local observer result, not an implication of that paper.

## 4. Time labels versus presentation order: NAV-07

Six event pairs contain negative and positive evidence. Factorially vary which
event is today versus six months ago, and independently reverse display order.
Two scenes construct the new present-focused question's references; four scenes
are held.24 worlds x2 questions plus18 reference calls =66calls,19.00seconds.
Event time is an experimental fact, not a synthetic personality label.

| query | readout | axis | r | r_lower | r_upper | mean_native_delta | mean_readout_delta | mae |
|---|---|---|---|---|---|---|---|---|
| adapted_dejected5 | R_real | negative_now | 0.5570 | -0.6339 | 0.8680 | 0.0080 | 0.0930 | 0.1001 |
| adapted_dejected5 | R_real | reverse_display | 0.5706 | -0.6656 | 0.9808 | -0.0207 | -0.0317 | 0.0477 |
| adapted_dejected5 | R_synth | negative_now | 0.6303 | -0.8499 | 0.9306 | 0.0080 | 0.0772 | 0.0879 |
| adapted_dejected5 | R_synth | reverse_display | 0.6872 | -0.7537 | 0.9750 | -0.0207 | -0.0188 | 0.0363 |
| adapted_dejected5_now | R_new | negative_now | 0.8543 | 0.6691 | 0.9838 | 0.0365 | 0.0426 | 0.0384 |
| adapted_dejected5_now | R_new | reverse_display | 0.7514 | 0.6174 | 0.9565 | -0.0250 | -0.0405 | 0.0304 |

The present-focused function follows time-assignment effects at r=.854
[.669,.984] and order effects at.751[.617,.957]. Its native mean absolute
time/order changes are.079/.057; order is not negligible. Native mean
time-by-order interaction is-.082 for the present-focused question and-.103 for
the general query. The input relation is **query x event time x presentation
order**, not one universal recency weight.

This is a four-held-scene local result. The earlier46-person paired post-state
material comparison remains a separate observational link.
[Prompt permutation sensitivity](https://aclanthology.org/2022.acl-long.556/)
motivates checking order, but concerns demonstration ordering, not our event-time
manipulation; the mechanisms must not be conflated.

## 5. Mathematical update and next decision

~~~text
text + target + actor/endorsement/opportunity/time/order
  -> contextual state H_k
  -> target-conditioned evidence operation T
  -> P_hat,Q_hat (route A) OR z_hat1,z_hat2 (route B)
  -> fixed affine function
  -> instrument output scale/prior
  -> separate, frozen human external connection
~~~

Next implement a construction-only early-state residual operator (constant,
affine and one fixed quadratic kernel), generate actual P/Q coordinates, and
immediately repeat held-world, natural-edit and real-author comparisons. This asks
whether missing conditional operations can be represented by a reduced state map.
No full-model training, human-label tuning or MEPS fitting is needed.

NAV-04,05,07 now have actual first-packet closure. Useful-state construction,
real GRAIN group responses, fixed author norms, exact-prefix cache and runtime
packaging remain active. Earlier positive findings remain in force.

## Artifacts and replay

Private artifacts under results/grain_vector_mechanisms/navigation_v1:
- NAV-04/natural_evidence_r1: frozen source/edit designs, native/PQ and references.
- NAV-04/natural_programs_r1: independent token and contextual predictions.
- NAV-04/natural_evidence_r1/analysis: paired effects/components, metrics and CIs.
- NAV-05/query_identity_r1:2x2 instruments, own references and comparison matrix.
- NAV-05/query_identity_r1/logodds_analysis_r2: construction offsets/held diagnostics.
- NAV-07/temporal_order_r1/analysis: time/order factorial and interaction terms.
- round3_evidence.json: aggregate source hashes and report-code provenance.

Replay: run_grain_evidence_packets.py prepare/collect/evaluate with
--packet identity|evidence|time; evaluate_grain_evidence_programs.py;
analyze_grain_evidence_packets.py; analyze_grain_query_logodds.py.
Analysis requires a fresh output directory rather than overwriting completed
evidence. No raw author text, token arrays, keys or labels are published.
