# GRAIN V8: LLM-native Evidence-Carrying Assessment

Status: `V8_CANONICAL_GEOMETRY_CONFIRMED__CURRENT_BEHAVIOR_OBJECT_STOP`
This document reports technical experiments only. It reports no new
personality, clinical, human-state, reliability, or external-validity result.

The V8.0 technical scaffold passed an independent adversarial audit after four
hardening rounds. This is a contract/security result only, not scientific
validation.

## 1. Purpose

V8 extends GRAIN from an operator-indexed text geometry into an
**LLM-native, evidence-carrying psychological measurement compiler**.

The intended system may use black-box representations internally, but every
human-facing interpretation must be backed by a frozen measurement object,
declared support, joint uncertainty, positive and counterevidence,
counterfactual tests, and an evidence license. A fluent LLM narrative is not
measurement evidence.

V8 is not a system that asks an LLM to directly decide a person's personality.
It separates:

1. representation and discovery;
2. structured semantic observation;
3. deterministic aggregation and statistical measurement;
4. evidence-bound explanation;
5. candidate-only adaptation; and
6. independent audit and human authority.

## 2. Measurement Object

For author \(u\), occasion \(t\), and text unit \(i\), let

\[
D_u=\{x_{uit},c_{uit},m_{uit},\tau_{uit},h_{uit},o_{uit}\},
\]

where \(x\) is text, \(c\) observed condition, \(m\) the available opportunity
menu, \(\tau\) time, \(h\) interaction history, and \(o\) the observation
operator.

A versioned observation becomes

\[
z_{uit}^{(o,v)}
=
\left[
E_v(O_o(x_{uit})),
T_{\psi,v}(x_{uit})
\right].
\]

\(E_v\) is a frozen vector representation. \(T_{\psi,v}\) is a frozen LLM
semantic transducer that emits schema-constrained observation events. It is
not the final personality scorer.

The conditional response model is

\[
c_{uit}\sim q_u(\cdot\mid m_{uit},h_{uit}),
\]

\[
z_{uit}\sim
\mathcal F_{\nu,o}
\left(
\theta_u,s_{ut},B_u;c_{uit},m_{uit},h_{uit}
\right).
\]

- \(\theta_u\): relatively stable author configuration. It is not called
  personality until repeated-measurement and external-construct gates pass.
- \(s_{ut}\): session-bounded or short-timescale state.
- \(q_u\): selection tendency within a logged opportunity menu.
- \(B_u\): response operator under repeated randomized fixed conditions.
- \(h_{uit}\): interaction absorption, alignment, resistance, repair, and
  preceding-turn history.

The output object is

\[
\mathcal M_u=
(g_u,\widehat q_u,\widehat B_u,\widehat s_{ut},U_u,\mathcal C_u),
\]

where \(g_u\) is V7-compatible relative geometry, \(U_u\) joint conditional
uncertainty, and \(\mathcal C_u\) an explanation certificate. Components that
are not identified by the study design must be absent with a refusal code.

## 3. Four-Layer Architecture

### Observation

Native turns, comments, semantic blocks, fixed windows, and whole documents
remain separately indexed operators. No operator is universally privileged.

### Representation

Deterministic features, embeddings, and LLM semantic events are versioned
measurement channels. Every channel freezes its runtime, schema, prompt,
decoding policy, and support domain.

The V8.0 machine contract additionally binds the prompt, observation schema,
deterministic aggregator, active bundle, candidate bundle, promotion ledger,
and auditor attestation to repository-relative paths and verified SHA-256
digests. The LLM event schema has a closed observation-level field allowlist;
personality, diagnosis, and person-level score fields cannot enter the
aggregator.

Repository-internal hashes establish integrity, not authority. V8 therefore
requires the caller to supply an active-bundle trust-root hash outside the
candidate contract. Auditor attestations, language licenses, evidence-status
ledgers, and clinical sign-offs use Ed25519 signatures checked against a
caller-supplied authority registry. A builder cannot promote a replacement
bundle or self-issue validation status merely by recomputing local hashes.
Signed licenses bind the contract and bundle; evidence ledgers additionally
bind the evidence-root and complete claim-hash set; clinical sign-offs bind
the certificate, claim hashes, and evidence root. Auditor attestations bind
the contract plus active/candidate hashes. This prevents authorization replay
across versions.

### Measurement

Final aggregation is deterministic code. It estimates only declared objects:
relative geometry, author configuration candidates, session state, selection,
fixed-condition response, or interaction dynamics. It cannot silently merge
them into one personality score.

### Explanation

The interpreter reads an `EvidenceGraph`, not unrestricted artifacts or raw
analysis logs. Each sentence must bind to an `ExplanationCertificate`.
Explanation is a rendering of evidence, not free inference.

V8.0 uses controlled renderer templates. Free-form psychological or clinical
sentences cannot be disguised as technical claims. Psychological and clinical
language licenses and evidence-status ledgers are separately hash-bound;
clinical rendering also requires a recorded, timestamped human sign-off
artifact.

## 4. LLM Roles

| Stage | LLM may | LLM may not |
| --- | --- | --- |
| Discovery | Query neighborhoods, contrast vector groups, propose anonymous candidates, generate falsification worlds | Inspect confirmation labels and then select a direction; equate readability with validity |
| Semantic observation | Emit frozen structured events at observation level | Emit the final personality, diagnosis, or clinical score |
| Explanation | Render positive evidence, counterevidence, uncertainty, alternatives, and license boundaries | Invent unrecorded numbers, causal stories, or trait labels |
| Adaptation | Compile a new request into a proposed operator, estimand, null, MDE, and candidate bundle | Mutate the active bundle, open a lockbox, or promote its own candidate |
| Audit | Attack a separately produced candidate with independent code/model context | Audit an output produced by the same agent identity |

This adds `R-SEMANTIC-TRANSDUCER` to the existing GRAIN role architecture.
The active scorer remains deterministic.

## 5. Explanation Certificate

For claim \(k\),

\[
\mathcal C_{u,k}=
(\widehat m,CI,S,e^+,e^-,
\Delta_{\mathrm{remove}},
\Delta_{\mathrm{preserve}},L,H).
\]

The fields are the measured value, interval, support status, positive evidence,
counterevidence, necessity test, sufficiency test, language license, and
artifact hashes.

An explanation must pass:

1. **evidence binding**: every statement cites recorded evidence IDs;
2. **counterevidence presence**: absence of counterevidence is itself explicit;
3. **necessity**: removing cited evidence changes the claimed measurement more
   than removing matched random evidence;
4. **sufficiency**: preserving only cited evidence retains a registered portion
   of the measurement;
5. **stability**: claim content is evaluated across source, prompt, model, and
   run perturbations;
6. **license**: psychological or clinical language is refused until its
   independent evidence coordinates pass.

Evidence nodes, sentence bindings, and claims carry canonical hashes. The
validator requires a closed chain from claim to typed node, source span,
actual evidence file, measurement field/value, and provenance. Necessity and
sufficiency are paired comparisons:
their registered confidence intervals must exclude zero, rather than merely
beat an unmatched point estimate.

## 6. Joint Uncertainty

V8 extends V7.5-X2:

\[
U=
U_{\mathrm{source}}+
U_{\mathrm{representation}}+
U_{\mathrm{LLM-run}}+
U_{\mathrm{prompt}}+
U_{\mathrm{reference}}+
U_{\mathrm{model}}
+2\sum_{a<b}\operatorname{Cov}(U_a,U_b).
\]

A total interval must come from joint nested draws. V7.5-X2 observed a median
joint/independence SEM ratio of about 1.37 in its frozen technical geometry,
so independent variance addition is not licensed as a default.

## 7. Self-Iteration Without Self-Validation

An adaptation agent may create a new candidate version. It cannot alter the
currently active instrument. Candidate promotion requires:

- confirmation data hidden during proposal;
- explicit operator, estimand, null, MDE, and support rules;
- replayable model/prompt/schema hashes;
- shadow evaluation against the active instrument;
- independent audit; and
- human authorization for lockbox opening or applied interpretation.

The active and candidate manifests have distinct verified hashes; the
candidate names the active hash as its parent. The active before/after hashes
must remain identical, and a candidate-only promotion ledger must record no
authorization event. A boolean assertion of immutability is insufficient.

Audit independence is bound to agent, session, runtime, implementation, and a
separate attestation artifact. Different display names alone do not constitute
an independent audit.

Therefore “LLM can iterate GRAIN” means it can generate and test versioned
candidates. It never means that a deployed measurement silently changes.

## 8. V7 Inheritance

Retained:

- operator indexing and target-local reference geometry;
- context and choice as measured channels rather than default residuals;
- support refusal, estimand registry, artifact hashes, and append-only ledgers;
- author-disjoint fitting, empirical nulls, cross-fitting, and independent
  audit;
- joint nested uncertainty; and
- anonymous coordinates until external evidence supports naming.

Not inherited as established:

- a universal best slicing operator;
- V7 geometry as personality;
- a mean-invisible personality axis;
- PANDORA coordinates or norms outside PANDORA;
- technical consistency as psychological reliability; or
- LLM confidence, attention, rationale, or agreement as mechanism evidence.

V7.5-X1 retains anonymous cross-fit-clean path candidates but removes their
earlier strong interpretation. V7.5-X2 estimates conditional technical error,
not person-score reliability.

## 9. Formal V8.1-V8.4 Evidence Status

The full currently executable technical battery was run on 2026-07-24.

V8.1 recovered the planted event codebook with macro-F1 0.9981, 200/200 valid
schema outputs, minimum run ICC 0.9734, and minimum prompt/model CKA 0.9992.
However, maximum standardized run drift was 0.1196 against a frozen maximum of
0.10. The semantic transducer is therefore limited to candidate
evidence-rendering use. This is not overridden by its high classification
accuracy.

V8.2 passed planted evidence precision, recall, necessity, sufficiency,
mechanism sensitivity, null uncertainty refusal, and graph-tamper refusal.
V8.3 recovered the known theta, state, opportunity-conditioned choice,
fixed-condition response, and interaction-history objects and refused all
registered non-identified designs. These results validate the technical
formalism in known worlds only.

V8.4 used label-free author-relative endpoints in PANDORA, Essays, MEPS, and
multilingual X text. The semantic channel did not improve frozen PANDORA V7
geometry: delta AUC -0.0368, 95% bootstrap CI [-0.1658, 0.0895]. A
corpus-local MEPS endpoint improved by 0.1061 [0.0076, 0.2273], but every
real-text run CKA fell below 0.70, including MEPS at 0.5534. The local MEPS
signal is therefore a follow-up observation, not a promoted measurement
result.

The empirical conclusion is narrower than the architecture: deterministic
geometry, evidence contracts, and identified-design decomposition remain
viable; the current free-running LLM semantic numeric channel is not stable
enough for scoring. It may render evidence under a controlled certificate but
cannot be the active measurement channel.

### Constrained interpreter follow-up

The follow-up removed LLM numeric scoring entirely. A closed LLM observer
coded explicit behavior events, deterministic code compiled registered
SUICA/event candidates, an LLM rendered only those candidates, and an
independent critic audited them. The planted formal gate passed, including
observer macro-F1 0.943, multilabel Fleiss kappa 0.988, and key-evidence
deletion effect 0.479 [0.403, 0.556].

On formal PANDORA, runtime and repeated-input stability passed, but the
real-text measurement gate did not. Fourteen registered links all collapsed
to `novelty_expression`; only 3/80 held-out half-profiles were
candidate-eligible; same-author AUC was 0.500 and neighbor AUC 0.525. Candidate
retention was exactly 1.000, so the LLM interpreter added no demonstrated
selection information beyond deterministic compilation. The valid conclusion
remains renderer-only. See
`reports/V8_INTERPRETER_STABILITY_REPORT.md`.

### Geometry-object correction

The subsequent label-free bridge showed that the failure was upstream of
interpretation as well as downstream of it. The frozen sorted landmark
distances had effective rank 1.446 and 82.8% of their variance in PC1.
Sorting treated the distance profile as an anonymous multiset and discarded
which structurally distinct landmark generated each distance. The registered
geometry-to-behavior bridge consequently stopped at cross-modal author AUC
0.493; the behavior view, based on only three segments and six event codes,
was also weak at self-author AUC 0.551.

V8 therefore distinguishes **nuisance identity** from **structural identity**.
Raw landmark array position is nuisance identity, but a landmark's intrinsic
distance fingerprint can define anonymous structural identity when the
fingerprints are unique. For landmark internal-distance matrix \(D\),

\[
f_i=\operatorname{sort}\{D_{ij}:j\ne i\},\qquad
\phi_{\mathrm{can}}(x;D)=P_{\pi_D}d(x).
\]

This coordinate is invariant to landmark input permutations and common
Euclidean isometries. The implementation refuses unresolved fingerprint
collisions. In the current bundle all 16 fingerprints are unique.

An opened-panel, post-hoc scale-residual audit produced cosine author AUC
0.774 and author-cluster estimate 0.796 [0.687, 0.886], compared with 0.498
for the sorted multiset. Its paired delta over sorted geometry was +0.280
[0.154, 0.387], and topology-coupling shuffling reduced AUC by 0.271 to a
mean of 0.503. Scale-matched negatives retained AUC 0.754. These results
support the mathematical mechanism that stable author information resides in
relative landmark-coupling shape rather than only global radius.

Because the panel had already been opened under earlier V8 estimands, this is
an exploratory method result. It is not personality evidence and cannot
replace the frozen V7 object until a fresh registered author panel confirms
it. See `reports/V8_CANONICAL_ORBIT_GEOMETRY_REPORT.md`.

The registered internal confirmation then excluded the full 240-author V7
eligible set and sampled new PANDORA authors. Eighty-nine frozen-geometry-ready
authors produced cosine AUC 0.691 and author-cluster estimate 0.689
[0.636, 0.740], versus 0.497 for sorted distances. The paired improvement was
+0.190 [0.118, 0.261]; topology shuffling removed 0.190 AUC and
scale-matched negatives retained 0.625 AUC. The ready-support Wilson lower
bound was 0.492. This internally confirms the structural-identity mechanism
for new PANDORA authors while leaving psychological validity and independent
corpus replication open.

## 10. V8 Technical-Core Completion

The V8 technical core is not closed until:

1. model, prompt, schema, representation, operator, reference, and explanation
   objects are frozen and replayable;
2. planted worlds identify \(\theta,s,q,B\) separately and wrong designs
   refuse;
3. every output carries support and joint uncertainty;
4. explanations pass evidence, necessity, sufficiency, and adversarial tests;
5. adaptation produces candidates only; and
6. an independent implementation reproduces the headline decisions.

The internal fresh-author geometry gate is now closed. Behavior-bridge work
may resume as a bounded no-new-LLM diagnostic, but psychological
interpretation still requires separately identified external evidence.

That bounded reconnection has now been executed. Canonical geometry
self-author AUC was 0.735, while the unchanged explicit behavior view remained
at 0.556. Cross-modal AUC reached only 0.547 [0.447, 0.616], with zero stable
registered links. The evidence therefore separates the two layers:
structural author geometry is internally confirmed, but the current
three-segment/six-event behavior object is not a sufficient measurement
channel. The renderer remains downstream and cannot repair this information
deficit.

The next behavior experiment changed the measurement object rather than the
renderer. Fourteen exact-span atomic events were observed over twelve comments
per author half. Atomic observation was repeatable, but exact-subreddit
residualization failed because most condition cells contained one author and
the target author's own observations entered the fitted condition mean. A
leave-one-author-out estimator corrected this leakage but remained null.

The opened behavior-v2.1 candidate therefore uses a hierarchy:

\[
\text{evidence spans}
\rightarrow
\text{atomic events}
\rightarrow
\text{coarse behavior families}
\rightarrow
\text{author-relative configuration}.
\]

Atomic events remain the evidence layer. Stable superordinate families are
the candidate scoring layer. Two observer runs are averaged instead of
turning disagreement into absence. On the opened panel this object reached
AUC 0.699 [0.554, 0.798], with both individual repetitions above 0.66 and
cross-model macro-F1 0.771. A nested comment-count curve increased
monotonically from 0.589 at three comments to 0.699 at twelve.

This does not establish condition-invariant response. Condition/opportunity
selection itself reached AUC 0.928, matching lacked overlap, and adding coarse
behavior did not improve it. V8 therefore retains two distinct future
estimands:

\[
Q_u(c)=P(\text{opportunity/context}=c\mid u)
\]

and

\[
R_u(y\mid c,O)=P(\text{explicit behavior}=y\mid c,O,u).
\]

Natural Reddit text currently supports a joint author-relative
choice-and-expression profile. Separating \(Q\) from \(R\) requires real
condition overlap or randomized fixed opportunities. No geometry bridge or
psychological interpretation is licensed from the opened result.

The next identification sequence is frozen rather than tuned further on the
opened panel. First, two blind human coders and an adjudicator establish
whether the explicit event hierarchy and span bindings are accurate. Second,
the unchanged hierarchy is replicated on authors excluded from every earlier
V7/V8 panel. Third, C1 and C2 are estimated separately: C1 as selected
condition/opportunity distribution, and C2 as cross-fitted response under
supported shared conditions. If Reddit cannot provide at least 70% complete
condition-crossed author profiles, the correct result is
`C2_NOT_IDENTIFIABLE_IN_REDDIT`, not evidence that C2 is absent.

This sequence preserves the role split: the LLM observes and binds evidence;
deterministic code aggregates, compares, gates, and scores. Details are fixed
in `docs/V8_BEHAVIOR_V21_HUMAN_AND_FRESH_PROTOCOL.md`.

The subsequent no-call support audit sharpened this boundary. Fresh
joint/C1 replication is feasible with a frozen 190-author thread-disjoint
panel. Exact-condition C2 is not feasible in the current Reddit data: five
supported subreddits covered only 15.8% of discovery observations, and no
confirmation author had all four required crossed cells within those shared
conditions. Hash-balancing unrelated contexts would improve cell counts but
would not identify response under the same condition. C2 therefore moves to
fixed-condition data rather than being approximated by another residualizer.

The fixed-condition estimator has now been tested independently in planted
worlds. For shared condition coordinates \(z_c\), it separates

\[
\Delta_{u,h,g}(c)
=
a_{u,h,g}+b_{u,h,g}^{\mathsf T}z_c
\]

into two non-interchangeable outputs. A raw fixed-quota, calibration-selected
ridge surface is the stable score. An unpenalized Firth link estimator is the
operator-inference object. Standard errors may govern intervals and refusal
but cannot be score coordinates: an explicit SE-only attack was strongly
author-identifying.

The final 500-world battery passed all 70 gates. At binary SNR 0.50, the score
AUC was 0.951 and the Firth operator achieved median cosine 0.843, distance
Spearman 0.660, recovery slope 0.993, and calibrated analytic/bootstrap
coverage. C1-only choice was independently strong but null in the C2 score,
and private or inconsistent condition coordinates refused output. This closes
technical recoverability under the registered planted design; it does not
identify the five behavior families, a human psychological construct, or a
clinical interpretation. Details are in
`reports/V8_BEHAVIOR_C2_PLANTED_WORLD_REPORT.md`.

The subsequent hierarchy experiment adds a group quotient to the response
operator:

\[
B_u=B_0+G_{k(u)}+\varepsilon D_u.
\]

Its stable cross-half covariance is

\[
C_{\mathrm{stable}}=\Sigma_G+\varepsilon^2\Sigma_D.
\]

After discovery-only group centering, the remaining stable covariance tends
to zero as \(\varepsilon^2\). The planted battery confirmed this law and also
proved that ordinary same-author AUC is not an individual-difference
estimand: a group-only world produced AUC 0.864 against all strangers but
0.500 against strangers from the same group.

V8 must therefore report three comparisons separately:

\[
A_{\mathrm{author-all}},\qquad
A_{\mathrm{author-within-group}},\qquad
A_{\mathrm{group}}.
\]

Residual magnitude is insufficient without cross-half persistence.
Single-observer persistence is also insufficient because an observer-only
artifact produced within-group AUC 0.867 but cross-observer AUC 0.499. Values
below the registered individuality detection limit must be reported as below
detection, not forced into group or noise categories.

The experiment did not identify whether stable group geometry is a discrete
mixture or a continuous manifold. Cluster recovery and predictive fit are not
topological proof. See `reports/V8_VANISHING_INDIVIDUALITY_REPORT.md`.

The follow-up topology-margin battery treats this as a distinct population
estimand. After discovery-only stable projection, it replaces raw distances
with empirical-CDF ordinal dissimilarities, preserving Vietoris-Rips
filtration order while removing scalar distance-distribution cues. Exact H0,
landmark H1 persistence, graph diagnostics, and cross-view persistence feed
an open-set decision with a mandatory grey refusal region.

In independent planted confirmation, the revised V1.1 rule reached minimum
clear-world accuracy 0.894 at N=320 and 0.926 at N=640, refused every
registered ambiguous primary world, and had primary wrong-decisive and null
one-sided 95% upper bounds of 0.00597. Conditional affine/permutation
invariance was 0.974 and 0.982. This is a technical identifiability result
within a narrow planted margin: at noise ratio 0.25 the rule refused all clear
worlds. It does not imply that a real dynamic axis is a personality construct
or that real author populations have a discrete, ring, or curve topology.
See `reports/V8_TOPOLOGY_MARGIN_REPORT.md`.

`STOP` ends a method branch when it cannot exceed its empirical null, adds no
cross-fitted information, produces unfaithful explanations, drifts beyond
calibration, or leaks confirmation evidence.

`REFUSE` applies to an individual output when support is insufficient,
runtime identity is missing, requested components are unidentified, schema
parsing fails, uncertainty exceeds the decision margin, or the requested
psychological language lacks an evidence license.
