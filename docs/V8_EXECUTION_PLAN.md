# GRAIN V8 Execution Plan

Status:
`V8_ROUTING_OPERATOR_SYNTHETIC_PASS_WITH_INFERENCE_CAVEATS__BLIND_LOCALIZATION_CANONICAL_PASS__REAL_TEXT_CLOSED`
Baseline: sealed `v0.2.1`; V7.5-X1/X2 are post-tag research results on `main`.

The integrated route and current evidence boundaries for all later
mathematical work, including V3.7G/H, H.4D/R2, residual completion, and M3,
are maintained in `docs/V8_MATHEMATICAL_RESEARCH_ROUTE.md`. This execution
plan preserves the chronological experiment record; the route archive is the
authoritative cross-line synthesis.

Formal execution date: 2026-07-24. All currently executable V8.1-V8.4
experiments are complete. V8.5 remains a future human repeated-measurement
study and was not simulated.

The reconciled mathematical object used by subsequent work is defined in
`docs/V8_CONDITIONAL_RESPONSE_GEOMETRY_THEORY.md`.

## V8.0 Contract and Reconciliation

Deliverables:

- V8 theory and role boundary;
- machine-valid assessment contract;
- explanation certificate;
- refusal tests for LLM self-audit, final-LLM scoring, unidentified \(q/B/s\),
  missing counterevidence, and self-modifying adaptation;
- append-only reconciliation of README/evidence state with V7.5-X1/X2.

Exit gate: contract tests and an independent adversarial re-audit pass. The
required attacks are LLM person-score copy-through, fake deterministic
aggregator, fake bundle immutability, auditor aliasing, dangling evidence/hash
chains, free-form clinical language disguised as technical output, and
JSON-Schema/Python-validator structural divergence. The canonical Draft 2020-12
schema is executed inside the Python/CLI entry gate, and the deterministic
aggregator separately enforces event types, finite ranges, and source-span
shape. No scientific claim is made when this gate passes.

The repository template is intentionally not executable as-is. `DECLARE_*`
values and all-zero hashes must return `UNRESOLVED_PLACEHOLDER`; replacing
them with frozen run identities is part of experiment registration.

Contract-only smoke:

```bash
python scripts/validate_suica_v8_contract.py \
  --contract configs/v8_assessment_contract.smoke.json \
  --expected-active-sha256 \
    22f71780a9eb4da8ac8d50a9fb2ba84a689d692e23b1de1064203d9634f398ba \
  --trusted-authorities configs/v8_trusted_authorities.smoke.json
```

The bundled authority registry and active hash are synthetic test fixtures.
For a real candidate, both trust inputs must be supplied by the release
controller outside the candidate workspace.

## V8.1 Frozen Semantic Transducer

Implement a provider-neutral transducer that records:

- provider, model ID and immutable revision when available;
- exact prompt and schema hashes;
- decoding parameters and repeated calls;
- raw-response hash, parse status, and structured event output;
- API/runtime errors as refusals rather than missing-zero scores.

Experiments:

- same-input variation across run, prompt paraphrase, and model family;
- schema/parse attack;
- assistant-echo and role-confusion attack;
- semantic event recovery in planted text worlds.

Decision: if LLM-run/prompt/model variance cannot be calibrated, the LLM
remains a report renderer and is not a measurement channel.

## V8.2 Vector Query and Explanation Fidelity

Implement restricted vector tools for:

- nearest-neighbor and matched-neighbor contrast;
- local direction traversal;
- group comparison under frozen reference;
- evidence occlusion;
- matched random occlusion;
- paraphrase/topic/format counterfactuals.

The LLM receives tool results, not unrestricted claims about raw vector
coordinates. It produces an `EvidenceGraph` and certificate.

Primary gates:

- cited-evidence removal has larger registered effect than matched random
  removal;
- evidence-only preservation retains a registered effect;
- sentence-level claims are entailed by artifact fields;
- explanation survives harmless paraphrase but responds to planted mechanism
  changes.

## V8.3 Measurement Decomposition in Simulation

Plant independent and coupled worlds for:

- stable author configuration \(\theta\);
- session state \(s\);
- opportunity-conditioned choice \(q\);
- fixed-condition response \(B\);
- interaction history \(h\);
- topic, format, length, and assistant echo.

Require component recovery and explicit refusal under absent exposure menus,
single occasions, non-random conditions, hidden partner input, and model
drift.

## V8.4 Label-Free Real-Text Pilot

Use existing corpora only to test:

- technical semantic-transducer stability;
- incremental cross-fitted information beyond V7 deterministic geometry;
- explanation fidelity and refusal behavior;
- target-local reference requirements.

Do not open new personality, symptom, behavior, or clinical labels.

If the LLM channel adds no cross-fitted information, retain it only for
evidence translation. This is a useful boundary, not system failure.

## Formal V8.1-V8.4 Result

### V8.1

The first 200-call run used `max_tokens=8192` and produced 183/200 valid
outputs. Fifteen of the seventeen refusals were provider `length`
terminations. The complete attempt is retained under
`results/v8_full/v8_1_semantic_attempt1_8192/`.
Its exact config is preserved as
`configs/archive/v8_full_experiment_v8_1_attempt1_4a99f85cd0.json`.

The registered rerun changed only the output capacity to 16384 tokens:

- parse rate: 1.000 (200/200);
- macro-F1: 0.9981, bootstrap lower bound 0.9965;
- worst-event F1: 0.9933;
- attack-safe rate: 1.000 and null false-positive rate: 0.000;
- minimum run ICC: 0.9734;
- minimum prompt/model CKA: 0.9992;
- maximum standardized run drift: 0.1196, above the frozen 0.10 gate.

Decision: `V8_1_DOWNGRADE_RENDERER_ONLY`. Accurate planted-event recovery is
not sufficient to license a stochastic LLM numeric channel.

### V8.2 and V8.3

Both planted technical worlds passed:

- evidence precision/recall were 1.000 in the planted evidence world;
- necessity advantage was 0.3684 with lower bound 0.3650;
- sufficiency median was 0.9962 and mechanism-flip detection was 1.000;
- the identified independent world recovered theta geometry at r=0.9969,
  state at r=0.9755, choice probabilities at r=0.8349, response operator at
  r=0.9795, and history operator at r=0.9901;
- all registered wrong-design and graph-tamper cases refused.

These are identification and evidence-contract results in known simulations,
not evidence that those components have human psychological meaning.
Their shared exact config is preserved as
`configs/archive/v8_full_experiment_v8_2_v8_3_e1450c4692.json`; both snapshot
hashes reproduce the original manifest commitments after the live V8.4 config
was expanded.

### V8.4

The label-free real-text run used 201 PANDORA, 72 Essays, 37 MEPS, and 72
multilingual X authors. It read no psychological, clinical, market-outcome, or
behavioral labels and persisted neither raw text nor raw identifiers.

- semantic segment coverage was 1.000 after complete-author filtering;
- all real-text run CKA values failed the 0.70 gate: PANDORA 0.474, Essays
  0.327, MEPS 0.553, and X 0.387;
- frozen V7 PANDORA geometry used 32 source-disjoint comments per half, with
  117 supported authors and 76/21/20 discovery/calibration/confirmation
  support;
- PANDORA frozen-geometry increment was -0.0368,
  95% bootstrap CI [-0.1658, 0.0895];
- corpus-local deltas were PANDORA -0.0594, Essays -0.0125, MEPS +0.1061,
  and X -0.0095;
- only MEPS had a positive lower bound, [0.0076, 0.2273], but its run CKA
  failed, so the result cannot be promoted as a stable measurement channel;
- formatting CKA was 0.342 for PANDORA and 0.903-0.944 for the other corpora.

Decision: `V8_4_SEMANTIC_CANDIDATE_RENDERER_ONLY`. The registered PANDORA
increment failed and real-text semantic observations were not repeatable
enough.

The final provenance verifier checked 70 frozen input, code, and inventoried
artifact hashes with zero failures: `V8_FULL_PROVENANCE_PASS`. See
`reports/V8_FULL_PROVENANCE_AUDIT.md`.

### V8.4b constrained evidence-bound interpreter

The architecture was then corrected so GRAIN remained the only numerical
measurement channel. The LLM was restricted to explicit behavior observation,
candidate-bound interpretation rendering, and independent critique.

- planted formal: PASS on all registered runtime, stability, safety,
  perturbation, critic, and key-evidence gates;
- PANDORA quick: technical runtime PASS;
- PANDORA formal: `V8_INTERPRETER_RENDERER_ONLY`;
- PANDORA candidate coverage: 3/80 held-out half-profiles;
- same-author AUC: 0.500; neighbor AUC: 0.525;
- all 14 discovery links mapped to the single
  `novelty_expression` event code;
- exact deterministic-candidate passthrough: 1.000.

The primary author/neighborhood gate failed, so secondary Essays, MEPS, and X
runs were not executed. The next target is the registered geometry-to-behavior
bridge, not a return to direct LLM scoring. See
`reports/V8_INTERPRETER_STABILITY_REPORT.md`.

### V8.4c geometry-to-behavior bridge

The registered bridge expanded the sorted-distance geometry into quantile,
stable-FPCA, location/scale/shape, and spacing-ILR views and expanded behavior
into repeated event rates, opportunity-conditioned residuals, event pairs,
and exact order-null transitions. It stopped with
`V8_BRIDGE_STOP_NO_CONFIRMATION`:

- selected cross-modal author AUC 0.493;
- geometry self-author AUC 0.466;
- behavior self-author AUC 0.551;
- distance alignment 0.012, permutation p=0.460;
- zero confirmed registered links.

The sorted 16-distance view had effective rank 1.446 and 82.8% of its variance
in PC1. This identified a representation bottleneck upstream of the renderer:
global sorting retained a radial distribution but discarded the coupling
between distances and landmark identity. The three-segment behavior view was
also under-resolved. No new LLM calls or labels were used.

### V8.4d canonical-orbit geometry audit

An invariant structural coordinate was then derived from each frozen
landmark's sorted internal-distance fingerprint. All 16 fingerprints were
unique, so the anonymous canonical coordinate exactly reproduced indexed
landmark distances while remaining invariant to landmark input order and
common Euclidean isometries. Fingerprint collisions are refused.

The post-hoc scale-residual candidate removed query-wide radius and retained
the shape of relative proximity to structurally identified landmarks. On the
already opened confirmation panel:

- cosine AUC 0.774; author-cluster estimate 0.796 [0.687, 0.886];
- sorted-distance cosine AUC 0.498;
- paired delta over sorted +0.280 [0.154, 0.387];
- paired delta over spectral energy +0.197 [0.088, 0.312];
- scale-matched hard-negative AUC 0.754;
- topology-coupling shuffle drop 0.271, p=0.0020;
- canonical/indexed equivalence error 0 and maximum invariance error
  \(3.68\times10^{-14}\).

Decision: `V8_CANONICAL_SCALE_RESIDUAL_EXPLORATORY_PASS`. This is an
opened-panel method result, not lockbox confirmation and not personality
validity. The representation must pass a fresh registered author panel before
it can replace the frozen V7 object or reconnect to the behavior interpreter.
See `reports/V8_CANONICAL_ORBIT_GEOMETRY_REPORT.md`.

### V8.4e internal fresh-author confirmation

The first proposed confirmation-only holdout was unavailable before metric
calculation: all 20 original confirmation authors that passed the frozen
geometry support gate had already entered the opened V8 panel. The failed
design is preserved under
`configs/archive/v8_canonical_geometry_fresh_panel_unavailable_confirmation_only.json`.

The corrected registered panel excluded all 240 authors in the full V7
eligible set and deterministically sampled new authors from the PANDORA raw
comment pool. It produced 89 ready authors from 156 source-eligible authors:

- canonical scale-residual cosine AUC 0.691;
- author-cluster estimate 0.689 [0.636, 0.740], p=0.00010;
- sorted-distance cosine AUC 0.497;
- paired delta over sorted +0.190 [0.118, 0.261];
- paired delta over spectral energy +0.109 [0.056, 0.161];
- scale-matched AUC 0.625;
- topology-coupling shuffle drop 0.190, p=0.0010;
- support-rate Wilson lower 0.492.

Decision: `V8_CANONICAL_SCALE_RESIDUAL_INTERNAL_FRESH_PASS`; the stricter
independent audit also passed. The result confirms an author-geometry
mechanism inside PANDORA, not personality meaning or cross-corpus transport.
The behavior bridge may now be reconnected once without new LLM calls.

### V8.4f canonical geometry-to-behavior reconnection

The confirmed geometry was reconnected to the unchanged observer cache and
behavior pipeline. Calibration selected repeated rates plus event pairs. On
confirmation:

- geometry self-author AUC 0.735;
- behavior self-author AUC 0.556;
- cross-modal author AUC 0.547 [0.447, 0.616];
- element Spearman 0.152;
- distance Spearman 0.143, p=0.073;
- zero stable links and zero distinct registered behavior targets.

Canonical geometry improved geometry self-AUC by 0.269 and cross-modal AUC by
0.053 over the old bridge, but behavior self-AUC changed by only 0.006.
Decision: `V8_CANONICAL_GEOMETRY_PASS_BEHAVIOR_VIEW_STOP`. Stop the current
three-segment/six-event object and do not tune the renderer. See
`reports/V8_CANONICAL_BEHAVIOR_BRIDGE_REPORT.md`.

### V8.4g high-resolution behavior-v2 pilot

A 24-author, 12-comment-per-half pilot replaced the old six-event object with
14 exact-span atomic events and six explicit opportunity codes. Profile IDs
were made opaque and paired author halves were separated across API calls.
The LLM remained an observer; deterministic code remained the sole aggregator.

The observer channel was technically repeatable (two-run macro-F1 0.849), but
the registered atomic condition-residual object stopped: AUC fell from 0.561
at three comments to 0.385 at twelve comments, while condition/length alone
reached 0.928. Cross-model macro-F1 was 0.621. Decision:
`V8_BEHAVIOR_V2_TECHNICAL_PILOT_STOP`.

### V8.4h behavior-v2.1 opened candidate

No-call diagnostics proved that strict event intersection and sparse
in-sample condition residualization were destructive. The corrected
leave-one-author-out atomic residual remained null (AUC 0.492), so exact
subreddit conditioning removes rather than isolates the stable signal.

A deterministic five-family hierarchy over the same evidence produced:

- soft coarse-family AUC 0.699 [0.554, 0.798], p=0.0010;
- repetition AUC 0.669 and 0.683;
- cross-model coarse macro-F1 0.771;
- nested 3/6/9/12-comment AUC 0.589/0.639/0.679/0.699.

All technical candidate gates passed, but condition matching lacked overlap
and condition plus behavior did not improve condition alone. Decision:
`V8_BEHAVIOR_V21_CANDIDATE_FROZEN_CONDITION_UNRESOLVED`. This is an opened,
post-hoc candidate only. Human coding and a fresh author panel are required;
the geometry bridge remains stopped. See
`reports/V8_BEHAVIOR_V21_CANDIDATE_REPORT.md`.

### V8.4i human-gold and fresh-panel gate

The blind human packet and its fail-closed evaluator are ready:

- 48 coder-training items;
- 336 fixed-hash natural-prevalence items;
- 192 disagreement/rare-event boundary items;
- 528 evaluation items per coder;
- 192 hidden independent-model audit predictions;
- zero additional LLM calls.

Natural items alone define headline observer accuracy. Enriched items are
restricted to ontology-boundary and recall diagnostics. Blank or incomplete
coder files return `WAITING_FOR_HUMAN_CODING`; completed independent files
produce a blinded adjudication queue; only completed adjudication can open
observer accuracy.

The subsequent fresh panel is frozen at 60 discovery, 20 calibration, and 80
confirmation authors, with 12-comment primary and 24-comment extension
profiles. It cannot start before `V8_BEHAVIOR_V21_HUMAN_GATE_PASS`. C1
selected-context behavior and C2 response-under-shared-condition are reported
as separate estimands. See
`docs/V8_BEHAVIOR_V21_HUMAN_AND_FRESH_PROTOCOL.md`.

The no-call preflight found 300/300 scanned candidates capable of two
thread-disjoint 32-comment halves and froze the complete 190-author
60/20/80/30 panel. Joint/C1 replication is therefore feasible. Exact C2 is
not: five supported subreddits covered 15.8% of discovery segments and no
confirmation author completed all four required crossed cells under those
shared conditions. Reddit C2 must report `C2_NOT_IDENTIFIABLE_IN_REDDIT`;
balanced hashes over unrelated subreddits are not a substitute for shared
conditions. After the human gate, discovery plus calibration run before any
confirmation opening. Confirmation must report coarse behavior, C1, and
joint separately; joint adds information only if its paired gain over C1 has
CI lower bound above zero. See
`reports/V8_BEHAVIOR_V21_FRESH_PREFLIGHT_REPORT.md`.

The existing MEPS session does not fill the C2 gap. Although all 46
participants answered the same three prompts, each prompt provides three
direct-answer slots and no independent repeat. Zero participants meet the
six-opportunity direct-answer minimum across all prompts; only 2/46 meet it
through optional AI-chat turns. MEPS remains a later ontology/procedure pilot,
not confirmation. See
`reports/V8_BEHAVIOR_V21_MEPS_C2_SUPPORT_AUDIT.md`.

### V8.4j C2 planted-world dual-track validation

The C2 estimator has completed a final independent seed-block battery with
500 repetitions, 999 paired permutations, 60/20/80
discovery/calibration/confirmation authors, eight crossed conditions, five
behavior families, and exactly six primary observations per cell.

The implementation freezes two distinct tracks:

- a calibration-selected ridge score over the raw fixed-quota link-response
  surface for stable same-author matching;
- an unpenalized Firth link estimator for operator recovery, uncertainty, and
  refusal.

Author-specific standard error is excluded from score coordinates. It is
used only for uncertainty and support checks. The binary SNR-0.50 result was
AUC 0.951, operator cosine 0.843, distance Spearman 0.660, recovery slope
0.993, analytic CI coverage 0.968, and bootstrap-t coverage 0.904. All 70
registered recovery and falsification gates passed, including C1 information
imbalance, intercept-only, unstable operator, private coordinates, numerator
shuffle, extreme prevalence, and SE-only attacks.

Decision: `V8_BEHAVIOR_C2_PLANTED_WORLD_PASS`. This validates estimator
identification in planted shared-condition worlds only. It does not open the
human behavior, personality, emotion, diagnosis, clinical, or geometry-bridge
claims. See `reports/V8_BEHAVIOR_C2_PLANTED_WORLD_REPORT.md`.

### V8.4k vanishing-individuality hierarchy

The follow-up planted battery introduced

\[
B_u=B_0+G_{k(u)}+\varepsilon D_u
\]

and separated author-all, author-within-group, and group pairing. All core
checks passed. Group-only structure produced author-all AUC 0.864 but
author-within-group AUC 0.500; individual-only structure produced
author-within-group AUC 0.866. Oracle residual covariance scaled as
\(\varepsilon^{1.999}\), while the finite estimator scaled as
\(\varepsilon^{1.687}\). The minimum detectable individuality under the
registered design was 0.50.

C1 group confounding, observer-specific artifacts, and half-unstable
dispersion were correctly rejected as C2 individual structure. Discrete
cluster recovery and continuous-manifold recovery each worked in their
planted worlds, but predictive reconstruction did not separate a discrete
mixture from a flexible local manifold by the registered margin.

Decision:
`V8_EPSILON_HIERARCHY_CORE_PASS_GEOMETRY_UNRESOLVED`. See
`reports/V8_VANISHING_INDIVIDUALITY_REPORT.md`.

### V8.4l open-set population topology margin

The unresolved discrete-versus-continuous question was converted into a
separate planted-world estimand. Four half/observer views of a stable
three-dimensional C2 response population were embedded in 15 dimensions.
Discovery-only projection, empirical-CDF ordinal distances, exact H0,
landmark H1 persistence, graph evidence, and cross-view stability were
combined with an explicit `TOPOLOGY_NOT_IDENTIFIABLE` outcome.

The original V1 stopped because the N=640 minimum clear-world accuracy was
0.880 and the wrong-decisive upper bound was 0.0645. Its audit exposed an
estimand error: absolute author AUC cannot be a necessary condition for
population topology. A new-seed V1.1 used author AUC only as a diagnostic,
introduced a calibrated H1 grey zone, and fitted thresholds separately by
sample size.

V1.1 passed every registered primary gate:

- minimum clear-world accuracy 0.894 at N=320 and 0.926 at N=640;
- all six ambiguous/adversarial worlds refused at rate 1.000;
- primary wrong-decisive and null-claim one-sided 95% upper bounds 0.00597;
- conditional affine/permutation invariance 0.974 at N=320 and 0.982 at N=640;
- positive H1 ring/curve refusal gaps at all four calibrated sample sizes.

Sensitivity scans sharply restrict the claim. At noise ratio 0.25 the method
refused all three clear worlds, and bridge-continuum recovery remained weak.
Decision: `V8_C2_TOPOLOGY_MARGIN_PLANTED_PASS`. This closes planted technical
identifiability only; it does not establish real-text topology or a
psychological construct. See `reports/V8_TOPOLOGY_MARGIN_REPORT.md` and
`reports/V8_TOPOLOGY_MARGIN_INDEPENDENT_AUDIT.md`.

### V8.4m response-set incidence geometry

Authors were represented as bounded affine response images
\(S_u=f_u(\mathcal Z)\) and lifted condition-response graphs
\(\Gamma_u=\{(z,f_u(z))\}\). The battery separated same-condition incidence,
free cross-condition coincidence, whole-map equivalence, near misses,
parallel sets, 3D skew lines, and line/surface/volume images.

The audited confirmation passed all executable gates: same/free AUC
1.000/1.000, relation balanced accuracy 0.9657, finite-scale dimension error
0.0863, principal-angle error 0.2147 degrees, map-equivalence graph F1 1.000,
cross-view Jaccard 0.9979, and attack refusal 1.000. A common-anchor family
produced 120 raw edges but zero map-equivalence edges, demonstrating that a
shared condition is not a pair-specific relationship.

Independent methodology decision: `PARTIAL`. Current graph excess is affine
map equivalence, not population-calibrated excess incidence; nonlinear sets,
persistent tubes, transversality, intersection dimension, and real text are
open. The method refuses all ordinary worlds from noise 0.06 onward.
See `reports/V8_RESPONSE_SET_INCIDENCE_REPORT.md`.

### V8.4t author-specific stochastic routing operator

V3.7A replaced pooled transition kernels with an anonymous author-relative
probability operator integrated over a frozen reference opportunity design.
The primary world used a fresh hidden rank-6 Haar subspace in every
population. The estimator did not receive that basis or author links during
fitting.

Across 200 sealed confirmation populations:

- truth correlation 0.862;
- split-session reliability 0.650;
- unseen-context reliability 0.824;
- within-group same-author AUC 0.963;
- held-out log-loss gain 0.0523;
- 193/200 stable author claims.

The group-only control reached ordinary author AUC 0.848 but within-group AUC
0.500 and zero claims. Full-rank 48D recovery crossed the registered
reliability gate at 128 events per context-session but not at 32.

Independent decision: `PASS_WITH_MAJOR_CAVEATS`. The accepted result is
synthetic low-rank operator recovery under a registered four-group design.
The variance-share estimator produced invalid negative/out-of-range
components, the reported interval is an empirical band rather than a formal
confidence interval, the estimator was told \(K=4\), and low-budget F48 is a
reliability threshold rather than an information-theoretic refusal.
See `reports/V8_AUTHOR_ROUTING_OPERATOR_V37A_REPORT.md` and
`reports/V8_AUTHOR_ROUTING_OPERATOR_V37A_INDEPENDENT_AUDIT.md`.

### V8.4u blind junction localization

V3.7B removed the oracle event window. A locator reading only trajectory
geometry and order detected a registered pause-plus-perpendicular-cusp
signature. Discovery selected threshold 0.30 before 200 sealed confirmation
populations.

- localization F1 0.9996;
- P95 location error 1;
- false junctions 0.0025 per 1000 negatives;
- blind-versus-oracle operator correlation 0.959;
- predictive-gain retention 1.001;
- blind split/unseen reliability 0.652/0.827;
- blind within-group AUC 0.963;
- cue-leak author claims 0/200.

Canonical decision: `V8_BLIND_JUNCTION_LOCALIZATION_V37B_PASS`; independent
decision: `PASS_WITH_CAVEATS`. The result is
a family-specific synthetic upper bound: the generator plants the same
speed-dip/perpendicular-cusp mechanism searched by the locator. Open-world
event discovery, missing-not-at-random robustness, real-text existence, and
psychological interpretation remain closed. Shared-event oracle/blind
correlation, a circular conditional-P95 gate, negligible mask stress, and
reused derived RNG seeds also require a new sealed version. See
`reports/V8_BLIND_JUNCTION_LOCALIZATION_V37B_REPORT.md` and
`reports/V8_BLIND_JUNCTION_LOCALIZATION_V37B_INDEPENDENT_AUDIT.md`.

### V8.4ab paired-schedule opportunity refusal

V3.7H.1 R4 converts opportunity aliasing into a paired, same-author design
test. Two independent technical streams are collected under each of two
registered schedules. The signed excess schedule energy is normalized by the
stable energy after the frozen score operator. Only this paired statistic can
refuse stable-author interpretation; cumulative path predictability is a
mechanism diagnostic.

In 100-repetition synthetic power calibration:

- null refusal was 0/300, with Bonferroni familywise one-sided upper .0554;
- raw effect .10 and .20 were detected in 600/600 material cells, with every
  geometry-specific one-sided lower bound .9446;
- score-space \(Q\) tracked planted score-space effect at correlation .9958,
  mean bias -.00014, and RMSE .00503;
- simultaneous minimum qualified raw effect was .05 for low-rank response and
  .02 for random-rotation and nonlinear response.

Decision: `V8_SCHEDULE_REFUSAL_V37H1_R4_POWER_CANDIDATE_PASS`. This is design
and power evidence only. Correlated/common-shock noise, pairing failure,
minority and near-kernel response, transient/reversal response, panel scaling,
real text, and psychological meaning remain open. See
`reports/V8_SCHEDULE_REFUSAL_V37H1_R4_POWER_REPORT.md`.

### V8.4ac repeated-opportunity common-shock frontier

V3.7H.2 separated technical-stream replication from repeated opportunity
occasions. The legacy stream-only correction was refuted under correlated
streams and common shock: its contaminated-null refusal rate was 92.03% in
the broad frontier and 100% in the focused worst-case audit.

The repeated-opportunity estimator subtracts across-occasion variance. D1
tested 216 cells and failed only one finite-sample familywise null upper:
2/100 refusals at \(K=2,\rho=.6,c=.05\), heteroskedastic \(t_5\). D2 used
fresh seeds, 1,000 repetitions per focused cell, added \(K=3\), and tightened
the gates. It passed:

- \(K=2\): heavy-tail null 11/1000, familywise upper .02224;
- \(K=3,4,8\): 0/1000 null refusals in both noise modes;
- all \(\eta=.10\) cells: 1000/1000, familywise lower .99494;
- maximum simultaneous bias bounds: .000633 Gaussian and .000584 heavy-tail;
- global shift: total 1000/1000 versus author-centered 0/1000;
- persistent response/confound identity error: 0.

Legacy contamination decayed as \(K^{-1}\), while repeated-estimator null
standard deviation contracted approximately as \(K^{-1.12}\). Decision:
`REPEATED_OPPORTUNITY_IDENTIFICATION_PASS_WITH_REPLICATION_BOUNDARY`.
\(K=2\) is the validated minimum identifiable configuration, \(K=3\) the
validated minimum robust configuration, and \(K=4\) the recommended
confirmation default. Pairing/missingness,
minority/near-kernel response, transient reversal, real text, and psychology
remain open. See
`reports/V8_COMMON_SHOCK_IDENTIFIABILITY_V37H2_REPORT.md`.

### V8.4ad multiscale zero-identification calculus

V3.7H.3 formalized component "vanishing" as a registered amplitude and
aggregation path rather than assuming that a component literally disappears.
The balanced additive synthetic model separated society, nested group,
author, condition, author-by-condition response, opportunity, and technical
terms.

Across 60 repetitions in Gaussian and heteroskedastic \(t_5\) worlds:

- K=2/3/4 minimum stable recovery \(R^2\) was .911/.941/.956;
- K=2/3/4 minimum split-panel correlation was .849/.894/.918;
- maximum off-diagonal symmetric zero-path leakage was .000743;
- independent author and group aggregation slopes were -1.0003 and -.9999;
- group-common and society-common slopes were approximately zero;
- ICC=.30 author structure produced slope -.205, exposing a non-vanishing
  common floor;
- author-dependent condition selection reversed the author direction
  (mean \(r=-.907\)); balanced standardization restored it
  (mean \(r=.997\));
- raw group/condition projection commutator was .472 versus approximately
  zero under the balanced product reference;
- persistent response/confound identity error was zero and correctly refused.

Machine decision:
`V8_MULTISCALE_ZERO_IDENTIFICATION_V37H3_DISCOVERY_PASS`.
Scientific decision:
`DISCOVERY_PARTIAL_PASS_ADDITIVE_BALANCED_WORLD_ONLY`.
Exact reconstruction, quadratic slopes, alias identity, and the
\(p\alpha\eta\) near-kernel relation are construction checks rather than
independent discoveries. Nonadditive misspecification, dependent/non-ergodic
opportunity structure, unknown hierarchies, real text, and psychological
meaning remain open. See
`reports/V8_MULTISCALE_ZERO_IDENTIFICATION_V37H3_REPORT.md`.

### V8.4ae misspecification and anchored-cell transport refusal

V3.7H.4 deliberately fitted the additive H.3 coordinate system to low-rank
nonlinear, latent-hierarchy, and persistent opportunity worlds. Observation-
only CRC, residual-rank, and held-out-gain diagnostics used independent
panels, author cross-fitting, and Holm familywise control.

Across 100 repetitions per registered cell:

- both additive controls produced 2/100 false refusals; their one-sided 95%
  upper bounds were .06162 and failed the frozen `<.05` gate;
- W1, W2, and W3 were refused in 100/100 repetitions at both .05 and .20
  planted effect shares under both noise modes;
- strong-effect gap-closure lower bounds were at least .9325;
- strong-effect CRC means were at least .8819;
- non-ergodic residual K=8/K=2 persistence was at least .9986;
- response/opportunity observational-alias error was zero;
- all specific psychological/causal attribution remained closed.

Frozen machine decision:
`V8_MISSPECIFICATION_TRANSPORT_V37H4_REFUTED_OVERCONFIDENT_ADDITIVE`.
Independent scientific decision:
`PARTIAL_DETECTS_NOT_LOCALIZES_CALIBRATION_UNRESOLVED`.

The observed 2% false-refusal point estimates do not demonstrate error
inflation; the 100-repetition study failed to certify a strict upper bound
below 5%. The supported object is sensitivity to the registered planted
cross-panel residual structures and anchored held-out author-by-condition
cell transport, not general misspecification awareness or zero-shot condition
transport. A fresh 1,000-repetition-per-noise W0 calibration is the next
prospective gate. See
`reports/V8_MISSPECIFICATION_TRANSPORT_V37H4_REPORT.md`.

### V8.4af independent additive-control calibration

V3.7H.4C retained the exact H.4 99-permutation detector and used 1,000 fresh
additive controls per registered noise mode. It did not pool the 200 H.4 W0
trials and did not change diagnostics, Holm alpha, ranks, panels, or generator
parameters.

- Gaussian: 16/1000 false refusals, simultaneous upper .02585;
- heteroskedastic \(t_5\): 30/1000, upper .04255;
- all source-lock, row-count, seed, numeric, manifest, and inventory checks
  passed.

Decision:
`V8_MISSPECIFICATION_W0_CALIBRATION_V37H4C_CALIBRATED`.

The H.4 frozen machine STOP remains unchanged. The combined scientific result
is now: H.4 discovery sensitivity plus H.4C independent specificity
calibration for the exact registered synthetic detector. Arbitrary noise,
reference-measure drift, missing support, full-rank or localized effects,
real text, psychology, and causal localization remain open. See
`reports/V8_W0_CALIBRATION_V37H4C_REPORT.md`.

### V8.4ag reference-measure smoke refutation

The first H.4D smoke protocol attempted to classify absolute interaction
score relevance after author centering. Smoke and a 99-permutation preflight
showed that the registered projection was algebraically forced near zero for
all residuals. The protocol was stopped before full execution.

Decision:
`V8_REFERENCE_MEASURE_FRONTIER_V37H4D_SMOKE_REFUTED_IDENTIFIABILITY`.

The refutation establishes an A/Q gauge boundary: total author score is
estimable under support, but the author-constant projection of interaction
cannot be separated from the author term without an external anchor. See
`reports/V8_REFERENCE_MEASURE_FRONTIER_V37H4D_SMOKE_REFUTATION.md`.

### V8.4ah gauge-invariant reference-contrast frontier

H.4D R1 replaced the unidentifiable absolute interaction projection with a
gauge-invariant cross-reference contrast and replaced the condition-
permutation null with an author-level wild-sign null.

Across 3,760 Monte Carlo rows:

- W0 false refusals were 26/1000 Gaussian and 24/1000 heavy-tail, with
  simultaneous upper bounds .03787/.03550;
- fixed-reference correction produced RCG lower .800/.805, D-star upper
  .101/.101, and score-correlation lower approximately .995;
- contrast-sensitive mean \(D_C=.639/.647\), with 100/100 correct detections
  and classifications;
- contrast-kernel mean \(D_C=.0039/.0063\), also 100/100 correctly detected
  and classified;
- support violation was refused 100/100;
- full-rank structure was detected 100/100 without requiring the low-rank
  channel;
- minority-local non-low-rank detection failed the frozen gate at 91/100
  Gaussian and 83/100 heavy-tail.

Machine decision:
`V8_REFERENCE_CONTRAST_FRONTIER_V37H4D_R1_PARTIAL_REFERENCE_CONTRAST`.
Scientific decision:
`REFERENCE_CONTRAST_CORE_PASS_MINORITY_SENSITIVITY_UNRESOLVED`.

See `reports/V8_REFERENCE_CONTRAST_FRONTIER_V37H4D_R1_REPORT.md`.

### V8.4ai minority information frontier

H.4D R2 retained the frozen R1 detector and varied fixed versus random
minority support, global versus active-SNR scaling, and iid versus
intrinsic-zero-sum geometry across 11,000 Monte Carlo rows.

- both iid arms selected \(m=8\) as the discovery support boundary;
- fixed active-SNR CRC-or-HC power rose from .227/.240 at \(m=2\) to
  1.000/.990 at \(m=8\);
- fixed-information curves predicted random-support power within +/- .05;
- intrinsic-zero-sum geometry had comparable residual information but
  .19-.67 lower power than iid geometry.

Decision:
`V8_H4D_R2_RANDOM_SUPPORT_EXPLAINED_SCALAR_INFORMATION_GEOMETRY_REFUTED`.

Realized support explains the R1 random-split instability inside the iid
family, but scalar information is not sufficient across geometry. See
`reports/V8_MINORITY_INFORMATION_FRONTIER_V37H4D_R2_REPORT.md`.

### V8.4aj narrow iid confirmation

R2A froze active-SNR, fixed-support, iid-block, \(m=8\), and the discovery
source hashes before using 4,000 fresh Monte Carlo repetitions.

- W0 was 28/1000 in each noise mode, simultaneous upper .0402;
- iid \(m=8\) CRC-or-HC power was .997/.998, with lower bounds
  .9913/.9928;
- source lock, row count, 12,000 unique seeds, numeric integrity, and
  projection compatibility passed.

Decision:
`V8_MINORITY_INFORMATION_FRONTIER_V37H4D_R2A_PASS_IID_FIXED_M8`.

This confirms an iid operational boundary only. It does not confirm a
cross-geometry scalar threshold. See
`reports/V8_MINORITY_INFORMATION_FRONTIER_V37H4D_R2A_CONFIRMATION_REPORT.md`.

### V8.4ak geometry information operator

H.4D R2B matched scalar residual information across seven geometries and
evaluated a six-coordinate operator by simultaneous geometry-family and
base-seed holdout.

- all 12,600 rows and 1,800 paired base worlds passed integrity checks;
- iid-minus-intrinsic power-gap lower bounds were .140-.623;
- operator-minus-scalar log-loss gain was .04650
  [.03671, .05655];
- Brier gain was .02228 [.01893, .02566];
- intrinsic held-out log-loss gain was .26359
  [.23960, .28651];
- effective sign support was not separately identified;
- halo-sweep calibration failed, with maximum family error .18267.

Decision:
`V8_GEOMETRY_INFORMATION_OPERATOR_V37H4D_R2B_PARTIAL_OPERATOR_PREDICTIVE_NOT_SUFFICIENT`.

Geometry matters beyond total scalar information, but the registered finite
coordinate vector is not a sufficient operator. The next prospective step is
a resolution-indexed permutation-orbit frontier using an independent
mechanism panel. See
`reports/V8_GEOMETRY_INFORMATION_OPERATOR_V37H4D_R2B_REPORT.md`.

### V8.4al resolution-indexed permutation-orbit frontier

H.4D R2C estimated the exact finite wild-sign/Holm orbit on mechanism panels
0/1 and tested it against independent outcome panels 2/3 across 10,800 fresh
synthetic rows.

- all seven integrity checks passed;
- the m=4, lambda .03 all-author halo gap replicated at .910/.897;
- all 36 design-cell mean calibration errors were at most .0482;
- mean cell calibration error was .0154 versus .1120 for frozen R2B;
- row-level log-loss gain was -.6630 [-.7910, -.5365];
- Brier gain was -.00339 [-.00888, .00196];
- within-cell N1/N2/Ninf retest correlations were
  .628/.775/.790, despite pooled correlations of .995/.994/.960.

Machine decision:
`V8_PERMUTATION_ORBIT_FRONTIER_V37H4D_R2C_REFUTED_PERMUTATION_ORBIT_EXPLANATION`.

Scientific decision:
`GROUP_CELL_CALIBRATION_PASS / INDIVIDUAL_CONDITIONAL_PREDICTION_REFUTED`.

The finite orbit explains aggregate exchangeable-cell power but is
overconfident for a particular new panel. The next candidate must integrate
posterior uncertainty over latent interaction and new-panel measurement
noise, not clip or recalibrate orbit probabilities. See
`reports/V8_PERMUTATION_ORBIT_FRONTIER_V37H4D_R2C_REPORT.md`.

### V8.4am conditional-heterogeneity resource gate

Before fitting the proposed R2D posterior model, H.4D Gate-0 held each latent
interaction and geometry fixed while generating 32 independent detector
outcomes on each of 400 fresh bases.

- all eight integrity checks and both artifact verifiers passed;
- 12,800 independent outcome-panel detector evaluations completed;
- pooled conditional-probability variance was .002925
  [.001456, .004114];
- its upper bound crossed the frozen `.005` futility boundary;
- A16-to-B16 base prediction lost .02283 log loss and .00761 Brier score
  relative to a leave-one-base-out cell mean;
- the `lambda=.01, all64` halo cells retained the largest local variance
  estimates, but did not pass the registered simultaneous gate.

Machine decision:
`V8_R2D_GATE0_STOP_NO_EXPLOITABLE_CONDITIONAL_TARGET`.

Scientific decision:
`REGISTERED_DGP_POWER_IS_PRIMARILY_CELL_LEVEL`.

The full R2D model is paused because the current synthetic generator has too
little pooled within-cell target, not because posterior computation failed.
The next admissible direction is a fresh local halo-support heterogeneity
surface, not a 64-replicate retry or outcome-calibrated predictor. See
`reports/V8_CONDITIONAL_HETEROGENEITY_GATE0_V37H4D_R2D_REPORT.md`.

### V8.4an controlled geometry injection

The first R2E implementation was invalidated after execution. Its recovered
all-author halo used a support-4 zero-halo core while the baseline used the
support-64 RNG path. The resulting halo mixed two unrelated cores and did
not reproduce the frozen builder's normal frontier. Its machine STOP is not
scientific evidence.

R2E.1 corrected the RNG provenance, directly verified every normal geometry
against the frozen builder, and passed all 18 integrity checks. Across
33,280 fresh detector outcomes:

- normal \(J\) was .1906/.1898 and normal \(\Delta V\) was .1836/.1815;
- tangent \(J\) was -.00008/.00015;
- tangent \(\Delta V\) was .00028/.00017, pooled .00023.

The frozen nine-endpoint unstudentized simultaneous radius was about `.010`,
so the machine decision correctly remained
`V8_R2E1_INCONCLUSIVE_NO_TUNING`. The discovery showed a strong normal
channel and near-zero tangent estimates but could not certify a practical
tangent null. See
`reports/V8_CONDITIONAL_HETEROGENEITY_INJECTION_V37H4D_R2E_ERRATUM.md` and
`reports/V8_CONDITIONAL_HETEROGENEITY_INJECTION_V37H4D_R2E1_REPORT.md`.

### V8.4ao fresh normal-versus-tangent endpoint confirmation

R2E.2 used 160 fresh parents, five prospectively frozen geometries per
parent, 32 outcomes per geometry, and separate normal and tangent
Bonferroni families.

- normal \(J\) was .19689/.18546, with lower bounds .19153/.17972;
- tangent \(J\) was -.000164/-.000019;
- tangent \(\Delta V\) was .000289/.000069, pooled .000179;
- all five tangent upper bounds were at most .001740, below the `.005`
  practical threshold;
- studentized max-t sensitivity agreed;
- all three artifact manifests and inventories reverified.

Decision: `V8_R2E2_CONFIRMED_NORMAL_ONLY`.

The frozen detector can transmit deliberately planted within-cell variation
along the registered normal direction, but the tested macro-coordinate-
matched tangent rotation is practically null. This narrows the R2D
interpretation: the current generator lacks variation along
detector-relevant within-cell directions. It does not establish that every
tangent direction is null or identify any psychological construct. See
`reports/V8_CONDITIONAL_HETEROGENEITY_INJECTION_V37H4D_R2E2_REPORT.md`.
Independent mathematical audit:
`PASS / ACCEPT_V8_R2E2_CONFIRMED_NORMAL_ONLY`.

### V8.4ap constrained local response operator

R2F constructed a four-dimensional local tangent chart inside the
5,670-dimensional double-centered geometry space. Twenty-four shared-seed
probes were projected into the kernel of a nine-coordinate numerical
Jacobian; A/B outcome halves then estimated cross-fitted gradient and
curvature operators.

The first smoke incorrectly applied the new chart's `.05 SD` finite-drift
gate to the legacy R2E.2 tangent-null control. R2F.1 corrected that category
error before formal outcomes, retained the null as a separate outcome
control, and used fresh seeds.

Formal R2F.1 results:

- geometry preflight: 80 parents, 3,600 geometries, zero outcomes, maximum
  constraint residual `1.74e-15`, maximum chart drift `.03637 SD`;
- formal discovery: 115,200 detector outcomes, all integrity checks passed;
- normal \(J=.19277/.18826\), with lower bounds `.18487/.17912`;
- registered-null \(J=-.000038/.000025\), with upper bounds
  `.000214/.000290`;
- leading first-order strength `.000284`, only 5.7% of the `.005` gate,
  with principal-angle upper `67.5 degrees`;
- curvature strength `-.000304`, with principal-angle upper `59.9 degrees`;
- quadratic-adequacy upper MSE `.0000488 < .0004`.

Decision:
`V8_R2F_STOP_NO_LOCAL_TANGENT_CANDIDATE`.

Independent audit:
`PASS / ACCEPT_STOP_NO_LOCAL_TANGENT_CANDIDATE`.

No candidate or fresh confirmation is authorized. This freezes the current
synthetic tangent-search route without asserting that the remaining 5,666+
directions, higher-order effects, real text, or psychological mechanisms are
null. See `reports/V8_LOCAL_RESPONSE_OPERATOR_V37H4D_R2F1_REPORT.md`.

## V8.5 Future Human Study

Deferred until the technical core passes. Required design:

- same people across multiple bounded sessions;
- logged free-choice exposure menus;
- repeated randomized fixed conditions;
- session-level state anchors and independently replicated construct anchors;
- human review of explanation correctness and harmful overinterpretation.

This phase is the first point where reliability, state/trait separation,
psychological validity, or clinical usefulness can be claimed.

## Immediate Work Packages

| ID | Work package | Current state |
| --- | --- | --- |
| V8-WP0 | Theory, externally rooted contract, schema parity, adversarial refusal tests | completed; independent audit PASS |
| V8-WP1 | Semantic transducer runtime and drift ledger | completed; renderer-only |
| V8-WP2 | Vector-query API and evidence graph | completed; planted PASS |
| V8-WP3 | Explanation necessity/sufficiency harness | completed; planted PASS |
| V8-WP4 | \(\theta/s/q/B/h\) planted-world simulator | completed; identified-world PASS |
| V8-WP5 | Label-free real-text pilot | completed; no PANDORA increment, stability FAIL |
| V8-WP5b | Constrained evidence-bound interpreter | completed; planted PASS, PANDORA renderer-only |
| V8-WP5c | Geometry-to-behavior bridge | completed; STOP, both views under-resolved |
| V8-WP5d | Canonical-orbit geometry method audit | exploratory PASS; fresh panel required |
| V8-WP5e | Canonical geometry fresh-author confirmation | internal PASS; 89 new authors |
| V8-WP5f | Canonical geometry-to-behavior reconnection | geometry PASS; behavior object STOP |
| V8-WP5g | High-resolution behavior-v2 technical pilot | completed; atomic residual object STOP |
| V8-WP5h | Behavior-v2.1 no-call hierarchy candidate | opened candidate frozen; condition unresolved |
| V8-WP5i | Blind human-gold and fresh-panel protocol | packet/evaluator ready; fresh joint/C1 support PASS; exact Reddit C2 unavailable; waiting for human coding |
| V8-WP5j | C2 fixed-condition dual-track planted validation | completed; 70/70 gates PASS; human C2 still untested |
| V8-WP5k | Vanishing-individuality group/residual limit | core PASS; discrete-versus-continuous topology unresolved |
| V8-WP5l | Open-set C2 population topology margin | planted PASS at registered low-noise/N>=320 margin; real-text topology untested |
| V8-WP5m | Affine response-set incidence geometry | executable planted PASS; independent method audit PARTIAL; nonlinear/real-text closed |
| V8-WP5n | Distance-matched persistent incidence V3.1 | technical planted PASS; local recurrence adds information beyond integrated whole-path distance |
| V8-WP5o | Strong-chain and condition-aware pairwise fairness V3.2/V3.3 | closed as condition-aggregation information-loss theorem; no generic hypergraph superiority |
| V8-WP5p | Common-ball incremental geometry V3.4 | planted PASS beyond finite binary pair adjacency; complete distances and psychological meaning remain outside the claim |
| V8-WP5q | Continuous scale, matched null, transversality, and nonlinear sheets V3.5 | frozen gates PASS; independent PARTIAL: exact birth/null/local templates accepted, general dimension/manifold recovery not accepted |
| V8-WP5r | Spacetime junction-flow and depth-three tree V3.6 | registered-DGP PASS; tree capacity PARTIAL and explicit-time result is a construction check |
| V8-WP5s | Whole-path and static-null scope correction V3.6.1 | post-seal complete; random path MI bias removed, deterministic local composition verified, planning/intelligence still closed |
| V8-WP5t | Author-specific routing operator V3.7A | synthetic low-rank recovery PASS_WITH_MAJOR_CAVEATS; group-free inference, valid variance decomposition, and formal uncertainty open |
| V8-WP5u | Blind junction localization V3.7B | canonical family-specific PASS_WITH_CAVEATS; independent-panel, out-of-family, MNAR, group-free, and real-text localization open |
| V8-WP5v | Group-free routing transport V3.7C | sealed canonical 20/20 PASS_WITH_CAVEATS; group-free operator recovery, independent event panels, bounded transport, and registered MAR correction close; recovery and identity resolution separate |
| V8-WP5w | Dimension-density-event-budget phase diagram V3.7D | sealed confirmation 13/13 PASS_WITH_CAVEATS; recovery and identity are bidirectionally separable in the registered synthetic design |
| V8-WP5x | Adaptive rank and fixed-reference refutation V3.7E | sealed confirmation 16/16 PASS_WITH_CAVEATS; rank12 reverse separation explained by fixed-rank misspecification; contrast transport supported, absolute norms open |
| V8-WP5y | External zero, nested uncertainty, and residual sufficiency V3.7F | sealed confirmation 25/25 PASS; independent PASS_WITH_MAJOR_CAVEATS; event error practically vanishes while state/capacity floors remain, dense residual retains author information |
| V8-WP5z | Reliability spectrum and resolution filtration V3.7G | sealed 60-rep NMSE confirmation PASS_WITH_CAVEATS; all 20 coded gates passed, dense/broken NMSE reduced 28.9%/20.7%, core model-assisted latent coverage 0.957-0.966; sealed prose mislabels NMSE as NRMSE, informative-precision overcoverage and true cross-budget filtration remain open |
| V8-WP5aa | Genuine nested-prefix resolution filtration V3.7H | 30-rep discovery complete; four stationary core worlds show 79.6%-90.0% endpoint NMSE reduction, 0.298-0.437 endpoint radius ratios, and near-zero observable update predictability; opportunity drift exposes biased-center contraction and requires a cumulative path-drift/refusal layer before confirmation |
| V8-WP5ab | Projection identity and cumulative-path audit V3.7H.1 | 30-rep discovery refutes posterior-martingale equivalence at the registered +/-0.01 margin; supports a risk-contractive resolution flow with core pooled cumulative kappa 0.008-0.011 versus opportunity drift 0.151; paired-schedule refusal and matched power remain open |
| V8-WP5ac | Paired-schedule refusal and matched power V3.7H.1 R4 | synthetic design/power candidate PASS; paired-only score-space Q has 0/300 null refusals and 600/600 material detections under 15-cell familywise bounds; adversarial noise, pairing, sparse/kernel/transient response, and confirmation remain open |
| V8-WP5ad | Repeated-opportunity common-shock frontier V3.7H.2 | D1 broad frontier exposed one K2 heavy-tail tail-certification gap; D2 1000-rep focused audit PASS, legacy stream-only Q refuted, repeated Q calibrated; K2 minimum identifiable, K3 minimum robust, K4 confirmation default; seal, minority/kernel/transient response, and real text remain open |
| V8-WP5ae | Multiscale zero-identification calculus V3.7H.3 | registered discovery gates PASS; scientific PARTIAL: balanced additive finite-sample identification supported, unrestricted social-scale vanishing refuted; nonlinear misspecification and non-ergodic opportunity adversary next |
| V8-WP5af | Misspecification and anchored-cell transport V3.7H.4 | frozen STOP on W0 certification: 2/100 false refusals per noise gives upper .06162; scientific PARTIAL, with complete planted-structure sensitivity but calibration below .05 and general localization unresolved |
| V8-WP5ag | Independent W0 detector calibration V3.7H.4C | CALIBRATED for exact 99-permutation detector: Gaussian 16/1000 upper .02585, heavy-tail 30/1000 upper .04255; H.4 STOP preserved, arbitrary worlds and reference-measure transport open |
| V8-WP5ah | Absolute reference-measure projection H.4D smoke | STOP before full run: author centering forced the proposed score-projection ratio near zero and exposed A/Q gauge non-identifiability |
| V8-WP5ai | Gauge-invariant reference contrast H.4D R1 | PARTIAL: reference-contrast core, support refusal, W0 calibration, contrast-sensitive/kernel separation, and full-rank detection pass; minority-local sensitivity fails at 91/83 non-low-rank detections |
| V8-WP5aj | Minority information frontier H.4D R2 | discovery COMPLETE: iid m=8 candidate found and random-support transport explained within +/- .05; scalar information refuted across intrinsic geometry |
| V8-WP5ak | Narrow iid confirmation H.4D R2A | PASS: W0 upper .0402/.0402 and iid m=8 power lower .9913/.9928 under exact frozen detector |
| V8-WP5al | Geometry information operator H.4D R2B | PARTIAL: geometry descriptors improve LOGO log loss by .0465 and Brier by .0223, but sign support is unidentified and halo-sweep calibration error is .1827 |
| V8-WP5am | Resolution-indexed permutation-orbit frontier H.4D R2C | REFUTED for individual conditional prediction: cell calibration PASS, but row log loss worsens .663 and within-cell spectrum reliability is .628-.790 |
| V8-WP5an | Posterior-predictive orbit H.4D R2D | Gate-0 STOP: pooled base-level probability variance .002925 [.001456, .004114] is below the .005 resource bound; full posterior model paused, with a local all-author halo heterogeneity signal reserved for a fresh protocol |
| V8-WP5ao | Controlled geometry injection H.4D R2E/R2E.1 | original R2E invalidated by core RNG mismatch; corrected R2E.1 valid but machine-inconclusive under its frozen nine-endpoint interval, with strong normal and near-zero tangent point estimates |
| V8-WP5ap | Fresh endpoint confirmation H.4D R2E.2 | CONFIRMED_NORMAL_ONLY for the registered synthetic detector: normal lower bounds .180-.192, all tangent practical-null upper bounds <=.001740 under threshold .005; all-tangent and psychological claims remain closed |
| V8-WP5aq | Constrained local response operator H.4D R2F.1 | STOP_NO_LOCAL_TANGENT_CANDIDATE after 115,200 outcomes: first-order strength .000284 and unstable, curvature unresolved, controls valid; current synthetic tangent search frozen |
| V8-WP6 | Human repeated-measurement and construct study | not run; future-data gate |

## Governance

- V8 work does not modify the sealed `v0.2.1` tag or release bundle.
- Living ledgers are append-only.
- The builder cannot sign its own promotion.
- Active-bundle trust roots and authority public keys come from outside the
  candidate contract.
- An adaptation agent may output only `PROPOSED`.
- External labels and clinical language remain closed until separately
  registered evidence gates pass.
