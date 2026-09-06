# VM05 direct trait readout: protocol and deployment readiness

2026-09-06, round27. **Full material support, cross-platform token checks and
six-author real direct readout qualification are complete; source prediction
evaluation is not yet run.** This is a research
control alongside GRAIN, not a replacement for its deterministic measurement path.

## Scientific contrast

A supervised head reading a frozen vector and a frozen LLM choosing its own
trait answer are different estimators. The former can expose information that
the latter does not use, while the latter can impose learned response priors.
This control compares them on the same existing author texts, not on added
memories, different text budgets or outcome-selected prompts.

For each original common-material author, retain the three complete12-comment
views and raw/cue/placebo variants. Big5-only authors receive five continuous
trait tasks; MBTI-only authors receive four binary-axis tasks. Bridge text may
receive all nine without reading bridge psychological values. No label-guided
selection, new training, prompt search, score repair or polarity inversion occurs.

For candidate logits z, answers A and fixed numeric values v:

```
q_j = exp(z_j) / sum_{k in A} exp(z_k)
f = sum_j q_j v_j
candidate_mass = exp(logsumexp(z_A) - logsumexp(z_full_vocabulary))
df/dz_j = q_j (v_j - f)
```

Big5 uses digits0-9 with values j/9; MBTI uses I/E, S/N, T/F and P/J with positives
E/N/F/J. The local derivative in final hidden coordinates is W_A^T times the
candidate-logit derivative. This is standard softmax calculus and a local model
response direction, **not a newly discovered human personality axis**. Its
finite-difference check passed on a synthetic seven-dimensional hidden state.

Scores conditioned on A can be extreme even when almost all vocabulary
probability lies outside A. Therefore we save candidate logits, the full-vocabulary
log normalizer, conditional probabilities, entropy, candidate mass and unrestricted
greedy first-token compliance. None is a complete-response parse rate. Independent
arithmetic reconstruction uses shifted exponentials rather than the collector's
readout function. This checks saved arithmetic, not the truth of a model output.

Big5 digit expectations are not human percentiles; MBTI conditional scores are
not calibrated probabilities of a person's type. Three-view averaging is fixed,
not Bayesian posterior inference. Primary external readouts remain rank/correlation
and axis discrimination, with scale-dependent MAE/RMSE identified separately.

## Completed material and software checks

| Evidence | Result |
|---|---|
| Complete original common material |7,266 authors;624 Big5-only,6,393 MBTI-only,249 bridge text records |
| Whole-author support |7,266 supported,0 excluded; no text truncation |
| Logical task/view/arm prompts |278,397 |
| Unique exact prompts, deduplicated within author only |143,045 |
| Full token-support preparation |176.76s; no neural forwards or target values |
| Independent role/grid/count recount |All authors and prompt totals reproduced |
| Technical panel |Two Big5-only and four MBTI-only hash-selected authors;234 prompts |
| Mac / Windows token reconstruction |2.87s /6.50s; same code, cases and semantic records |
| Real pinned-CUDA qualification |234 logical prompts,86 unique complete inputs per pass,172 forwards including full replay;83.72s |
| Independent native-to-derived arithmetic |234 rows, max error3.55e-15; native repeated logits/readouts exactly equal |
| Scoped direct tests |10 passed, including full synthetic qualification and deliberately invalid packets |
| Wider direct/history/readout tests |51 passed; subsequent UTF-8-specific affected regression suite25 passed |

The two test counts overlap and must not be added. Existing SWIG deprecation
warnings remain. Support tests are token-only; the real qualification row is
separate from synthetic model fixtures and does not establish prediction accuracy.
In its234 repeated/related requests, unrestricted first-token compliance was1.0
and median candidate mass was.999441. These are technical descriptors, not234
independent people or semantic-correctness estimates. No natural target values
were opened to judge these six authors.

Actual preflight caught a schema mismatch: the direct material names its tokenizer
hashes `tokenizer_sha256`, unlike the history packet's field. A direct-schema
checker fixes this without editing the historical certificate. Complete support
is reconstructed from the recorded group counts, not an invented summary field.
The exported source runs independently of the full development tree. Source/readout
JSON readers also now explicitly use UTF-8, preserving Windows portability.

## Reproducible artifacts

- Material: `results/grain_vector_mechanisms/vm05_direct_trait_material_r1`,
  identity `87fa70691d3b10b12fa5819199bfb657e6eca9ea23dfa78f9286b847a220fc6d`.
- Independent recount: `vm05_direct_material_recount_r1`.
- Current Mac token preflight: `vm05_direct_technical_token_preflight_r2`,
  identity `9d13e210c5927f040800c144ae2be01f59f685064ac152ceeb929d012441f175`.
- Windows staging and copied token evidence: `vm05_direct_windows_stage_r1`.
  Remote preflight identity
  `7c3df54c058f46216b3f9a1849e67f7e74c5d4851a227c23ce1e84ff8390bcfb`.
- Real qualification: `vm05_direct_real_qualification_r1`, identity
  `adb90a56e715e01c995d873e9df0ebb6e88810025f6b8ddec23393afdb752814`.
  Independent cache/arithmetic and compliance table:
  `vm05_direct_real_independent_audit_r1`.
- Archive:78,316,630 bytes, SHA256
  `5554b7a5482f9dd6d7d81c30ac9d0401b8ce1f078dde087bd4e0ebe1b27b362c`.
  It contains approved PANDORA text/token metadata and source only, no MEPS,
  questionnaires, secrets or model downloads.
- Execution source: `1873cc3` protocol/preparation; `e2095ae` collector and
  independent arithmetic; `c93d117` later source-readout UTF-8 portability.

The private result folders remain ignored. Historical Mac preflight r1 has an
older code fingerprint and is not the current execution parent.

## Next execution gate

The separate remote root is `D:/projects/GRAIN-vector-direct-vm05`. Its real
qualification ran serially after the original72-shard queue and history technical
replay. Direct weights have exited. The following is the already executed command,
retained for reproduction with a fresh output name, not an instruction to rerun:

```powershell
$py = 'C:/Users/huoke/AppData/Local/Programs/Python/Python311/python.exe'
$R = 'results/grain_vector_mechanisms'
$old = 'D:/projects/GRAIN-vector-research-vm01/results/grain_vector_mechanisms/vm05_full_vectors_r1'
& $py scripts/collect_grain_direct_traits.py --views "$R/vm05_qualification_r2/common_views.json" --material-dir "$R/vm05_direct_trait_material_r1" --tokenizer-dir "$R/local_model_metadata/qwen_989aa798" --observer-queue-dir $old --qualification --output-dir "$R/vm05_direct_real_qualification_r1"
```

Every one of234 logical readouts is repeated, with exact native replay required.
Only a real passed six-author packet can authorize source shards. Collection is
bounded by per-run time and cache limits; a failed/partial directory is preserved.
The independent verifier checks case membership, complete grids, raw-to-derived
arithmetic, view aggregation and output commitments before accepting a parent.

**Still open:** semantic anchors and answer-order sensitivity, the bounded full
direct collection queue, complete source metrics,
paired GRAIN/vector comparisons and source-frozen bridge evaluation. Do not filter
low-compliance cases or tune wording on natural outcomes. This pinned1.5B observer
cannot settle the capability of a different or much larger model family.
