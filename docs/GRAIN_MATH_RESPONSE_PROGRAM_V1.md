# Paired-response construction and cold scoring

Executed 2026-10-07. This is a bounded construction/mechanism experiment,
not the completion of the original 100-card program or a human personality scale.

## Build from zero

1. Freeze the target, evidence qualification, recorded conditions, operation library,
   and semantic material families. Do not put psychological scores in the text.
2. Keep every factorial variant of a family in the same role. The primary bank has
   16 source, 8 validation, 8 test families, each with 16 texts.
3. Encode using frozen local model snapshots. E5 uses `passage: ` and mean pooling;
   BGE uses no prefix and CLS pooling. Both normalize vectors and have dimension 1024.
   Revision, weight hash, pooling configuration, software versions and input hashes
   are saved under `00/`.
4. Estimate source forward operation columns in a low-dimensional action space.
   Compare the standard GLS/constrained dual solution with ridge, an additive
   ablation, an uncalibrated prototype and negative controls.
5. Freeze models and source leave-family-out unit resolution before testing.
   Evaluate paired operations and absolute units separately. Family bootstrap
   treats a material family, not its 16 rows, as the independent cluster.
6. In the result-guided follow-up, learn strong polarity/ownership/condition
   coordinates from source data and construct multilinear features. Calibrate
   these features using source out-of-family predictions, not source in-sample
   predictions. Test once on 16 new content families.
7. Cold scoring accepts only embeddings and a frozen NPZ bundle. It does not
   accept target labels, material identifiers, author identifiers or conditions.

## Program and units

The follow-up model is:

\[
z(h)=(h-\mu_h)W_z+b_z,
\quad\psi(z)=(p,o,c,po,pc,oc,poc),
\quad T(h)=(\psi(z)-\mu_\psi)W_T+b_T.
\]

These are signed continuous action coordinates, not independent probabilities.
The declared ideal material units are `T1=(p+po)/2` and `T2=(pc+poc)/2`.
`T2` is a condition-tagged coordinate; a single score is not an author response
slope or an emotional state. An author-level conditional contrast requires
qualified independent events and a declared reference distribution.
An ineligible text has no own-person contribution; an author with no eligible
events is unobserved, not neutral. Duplicate renderings are not new events.

## Reproduce

Use the existing environment (NumPy, SciPy, sentence-transformers, PyTorch,
pytest, Ruff). No installation or downloading is performed by the runner.
Both pinned model snapshots must already exist in the local Hugging Face cache.

```sh
python scripts/run_grain_math_response.py all --out results/grain_math_response_v1/reproduction_r1 --device mps
python -m pytest -q tests/test_grain_math_response.py
python scripts/verify_grain_math_response.py --out results/grain_math_response_v1/reproduction_r1
```

Use `--device cpu` on hosts without MPS. Windows training host `.20` and Linux
Intel evaluation host `.30` are not automatically accessed by this local package.
No CUDA availability is assumed.

Cold scoring, with a canonical follow-up bundle and its new-family cache:

```sh
python scripts/score_grain_math_response.py \
  --bundle results/grain_math_response_v1/r1/06/reader_bundles/e5_joint_binding.npz \
  --vectors results/grain_math_response_v1/r1/06/01B/embeddings/e5.npy \
  --output results/grain_math_response_v1/r1/cold_confirmation.npy
```

## Canonical artifacts and decisions

- `01A/`: 24 finite configurations, estimated columns, exact collision witnesses.
- `02/`: old frozen reader boundary and observation-channel diagnostics.
- `01B/`: 512-text bank embeddings, source fit/freeze/tolerances, held scores.
- `03/`: four-corner witnesses and score/response transport, including failures.
- `04/`: matched-spectrum coupled dynamics and a normal rotation counterexample.
- `06/`: 256 new texts, two-stage bundles, cold confirmation and error decomposition.
- `05/`: model/input checks, cold CLI replay, arithmetic verification and code snapshots.

The new-family follow-up T1 RMSE is .285 (E5) and .317 (BGE), compared with
.441 and .488 for the original ridge readout. T2 RMSE is .315/.387.
Both improvements have positive family-paired MSE improvement intervals.
Polarity/ownership/condition supervision is explicit construction information;
these numbers must not be described as Big5 accuracy or human personality validity.
The additive-vs-joint follow-up uses the same coordinate supervision and supports
the value of joint qualification, not the novelty of multiplication or ridge.

The exact dual is retained as an identifiability reference, not promoted as a
universal production scorer. It has positive held target actions but unstable
absolute scores on new family cores. Reader transport still has unit differences.
No new natural-text psychometric validation was performed here.

The full Chinese report and revised task ledger are in the parent Project Persona
repository under `reports/openai_math_review_20261007/execution/RESEARCH_RESULTS_ZH.md`
and `docs/GRAIN_MATH_RESPONSE_TODO_20261007.md`.
