# Second-observer capability and readout audit

Date: 2026-09-06. Status: bounded battery complete; neither observer qualifies
for answer-code-invariant semantic intervention claims. GRAIN scoring is unchanged.

## Design and results

Protocol: [GRAIN_SECOND_OBSERVER_PROTOCOL](../../docs/GRAIN_SECOND_OBSERVER_PROTOCOL.md),
committed cd99ee7 before execution; implementation d32984b. Each observer received
the same 208 unique synthetic cases (192 explicit facts and 16 answer-code controls)
plus eight fixed repeats. No natural participant text or psychological label was used.

| Readout | Fact accuracy, 192 cases | Normal code, /8 | Reversed code, /8 | Opposite output pairs, /8 | Main gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen BF16, full-vocabulary greedy generation | .8958 | 5 | 4 | 1 | Not qualified |
| Qwen BF16, restricted first-position Yes/No margin | .8802 | 5 | 4 | 1 | Not qualified |
| Llama Q4, Ollama generation | .8750 | 7 | 4 | 5 | Not qualified |

Qwen passes the explicit-fact gate, not the code gate. Llama fails the frozen
composition-family gate (balanced accuracy .6667), although a diagnostic below
shows this is a format-compliance failure rather than wrong leading answers.
Both observers reproduce all eight fixed repeated replies exactly. This is limited
repeatability, not robustness across prompts, model versions, or populations.

### Format and semantic errors are not interchangeable

Llama's 24 strict fact failures all concern partially satisfied conjunctions.
Every reply starts with the correct standalone `No.` sentence, then continues
with an explanation, often beyond the 16-token cap. The frozen parser correctly
rejects these as noncompliant. A separately labeled, post-hoc leading-sentence
diagnostic is 192/192; it does not assess explanation truth, change the primary
gate, or prove general semantic competence. The strict overall parse rate is
184/208 (.8846), not a personality-prediction parse rate.

Qwen's three generation-versus-margin differences are exactly the three BF16
Yes/No margin ties. Generation selects `No` and is correct on these cases;
restricted ties were explicitly unclassified and counted incorrect. There are
no non-tied disagreements. The .0156 accuracy increase therefore is not evidence
that generating an explanation added reasoning. All 16 code-control margins
exactly reproduce VM04 independent access values (maximum absolute error zero).

Both models still genuinely fail answer recoding: Llama outputs `Yes` on all
eight reversed-code cases; Qwen outputs `No` on all eight. Parsing changes cannot
repair that failure. The four code scenarios also use disposition wording, so
failure may involve evidence-to-disposition criteria as well as instruction/code
use. It does not uniquely identify a model-internal bias mechanism.

### Uncertainty

Qwen fact accuracy has scenario-cluster bootstrap CI [.8750,.9219]. Llama's
strict scenario-cluster interval is [.875,.875] because each of the twelve
scenarios has exactly the same error pattern. This degenerate resampling interval
does not mean population certainty; these are twelve designed template clusters,
not a random sample of all possible language tasks. Generalization needs new
templates/evidence families, not more copies of the same wording.

Paired strict fact accuracy difference Llama minus Qwen is -.0208
[-.0469,0]. That compares the whole size/architecture/quantization/template/backend
package, and it includes the format penalty. It is not a causal model-size test.

## Mechanism implication and next step

Separate the experimental objects: contextual representation `h`, evidence/task
readout `R_q(h)`, requested answer code `D_c`, and output serialization/parser.
The proposed decomposition `answer = D_c(R_q(h))` is an investigative hypothesis,
not an identified neural circuit. A perturbation of Yes/No logits can act on
several of these objects. VM04 supports internal access across current texts;
this follow-up does not yet make that access personality-specific or code-invariant.

The next semantic experiment should orthogonally vary factual versus disposition
questions, ordinary versus reversed answer codes, and candidate readout versus
generation on a bounded synthetic panel, with new templates reserved before
running. Do not tune natural-author prompts until code/task effects are separated.
Meanwhile VM05 can test information in pre-answer vectors with independently
fitted statistical heads; those heads are information detectors, not proof of
the LLM's own use of that information. No LoRA or larger-model training is justified.

## Artifacts and runtime

Under ignored `results/grain_vector_mechanisms/`:

- `vm03_second_observer_llama_r1`: complete 216 API calls, 101.142 s, 938 generated
  tokens, 25,418 prompt tokens. Existing Llama 3.1 8B Q4_K_M, Ollama 0.9.3 on
  evaluation host .30:11434; digest
  `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`.
- `vm03_second_observer_qwen_r1`: complete 216 prompts, 553 prefill forwards,
  82.485 s, 337 generated tokens, 24,986 prompt tokens. Qwen2.5-1.5B-Instruct
  revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, BF16 eager, Windows .20.
- `vm03_second_observer_analysis_r2`: canonical aggregate, per-family metrics,
  paired contrasts, strict-score recomputation and explicitly post-hoc format
  diagnostic. `r1` preserved; model outputs were not rerun or overwritten.

Pinned model checks, input material hashes, output rows and code snapshots are
retained. API behavior is not a whitebox replication. The older Ollama instance
was used through ordinary chat calls, not assumed to expose hidden states or
modern logprobs fields. No downloads, training, paid APIs, other-service eviction
or push were performed. Generated synthetic text is not real-person evidence.
