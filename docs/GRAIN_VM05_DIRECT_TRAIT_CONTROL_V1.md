# VM05 direct LLM trait readout control

2026-09-06. Freeze the following before any new trait-prompt forward or outcome
comparison. This is a research control, not a replacement for the released
deterministic GRAIN scorer or a clinical assessment.

## Question and input

Can the same frozen Qwen observer itself make label-related trait distinctions,
as opposed to a supervised Ridge/Logistic head extracting them from its vectors?
Use the original complete common-material authors, three12-comment views and
raw/cue/placebo arms. Quote the same full comment bodies and excerpt numbering;
do not add other histories, retrieval, questionnaires, demonstrations or identities.
Perform a label-blind full token-budget/support preflight. If a longer trait
prompt does not fit, exclude the whole author from the paired comparison and
report support, rather than truncate or choose a favorable view/task.

Big5-only sources receive all five Big5 prompts; MBTI-only sources all four axes.
Bridge material may receive all nine without opening outcome values. Training,
layer selection, model/prompt search or outcome-based support selection is absent.
This control uses Qwen2.5-1.5B-Instruct at the existing pinned revision, CUDA BF16,
eager attention,3072 fixed execution positions. It cannot establish the capability
or mechanism of GPT-class models generally.

## Output definition

Use one task-specific prompt and a closed set of single-token answer candidates.
Big5: digits0-9 from very low to very high. MBTI: I/E, S/N, T/F, P/J, with positive
E/N/F/J matching source encoding. Candidate IDs must be distinct single tokens in
the native tokenizer. The native chat prefix is not retokenized with answers.

For native logits z_j and the task's candidate set A, record

```
q_j = exp(z_j) / sum_{k in A} exp(z_k)
Big5_readout = sum_{j=0..9} (j/9) * q_j
MBTI_readout = q_positive
candidate_mass = sum_{j in A} softmax(z over full vocabulary)_j
```

These are **candidate-conditioned model readouts**, not calibrated human trait
probabilities or Big5 percentiles. Averaging three view readouts is a fixed
aggregation rule, not a derived Bayesian posterior. Report individual views and
their dispersion. Big5 MAE/RMSE against percentile labels are scale-dependent
descriptors; Pearson/Spearman are primary for this uncalibrated readout.

Also record whether the unrestricted greedy first token is a legal candidate,
the legal-candidate argmax, entropy and candidate mass. Do not call first-token
compliance a complete-response parse rate. Low candidate mass must stay visible;
do not drop an author because the model gives an inconvenient answer. No JSON
repair, answer extraction retries, self-reported confidence or free chain-of-thought.

## Evidence stages

1. Complete token/support preflight without weights or label values. Every arm
   and applicable task must fit all three original views; no silent sampling.
2. Synthetic output arithmetic and prompt/candidate controls, plus real frozen
   source technical cases. Repeated native states/logits must agree before full
   collection. Semantic anchor examples diagnose prompt/model capability, not
   human personality validity, and cannot tune the prompt on natural outcomes.
3. Bounded, cached full-supported source collection after existing GPU jobs. Save
   every native candidate logit, mass and first-token compliance, not only the
   conditional probability. No competing weights or new model download.
4. Same-author/arm paired source comparison against supervised vectors and GRAIN,
   official fold summaries and author bootstrap. No score-based exclusions or
   post-hoc polarity inversion. Any future source-fitted calibration is a separate
   statistical-head control, not silently included in the direct LLM result.
5. Freeze source choices before bridge evaluation; keep MEPS questionnaire and
   mood outcomes closed. Weak or null direct readout does not falsify the entire
   model-family reasoning hypothesis.

The earlier two social-initiation questions remain separate. They do not count
as these four MBTI and five Big5 tasks. Conversely, these conditional output
scores alone do not locate a causal personality mechanism inside the model;
the history/vector and functional intervention paths are still required.
