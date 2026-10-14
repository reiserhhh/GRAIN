# HM150 preparation-format recovery

Original joint_persona_style_r1 and its four sealed code/plan files remain.
The first generation invocation loaded pinned weights but failed BEFORE the
first forward: tokenizer.apply_chat_template returned a Mapping, and list()
stored its string keys as IDs. No article, sampling outcome or model parameter
was produced or selected. This is a concrete missed runtime contract, not a
scientific negative result or permission to tune the task.

New companion creates joint_persona_style_r2, accepting actual Mapping.input_ids
or flat integer lists and replaying all702 against the rendered template.
All assignments, order, prompts,54 batches, runtime caps, score units and
representations remain identical. Failed inputs/status/model manifest retained.
The original runner's generate/observe/analyze/verify consume the corrected
new manifest without editing its original sealed implementation.

Run prepare_grain_joint_persona_style_hm150_recovery.py once; every later stage
uses --output results/grain_math_response_v1/joint_persona_style_r2.
No network, API, training, private data, publication or other project changes.
