# HM255 MPS CPU64 technical companion

The primary sealed collection failed with0 completed batches/0 saved item
observations after1.6077266249340028 model seconds: MPS does not support
`full.double()`. Its manifest, scientific code, head, prepared inputs and
FAILED runtime remain unchanged. No outcome was available to choose a new
method. Before freezing, an eight-versus-twelve parser mismatch and tokenizer
BatchEncoding return were corrected; one premature collect invocation failed
before manifest/model loading. These are retained development events.

This separate companion changes ONLY the full-vocabulary log-normalizer:
copy the same float32 logits to CPU, THEN calculate logsumexp in float64.
Source/key/scorer, all tokens,20 complete items,12x16 inputs, model/dtype,
cache and batch size, hypotheses, signs, controls and inference head remain.
The scientific primary is not overwritten or restarted. The parent failed
receipt SHA and zero-batch boundary are checked before every companion stage.
The remaining cap is2200-1.6077266249340028 seconds, not a renewed budget.

Use `scripts/run_grain_item_bridge_hm255_mps.py` and its independently sealed
`item_observation_bridge_r1/mps_cpu64_recovery_r1` output. CPU tiny-model tests
must confirm the old and companion functions have identical states, selected
logits, normalizers, greedy tokens and prefix length. No precision, prompt or
scientific-result rescue is authorized; any new failure remains terminal.
