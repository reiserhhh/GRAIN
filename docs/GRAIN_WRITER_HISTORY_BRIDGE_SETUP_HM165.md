# HM165 preparation failure, preserved

Before creation of a result directory or any model forward, prepare exited1:
the current chat-template API returned a mapping, and list(mapping) yielded
the key `input_ids` instead of integers. The subsequently attempted source
command also exited1 because no manifest existed; it executed no model calls.

Fixed before scientific sealing by an explicit mapping/list compatibility
function with a synthetic test. No old corpus, sealed experiment or model
outcome was edited/retried. Future steps check each exit code before proceeding.
