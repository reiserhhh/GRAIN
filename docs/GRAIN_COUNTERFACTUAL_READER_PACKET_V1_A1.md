# Counterfactual reader packet V1, amendment A1 (input checkpoint, before any model forward)

Date 2026-09-26 JST. V1 (sha 616fb2d2…) is unchanged. No model forward had run: prepare is tokenizer-only, and
observer.calls = 0.

- **What triggered this.** The runner's own sanity check expected the question rows to begin at the token for
  "Estimate". V2's `content_indices(question)` returns only complete tokens inside the question text. The boundary
  token " Estimate" carries the preceding space, so it is excluded, and V2's rows begin at " the writer's
  extraversion…" and run to the end.
- **Change.** The question rows are defined exactly as V2's `full_context` defines them: the first complete token
  inside the question text, through the end of the prompt. This keeps the rows homologous with the reader site
  nominated in V2 (L12.H0).
- **Effect.** One token fewer than a literal "first token" reading. The rows remain identical across the four insert
  arms.
- **Unchanged.** Everything else in V1.
