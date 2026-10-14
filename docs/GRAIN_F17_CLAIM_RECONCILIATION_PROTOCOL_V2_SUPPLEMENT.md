# F17 claim-level reconciliation: supplement protocol V2 (packet 30 A)

Frozen 2026-09-27 08:05 JST. SUPPLEMENTS `F17/claim_level_reconciliation_r1` (immutable, kept) with a new root
`F17/claim_level_reconciliation_r2`. Same status vocabulary and pointer rules as V1. Metadata only.

- **Union.** Every r1 row is kept with its r1 status. Every one of the 50 ledger rows ALSO gets its current gap as clause
  row(s) (verbatim). An r1 current-gap row that is split in r2 is marked REPLACED_BY_SPLIT and stays in the file.
- **Overrides.** An r1 audit row may receive an r2 override (effective status + reason); the r1 values stay in their
  columns.
- **Mandatory fixes** (supervisor 0754):
  - R12: informative-context / condition-diversity, real-token and H01 clauses kept separately; the r1 "human-only
    remainder" inference is withdrawn; no R12 promotion.
  - R26: direct ALBA / scalar-IRT comparison kept separate from the accepted Gaussian comparison; no ALBA run.
  - R13, R17, R20, H05, H06: doc / executable work split from model and human portions.
  - R21 / R23: existing shuffles delivered; a new cross-construct null needs distinct matched construct inputs.
  - R36: planted choice-composition control (F14/known_structure_r1) delivered; it is not an established-construct
    control and is not repeated.
  - R03: unread open papers are retrieval work (browsing permitted by the source handoff), not an access blocker.
- **Invariance.** Ledger science / protocol / acceptance fields unchanged (21/44; 3/6). No R12 or R26 promotion.
- **Inputs.** r1 `clause_reconciliation.csv` sha e90c33b1ce87cf227bd9592c3698badf6e7b46356216ba9f52f14c971f55d83b; r1 summary sha 61f345af8e8c8197bb8ac6bbc036bdbaeb2ee2f0e2b331f523487a785e29dc11.
