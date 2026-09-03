# GRAIN v1.1.1 Release Integrity Remediation

Release date: 2026-09-03

The `v1.1.0` naming migration completed successfully, but its tag workflow
exposed a release-only CI defect: the historical V7 closure audit compared the
`v0.2.0` manifest with the current GRAIN tree. That comparison must fail after
any legitimate later change.

Version 1.1.1 fixes the release gate without weakening it:

- the immutable `v0.2.1` lockbox is still checked in full, including annotated
  tag identity and manifest hash;
- the V7 closure audit now runs inside an isolated `v0.2.0` worktree, where its
  registered manifest and evidence belong;
- current GRAIN code, documentation, naming policy, and tests continue to run
  against the current checkout in the Python 3.12 and 3.14 CI matrix.

No scientific result, estimand, frozen tag, or historical artifact was changed.
The `v1.1.0` tag remains immutable as the record that revealed the CI defect.
