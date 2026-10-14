# HM60 Result-After Rank-One Alignment Follow-Up

The primary result has a dominant rank-one energy mode and a named-label diagonal
advantage. These are not equivalent: ordered query/content strengths can produce
a good diagonal assignment even in a purely rank-one world.

This result-after, CPU-only follow-up retains ALL 100 primary operators and both
units. No new inputs, neural calls, fits, selected rank, or accuracy optimization.
The leading rank is fixed at one as a generic-mode diagnostic, not chosen for fit.

For D=U diag(s) V^T, define D1=s1*u1*v1^T and R=D-D1. Report named diagonal
excess and all 720 assignment ranks separately for D, D1, and R. Report complete
component entries and graph-cycle energy. These residuals are descriptive response
matrices, not noise removed from an author and not new psychological factors.

If the top singular value is numerically tied, its axis is nonunique; mark the
rank-one alignment unavailable rather than picking a convenient direction. Use
the primary fixed 1e-12 numerical floor. Verify D1 by the largest eigenvector of
D^T D, orthogonality, reconstruction, and trace/rank replay. Freeze this companion
code before computing its statistics. This is not preregistered evidence.
