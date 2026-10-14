# F13 uncertainty core repair

Old `f13.py` clips any negative paired-change variance to zero. For example,
V1=V2=1,Cov12=2 is not a valid joint covariance, not evidence of a perfectly
measured change. It is retained as historical code; the new canonical candidate
`f13_uncertainty.py` validates finite symmetric PSD covariance before propagation.
Tiny numerical roundoff may be clipped only after that check.

New interfaces implement paired change d=(-1,1), V_delta=d'Vd and model-based
MDC=1.96 sqrt(V_delta); this is not a clinical meaningful-change threshold.
Equal-weight finite source-fit mixtures use

    total covariance = mean(within-fit covariance)
                     + covariance(between-fit means, ddof=0).

Off-diagonal terms are retained. Tests compare this with a complete finite
distribution, reject impossible covariance, and distinguish repeated identical
measurements from independent measurement errors. Five new tests;13 related
F12/F13 tests and Ruff pass.

This is an implemented/tested core, not the required nested-bootstrap experiment.
Next: source-author200 x conditional-person/event200 draws with checkpoints,
beginning with F04 Delta, then F05 state/forecast and F08 next-p; retain all missing,
correlated, duplicate/correction and shifted-prior conditions from the original
protocol. F04 source files exist under `F04/source_full_r1/L_<seed>`; do not use
only toy covariance identities as completion evidence. No new model training or
human participation is required for these tasks.
