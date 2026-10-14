# GRAIN R20 / F11 protocol-form conditional-loading world, protocol V1

Frozen 2026-09-26 JST before computation. It closes the recorded F11-05 deviation
("protocol-form loading world lambda_r(z) = 1 + .3 r z not run") as a new world under a
new protocol. The frozen F11 r1 outputs are untouched. No data, no neural calls.

- Cells: trait k (3) x reader r in {-1, 0, 1} x occasion t (2) x condition z in {-1, +1},
  36 cells.
- Generating model:
  x_{u,cell} = lambda_r(z) f_{u,k} + m_{u,r} + e_{u,t} + eps, with
  lambda_r(z) = 1 + .3 r z.
  - Sigma_f has unit variances and correlations .3.
  - Method variance .25, occasion variance .10, error .25.
- Authors: 2,000 source and 1,000 application, seed 20260926.
- Moment least-squares fits on the source covariance, the same estimator family as
  `f11_mtmm.moment_fit`. Components: Sigma_f (6), method, event, error.
  - M1 protocol-form loadings (correct);
  - M2 occasion-tied loadings 1 + .3 r (2t-1), the F11 r1 implemented form, misspecified;
  - M3 unconditional loadings 1.
- Application: BLUP trait predictions for new authors from all 36 cells and from the
  z = +1 cells only.
- Metrics: RMSE against the true f, correlation, and nominal 95% interval coverage from
  the BLUP posterior variance.
- Rule: a model is calibrated when coverage lies in [.93, .97] for all three traits.
