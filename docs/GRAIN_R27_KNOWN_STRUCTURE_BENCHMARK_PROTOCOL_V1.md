# GRAIN R27/R36 known-structure benchmark on real text, protocol V1

Frozen 2026-09-26 JST before computation. Real PANDORA comments (22,464, local
authorized), their cached E5 vectors and GRAIN coordinates, and subreddit metadata. No
labels, no neural calls, no raw text in the outputs.

- Planted structure: pseudo-author u has z_u ~ N(0,1). Each of 36 comments is drawn by
  first sampling a subreddit among the 50 most frequent, with
  logit = log frequency_s + beta z_u w_s, where w_s ~ N(0,1) is declared and
  standardized. A comment is then sampled uniformly within that subreddit, without
  replacement within the author.
- 400 pseudo-authors per replicate; beta in {0, .5, 1.0}; 5 replicates each (seed
  20260926).
- Measures per pseudo-author:
  - M1 choice composition (50 subreddit shares);
  - M2 mean GRAIN coordinates (24);
  - M3 mean E5 vector (1024);
  - M4 subreddit-de-centred GRAIN: each comment's GRAIN minus its subreddit's mean over
    the pool, then averaged.
- Recovery of z:
  - unsupervised: |r| of PC1 with z;
  - supervised: 5-fold cross-validated ridge (alpha 1.0, fixed) prediction r with z.
- Rule:
  - a measure recovers the planted structure when its supervised r at beta 1.0 exceeds
    the beta = 0 null's 95th percentile across replicates by at least .10;
  - de-centring is predicted to erase choice structure (C2 rival: an operation removes
    present information).
