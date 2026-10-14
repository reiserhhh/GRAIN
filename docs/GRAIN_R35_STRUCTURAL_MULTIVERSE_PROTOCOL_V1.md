# GRAIN R35/R36 structural reliability multiverse on real text, protocol V1

Frozen 2026-09-26 JST before computation. Local authorized PANDORA (36-comment Big5-only
material) and Essays units from local_bilinear_r2. Text only: no labels, no MEPS, no
neural calls, and no raw text in the outputs. CPU TF-IDF and truncated SVD only.

- Universes, 12 per corpus:
  - k in {12, 24, 48};
  - n-grams (1,1) or (1,2);
  - weighting: sublinear TF-IDF (the frozen runtime default) or binary l2-normalized
    term presence;
  - max_features 5,000; strip_accents unicode.
  - The vocabulary and idf are fitted once per universe on the whole corpus
    (unsupervised); the SVD is fitted per sample.
- Split-half: 5 author-disjoint random halves (seed 20260926). Fit an SVD on each half.
  - Subspace similarity S = ||V_A^T V_B||_F^2 / k, in [0,1]. The random-subspace
    expectation is k/5000.
  - Matched loading congruence: mean |cos| of Hungarian-matched components.
- Bootstrap: 10 author-bootstrap resamples, each compared with the full-corpus SVD.
- Frozen-structure transport: the frozen GRAIN runtime vocabulary is applied to each
  corpus, and an SVD with 24 components is fitted on each of 5 author halves. S is taken
  against the frozen runtime components, which were fitted on a different Reddit
  discovery corpus.
- Rule: structural reliability is admitted for a universe when the mean split-half
  S >= .80. The multiverse is reported whole; no universe is selected by result.
