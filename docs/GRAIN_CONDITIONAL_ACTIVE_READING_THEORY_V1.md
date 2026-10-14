# Conditional Active Reading: Shared Carrier and Instance Pairing

HM37 is a verified finite source-only computation on the fixed HM35/HM36 bank.
This document updates interpretation without changing the frozen plan or code.
All original measurement requirements remain open where not actually tested.

## Compiled Arithmetic

For each target q, retain its separate physical token space and source-active
orthonormal basis U_q. Let v=d_P<d_HF,deltaH>, theta=U_q^T v/||deltaH||,
and eta=||deltaH|| U_q^T DeltaP. The supported scalar action is theta^T eta.
The normalization sets a unit; it does not correct differing H directions.

The supplied condition c=(H,B,Q) lies in the fixed eight-cell design. Source
wordings0/2 are equally averaged before the complete eight-character transform.
Every character is retained. Three fixed nodes define quadratic interpolation;
only those same nodes are evaluated here, not an arbitrary continuous theorem.
Global/node/fiber/full/mismatched laws all remain reported. No application
covector is an argument to the prediction function.

## Observations

Application Simpson sign agreement: global96/96, full92/96. Global relative
RMSE spans22.87%-68.90%; full28.73%-70.37%. Full coefficient reconstruction
errors32.11%-59.12% are not scalar prediction errors. Actual supported action
has substantially smaller missing error; carrier transfer and coefficient
transport therefore differ. Full conditioning is not promoted by default.

Source constant character carries97.64%-98.65% of coefficient squared norm.
This is not the fraction of human personality explained. Large common
covector energy can coexist with a material scalar error on a particular
input. Source full-law errors are not zero because source wordings were
averaged, not replayed as their own fitted observations.

## Pairing Versus Marginal Structure

For any fixed finite observation family with declared weights:

    E[theta^T eta] = E[theta]^T E[eta]
                    + E[(theta-E[theta])^T(eta-E[eta])].

This is the standard covariance identity, not a new theorem. Its relevance
here is operational: the source-averaged coefficient can retain almost all
energy while omitting the coefficient/input pairing that carries its action.
For two equal-weight wordings, the second term is

    ((theta0-theta2)^T(eta0-eta2))/4.

At the application conditional mean, a source full law yields the exact split

    actual_full_mean - source_prediction_mean
      = carrier_missing_mean
        + (theta_bar_app-theta_bar_source)^T eta_bar_app
        + pairing_covariance_app.

The last two terms separate shared coefficient transport from within-family
pairing. Signed covariance is retained; squared component errors need their
cross terms. With only two wordings the covariance rank is necessarily at
most one, so this rank cannot establish a new low-dimensional construct.

HM38 will measure these terms on the saved actual states/covectors with no
new fitting or neural calls. A large pairing term calls for a joint reading
law; a dominant mean-transport term calls for a state-to-coefficient response
mechanism. Neither automatically warrants a larger prediction grid.

## Scope and Literature

Finite characters are standard orthogonal coordinates on Z2^3, as in
[O'Donnell, Analysis of Boolean Functions, sections1.3-1.4](https://www.cs.cmu.edu/~odonnell/papers/Analysis-of-Boolean-Functions-by-Ryan-ODonnell.pdf).
The group is an experiment index, not a discovered physical/personality
symmetry. OpenAI/math's fixed generator observation-family/domain reasoning
motivates retaining all declared cells; its rational-formula guarantees do
not apply to these floating neural functions.
[Fixed generator manuscript](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Polynomial-Hitting-Lists-for-Noncommutative-Rational-Formulas-September-24-2026/build/sections/generator.tex).

The bank was previously evaluated by earlier packets. Prediction-before-
target access here makes computation auditable but does not convert the
reused materials into a new independent blind replication. Upstream neural
states and supplied conditions remain inputs. GRAIN author eligibility,
event identity, source reference and aggregation are not replaced by this
reader-action law; derivative sensitivity is not a person's trait score.

Verification:398output hashes,2,327,007numbers, maximum difference3.331e-16;
combined36.40/300seconds including10seconds reserved for pre-freeze failure.
10new tests/569math tests pass. No API/private data/new neural training/release.
