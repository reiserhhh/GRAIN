# S24: the frozen functional reader reaches the useful configuration pathway

2026-09-08. COMPLETE:24 source N authors at indices24:48, view0;
96forwards,0backwards,139.51seconds after model loading. No source Y joined.
Code/protocol d51628c. First8resource pilot32forwards/50.78seconds, then complete24.

## Finding

S20's frozen24-author gradient transports from the small pairing intervention
to the actual head displacement caused by S9's larger body-configuration change.
Its prediction of the symmetric head allocation has r=.95433,
bootstrap95%CI[.88975,.97925], MSE/head-effect-energy=.022989.
No new gradient estimation, score regression, calibration or sign choice occurs.

The head allocation itself correlates with the full configuration contrast
at r=.70647[.17751,.89350]. This directly joins the earlier operative gradient
to part of the useful configuration path, rather than merely correlating two
unrelated statistical decoders. The remaining paths carry substantial variation.

## Exact four-cell intervention

At block16, B=0 applies S9's mean projection to each comment-body token group;
B=1 keeps native text states. At layer17, H=0/1 donates the corresponding
projected/native final-query concatenated head, just before W_O. All other
positions continue under their assigned B condition. Four scores F11,F00,F10,F01
are measured per author. Native/projected endpoint replay versus S9 is exactly0.

C=F11-F00;
head_native=F11-F10;
head_projected=F01-F00;
phi_head=(head_native+head_projected)/2;
phi_other=((F11-F01)+(F10-F00))/2;
I=F11-F10-F01+F00.

phi_head+phi_other=C exactly for all24 authors. I is reported, not discarded.
The two-player average follows the marginal-contribution formula in
[Lundberg & Lee2017, Eq4](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf).
The chosen mixed-state computational game defines the allocation; these are
not natural human causal effects or unique allocations across all path definitions.

| Component | Signed mean | RMS | r with C |
|---|---:|---:|---:|
| Full configuration C | .00264681 | .01350153 | 1 |
| Head effect in native body | .00568268 | .00633222 | .63559 |
| Head effect in projected body | .00587284 | .00688824 | .75237 |
| Symmetric head allocation | .00577776 | .00659190 | .70647 |
| Other-path allocation | -.00313095 | .01165312 | .97978 |
| Interaction | -.00019016 | .00113006 | -.83728 |
| Frozen-gradient prediction | .00577033 | .00637592 | .66836 |

Positive head and negative other-path means partly cancel. RMS includes these
offsets, so the head/full RMS ratio is not a percentage of explained variance.
Other-path/C r=.9798 is partly an algebraic part-whole relation, not independent
predictive evidence. The interaction r interval[-.9560,.1659] is wide and crosses
zero; its small absolute amplitude is clearer than a directional generalization.
The full signed24-author table is preserved privately.

## Frozen functional approximation

Let delta_h=h_native-h_projected and z=g_N,S20^T delta_h. This is a different,
larger displacement from the pairing d used to estimate the shared gradient.

| Actual target | r(z,target) | Bootstrap95%CI | MSE/target energy |
|---|---:|---|---:|
| Head effect, native body | .95180 | [.87872,.97854] | .018573 |
| Head effect, projected body | .94400 | [.89405,.97367] | .039869 |
| Symmetric head allocation | .95433 | [.88975,.97925] | .022989 |

Sign agreement24/24 in all three rows. This supports transporting the functional
direction to the source configuration displacement. It is not a new human
personality r. The full C is not recovered by this one head alone; its frozen
linear reconstruction r=.66836 and error/C-energy=.79214 are both reported.

## Decision and artifacts

Do not spend another gradient grid refining the already good head approximation.
Next extract this actual configuration displacement using the frozen prefix
and local projected KV computation, then connect its three-view score to source N.
First replay stored S24 heads to verify that compute shortcut. Compare
with S20 pairing scores; do not add them blindly because the paths overlap.
Retain other-path allocation as a separate mechanism question rather than
calling it noise. The target is useful source inference, not perfect accounting
of every decoder branch before proceeding.

Private output: results/grain_vector_mechanisms/configuration_mediation_s24_r1/.
Contains manifest,24paired head NPZ/JSON packets, private_effects.csv,
checkpoint_008.json, metrics.csv, summary.json, run_status.json.
Reproduce with parent venv Python and a new output directory supplied to every
invocation of scripts/run_grain_configuration_mediation.py:
prepare; run (defaults8); inspect; run --max-authors 24; evaluate.

85focused tests pass; Ruff and whitespace checks pass. Initial unit-test fixture
used NumPy index groups while the frozen S9 helper expected lists; the new wrapper
normalizes group containers before calling it, and both tensor/tuple output plus
exception cleanup are tested. No failed neural collection, no protected data,
no training, no T/F/release changes, no private upload or remote push.
