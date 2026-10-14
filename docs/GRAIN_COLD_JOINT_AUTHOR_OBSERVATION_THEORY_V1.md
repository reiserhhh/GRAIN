# Source-Frozen Joint Evidence and Author Observation

## What this adds

HM78 established a finite ambiguity of author score means. HM79--80 showed
content/actor/role associations in the fixed reader that counts alone do not
represent. HM81 connects these discoveries to an executable observation
interface. It does **not** make a new kernel theorem or a better personality
predictor. Its empirical status and results live in the parent HM81 report.

This construction uses the existing flat lexical GRAIN-compatible source
reference (PCA/varimax). It neither replaces the token-slice hierarchy nor
silently transports historical SU7 factor names into these new coordinates.
G01--G04 are anonymous source structural coordinates. Six query channels are
fixed reader responses to the named Person A, not six validated trait scales.

## Three maps, not one

Let an observation unit e retain its identity, recorded condition c_e and
evidence qualification k_e. Let A_s(e)=g_e be a source-frozen structural observer
and T_q(e)=q_e be a fixed queried reader. The author provides a distribution of
units P_u. This packet measures expectations under the empirical P_u. It does
not identify an individual biological/psychological response operator B_u.

Source-only centering and scaling define G=(g-m_s)/d_s and Q=(q-n_s)/v_s.
Application authors never refit these units, axes or reference moments.

    phi(e) = (G_e, Q_e, vec(G_e Q_e^T), {Q_ej Q_ek : j<k})
    Z_u = (E_Pu[phi] - a_s) / b_s

Here a_s and b_s are source means and scales of phi. The calibrated 49-vector
can be computed from numeric records without the application's full support,
counterfactual score table, raw text, truth or fitted inverse.

## The order of operations matters

    E[G Q^T] = E[G] E[Q]^T + Cov(G,Q)

Products formed after author means erase the last term. Making marginal mean
estimates arbitrarily precise does not recreate that missing observation.
This is a standard identity, not our invention. HM81 makes this distinction
operational on cold records, with exact event-level score contributions.

The paired query/query products similarly retain within-unit co-response.
They do not imply independent psychological axes: a single shared nonlinear
reader can generate them (the HM65 alternative remains relevant).

## Context is retained, not subtracted

Let p_uc=P_u(c), mu_uc=E[phi|u,c]. Then:

    E_Pu[phi] = sum_c p_uc mu_uc
    Cov_u(G,Q) = sum_c p_uc Cov_u(G,Q|c)
                 + Cov_{c~p_u}(E[G|u,c], E[Q|u,c])

The output retains condition prevalence and conditional profiles so these two
terms can be inspected. Decomposition is not automatic noise deletion. Under
HM81's balanced constructed histories both condition shares are 1/2; they do
not provide an empirical author-choice effect. Conditions outside source support
are flagged, while the global source coordinates are still directly computable.
No calibrated source conditional-response law is fabricated for them.

## Distance and explanation

The fixed finite map psi=(phi-a_s)/b_s gives a positive-semidefinite kernel
k(e,f)=psi(e)^T psi(f). Thus the squared distance between observed author means
has the equivalent form:

    ||E_u[psi]-E_v[psi]||^2
      = E_{u,u}[k] + E_{v,v}[k] - 2 E_{u,v}[k].

This is an empirical kernel mean discrepancy. Finite products are not a
characteristic kernel and cannot uniquely recover arbitrary text distributions.
No distributional equality conclusion or formal test is claimed from zero
distance. Per-event contributions psi(e)/n sum to the observed author score;
that explanation is exact for this calculation, not an explanation of a human's
latent personality or of every internal neural computation.

## Construction versus discovery

Fixed source coordinates, joint observations and cold author expectations now
form a usable chain. The next discovery must answer which author expectation
corresponds to a meaningful psychological/behavioral contrast under a declared
observation law. It is not enough to increase teacher fidelity, count covariance
channels or make author distances larger. In particular, this packet supplies
neither natural-author validity nor a general automatic qualification parser.

## Sources

Gretton et al. (2012), A Kernel Two-Sample Test,
https://jmlr.org/papers/v13/gretton12a.html.
Muandet et al. (2017), Kernel Mean Embedding of Distributions: A Review and Beyond,
https://arxiv.org/abs/1605.09522.
These are the mathematical precedents; the full papers are not claimed to have
been reread in this packet. Their abstract/overview was checked, and the finite
feature identities above were separately derived and tested here. OpenAI/math
was the earlier source of projection-domain research ideas, not a transferred
psychological identification theorem.
