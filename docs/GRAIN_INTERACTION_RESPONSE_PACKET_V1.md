# M09: conditional author response and common conversation inputs

2026-09-13. Freeze before results; no clinical interpretation or human-label fit.

## Numerical mechanism

Single-session model y=a+b*z+(c+d*z)*u+s+e. Partner input u can share state s or
depend on prior y. A separately randomized assignment r can vary u. Only when r is
relevant and has no other route to y can it distinguish response c/d from shared
causes. Compare OLS, IV, and residual/control-function accounting; keep weak or
invalid assignment and constant-input counterexamples. Existing OU/state work is
inherited; this packet targets joint generation rather than another rate fit.

Fit each simulated person's first64opportunities, apply to subsequent64; no future
outcomes choose parameters. Three seeds,128persons/world. Actual state and response
operator are evaluation-only. HAC4 working covariance handles short dependence;
no guaranteed small-sample coverage or causal conclusion in natural data.

## Text mechanism

Six topics, two wording variants, six acts: accepting, rejecting, current help
request, correcting misunderstanding, reporting past help request, future intent.
Use pinned existing NLI on reply-only and role-labelled pair; swap only preceding
partner turn for accept/reject. A habitual-help query is a separate negative scope
control, not another name for asking once. Preserve every raw probability/address.
No tuning on outcomes. This is an observer experiment, not a new act-classifier
training task. Do not equate similarity, act evidence and personality tendency.

## Natural use

Existing English NLI is not silently declared validated for Japanese MEPS. Use the
already verified local Qwen2.5-1.5B only as a fixed dialogue-act observer: paired
versus same reply alone, output seven letter logits, no generated explanations.
First check twelve authored English/Japanese examples. Then first24eligible
participants by existing pseudonym, last complete adjacent AI->user pair fitting
the declared budget; no truncation. Different scopes on identical replies. Only
local pseudonyms, hashes, lengths and scalar probabilities saved, no raw text.
No questionnaires, no fitted personal parameters, no inferred instrument r.
Natural label frequencies are observer descriptions, not accuracy or validity.

## Prior methods

- [Stolcke et al. 2000](https://aclanthology.org/J00-3003/): dialogue acts use lexical
  and sequential context; an act is not a personality trait.
- [Bothe et al. 2018](https://aclanthology.org/L18-1307/): context-based act recognition.
- [Wooldridge 2015](https://doi.org/10.3368/jhr.50.2.420): control-function methods;
  use the explicit linear IV/residual-inclusion special case, not all nonlinear claims.
  Publisher full text was unavailable here; the author's
  [NBER lecture 6](https://www.nber.org/sites/default/files/2022-09/slides_6_controlfuncs.pdf)
  is the accessible primary mathematical reference.

## Acceptance

Response-function coefficients/held outcomes, common-input decomposition,
identifiability witness, typed text evidence and role/address-preserving natural
application. Archive failures and bounds; complete a finite case or counterexample,
not an all-language clinical effect claim. No strong-first-stage-only filtering of
results. No new model downloads, no APIs, no old long-run queues, no publishing.
