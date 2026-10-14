# HM198: joint target/person binding, not score calibration

Frozen before any new model observation, 2026-10-12 JST.

## Mechanism question and inventory

HM46 queried six targets separately. HM110 varied two events and read separate
targets. HM145/146 tested joint evidence diagnostics/person qualification;
HM154 compared target wordings. HM185 built a SOURCE Big5-profile mean atlas;
its cross-author result did not promote the atlas. HM197 closed top4
recalibration. None supplies the actual joint-query family tested here.

Competing constructions: (1) read each behavioral tendency for each person,
then multiply the two soft scores; (2) read a person-qualified conjunction or
disjunction as its own observation. This is not a claim that model soft scores
are calibrated event probabilities or that the product is statistical truth.
An additive person/target reader cannot recover the designed coupling contrast.
An independent product reader can; the question is whether it matches the
native compound-query action without fitting it.

If joint support follows assignment with AND/OR opposition in both wording
banks and both orders, include a joint target/person observation in GRAIN's
measurement interface. If the fixed product reproduces it, use that economical
composition on this support. If not, retain the joint reader rather than tune
product weights. Failure of a bank or order stays a failure, not a prompt search.

## New finite material and frozen native observations

Sixteen NEW fictional records: two wording/scenario banks x two opposite
presentation orders x four assignments of social initiation and worry to A/B.
Each bank has an identical entire whitespace word bag, identical behavior
counts and identical name counts. Keep all four records, not only matched
positive combinations. Their owner bindings differ. Neither bank is natural
human text; bank and scenario also differ together, so they are not separately
identified. No questionnaire or application labels.

Each body receives fourteen predeclared queries: four single person/tendency,
four AND and four OR over AA/AB/BA/BB, two unchanged blue-badge controls.
Queries use behavioral descriptions, not diagnostic personality labels.
The compiler records real chat-template token IDs and the common pre-question
body boundary, and validates that each question shares exactly that causal
prefix. Fixed cached Qwen float32/eager MPS; only Y/N next-token logits, full
vocabulary normalizer, block16 body endpoint and final normalized query state.
No sampling, generation, training, layer/head/dose/rank/grid or answer fit.
16 batches/224 new enquiries; cumulative model-time cap240 seconds including
load. No automatic retries or cap renewal. A stopped/failed run is preserved.

## Measured quantities and consequence

Let p_q(x) be Y probability conditional on the Y/N answer alphabet; also keep
log odds, answer-alphabet mass and entropy. These are model readings, not human
trait probabilities. Define for AND/OR respectively

    C_op = (F_AA - F_AB - F_BA + F_BB)/4
    J = (C_AND-C_OR)/2; U = (C_AND+C_OR)/2
    J_product = (p_EA-p_EB)*(p_NA-p_NB)/4
    remainder = J-J_product.

Keep signed four-corner coefficients 1,e,n,en in every bank/order, every query
and every field. Do NOT multiply log odds and call that a probability model.
Product comparison only uses conditional Y scores.

Two declared strictly positive evidence laws P+/-=(1+/-0.4*en)/4 have identical
individual tendency marginals. Their reading difference is0.8 times the en
coefficient. They are designed mixtures, NOT new authors or personality types.
All joint and single readings remain, including common/lexical effects.

At the final normalized query state, Y/N log odds equals the frozen output-head
covector times h (including bias when present). Audit in CPU float64; this
linear terminal identity does NOT make the upstream reasoning linear or localize
a unique causal circuit. A nonzero vector norm alone does not establish meaning.
Body states must agree across suffixes within tolerance3e-4; prefix identity and
causal independence are compatibility checks, not the discovery.

Success means a measured design distinction that changes the observation
interface. No materiality/significance threshold is optimized, no human
prediction accuracy claim, no original18/100/44 closure. Acknowledged historical
research-informed design is not an untouched lockbox/preregistration.

## Prior mathematics and actual reading

Feng/Steinhardt2024, https://arxiv.org/html/2310.17191v2 , Sections2 and4.1:
entity-attribute binding and additive binding-ID interventions already exist;
no novelty claim for binding, Fourier characters or activation capture.
The planned experiment adds this system's joint target/person query comparison,
not rediscovery of a binding ID or proof of a new psychological dimension.

OpenAI/math pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a:
Universal-strong-Kadison-Kastler-stability-September-23-2026/build/sections/
01-introduction.tex, reading the introduction's distinction between proximity
and multiplicative structure. Inspiration: terminal marginal agreement does
not substitute for a composition test. No C*-algebra hypotheses or theorem
are transferred to a neural decoder. Initial attempt at a guessed manuscript
path failed; correct path was found from the existing source manifest. No
scientific data were collected in that failed source lookup.

No private/API/new training/Git/Claude/automation/lockbox/old queue. Numerical
audit is same-agent alternate arithmetic, not independent blind replication.

Pre-seal checks: five synthetic tests passed on first execution. First lint
reported two missing shebangs, one import separator and three synchronous-hook
closure warnings. Fixed before prepare/sealing; no scientific model call or
result existed during these repairs.

First prepare exited before output-directory creation or any neural call:
the installed template API defaults to BatchEncoding rather than a list.
The compiler now explicitly requests return_dict=False, with a cached-tokenizer
integration test. A separate diagnostic shell quoting error also occurred
before tokenization. Neither changed materials, query roles or result criteria;
both terminal failures are retained in the parent execution record.
