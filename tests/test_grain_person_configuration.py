#!/usr/bin/env python3
"""Selected-law and person-sector tests without models, private data or labels."""

import itertools
from collections import Counter

import numpy as np
import pytest

from grain_research.construction import math_person_configuration as M


def truth():
    """Analytic probability fixture, not a fabricated model observation."""
    x = M.FIELD_POINTS
    return (0.5 + 0.3 * x[:, 0] * x[:, 1] * x[:, 2] * x[:, 3] * x[:, 4]).reshape(
        2, 2, 2, 2, 2
    )


@pytest.mark.parametrize("reference", [-1, 1])
def test_explicit_pair_selection_and_class_dependence(reference):
    law, probability = M.selected_pair_law(reference)
    assert probability == pytest.approx(0.125, abs=1e-15)
    assert law.shape == (2, 2, 8)
    np.testing.assert_allclose(law.sum(), 1)
    np.testing.assert_allclose(law.sum(axis=(1, 2)), [0.5, 0.5])
    np.testing.assert_allclose(law.sum(axis=(0, 2)), [0.5, 0.5])
    table = law.sum(2)
    np.testing.assert_allclose(table, [[0.16, 0.34], [0.34, 0.16]])
    assert not np.allclose(table, np.full((2, 2), 0.25))


def test_both_complete_person_posteriors_use_own_configuration():
    for b, c, p in itertools.product(M.POINTS, (-1, 1), (-1, 1)):
        np.testing.assert_allclose(
            M.pair_posterior(b, c, p), 0.5 + 0.3 * p * c * np.prod(b)
        )
        np.testing.assert_allclose(
            M.pair_posterior(b, c, p), 1 - M.pair_posterior(b, c, -p)
        )


def test_all_lower_observations_are_nondiagnostic_after_pair_selection():
    for c in (-1, 1):
        law, _ = M.selected_pair_law(c)
        for degree in range(3):
            for indices in itertools.combinations(range(3), degree):
                for values in itertools.product((-1, 1), repeat=degree):
                    mask = np.ones(8, dtype=bool)
                    for i, v in zip(indices, values, strict=True):
                        mask &= M.POINTS[:, i] == v
                    joint = law[:, :, mask].sum(2)
                    np.testing.assert_allclose(
                        [joint[0].sum() / joint.sum(), joint[:, 0].sum() / joint.sum()],
                        [0.5, 0.5],
                    )


def test_all_32_characters_and_full_reconstruction():
    h = M.character_matrix()
    np.testing.assert_array_equal(h.T @ h, np.eye(32) * 32)
    beta = M.coefficients(truth())
    want = np.zeros(32)
    want[[0, 31]] = [0.5, 0.3]
    np.testing.assert_allclose(beta, want, atol=1e-15)
    np.testing.assert_allclose(M.reconstruct(beta), truth(), atol=1e-15)


def test_first_choice_fact_has_person_binding_but_not_joint_diagnosis():
    x = M.FIELD_POINTS
    fact = (0.5 + 0.5 * x[:, 0] * x[:, 4]).reshape(2, 2, 2, 2, 2)
    beta = M.coefficients(fact)
    assert beta[17] == 0.5
    assert beta[31] == 0
    assert M.exchange_defect(fact) == {"rms": 0.0, "maximum": 0.0}


def test_general_context_joint_response_is_not_person_diagnosis():
    x = M.FIELD_POINTS
    context = (0.5 + 0.3 * np.prod(x[:, :4], axis=1)).reshape(2, 2, 2, 2, 2)
    assert M.coefficients(context)[15] == pytest.approx(0.3)
    assert M.coefficients(context)[31] == pytest.approx(0)
    _, personal = M.person_sectors(context)
    np.testing.assert_array_equal(personal, np.zeros((2, 2, 2, 2)))


def test_fixed_bag_observer_cannot_recover_assignment_interactions():
    observer = M.bag_observer(truth())
    np.testing.assert_allclose(observer, 0.5)
    assert M.coefficients(observer)[31] == pytest.approx(0)
    for mask in range(32):
        if mask & 7:
            assert M.coefficients(observer)[mask] == pytest.approx(0)


def test_actor_exchange_and_separate_common_person_sectors():
    common, personal = M.person_sectors(truth())
    np.testing.assert_allclose(common, 0.5)
    x = M.FIELD_POINTS.reshape(2, 2, 2, 2, 2, 5)
    np.testing.assert_allclose(personal, 0.3 * np.prod(x[..., 0, :4], axis=-1))
    assert M.exchange_defect(truth())["maximum"] < 1e-15
    biased = truth() + np.broadcast_to([-0.1, 0.1], truth().shape)
    assert M.exchange_defect(biased)["rms"] == pytest.approx(0.2)


@pytest.mark.parametrize("person", [0, 2, -2])
def test_unknown_person_rejected(person):
    with pytest.raises(ValueError):
        M.pair_posterior([-1, 1, 1], 1, person)


def test_incomplete_or_nonfinite_fields_rejected():
    for fn in (M.coefficients, M.person_sectors, M.exchange_defect, M.bag_observer):
        with pytest.raises(ValueError):
            fn(np.zeros((2, 2, 2, 2)))
        with pytest.raises(ValueError):
            fn(np.full((2, 2, 2, 2, 2), np.nan))
    with pytest.raises(ValueError):
        M.reconstruct(np.zeros(31))


def fake_rows():
    """Integer-only complete token-bag/scope fixture, not native model data."""
    from grain_research.construction import math_joint_diagnostic as J

    rows = []
    for bank, b, c in itertools.product(range(2), M.POINTS, (-1, 1)):
        action = list(
            itertools.chain.from_iterable(
                [31, 11, 32, 12] if sign == 1 else [32, 11, 31, 12] for sign in b
            )
        )
        reference = J.population_counts(c)[0].tolist()
        rows.append(
            {
                "id": len(rows),
                "bank": bank,
                "b": b.tolist(),
                "c": c,
                "history": list(range(2, 22)),
                "current": [23],
                "action": list(range(2, 14)),
                "reference": list(range(14, 22)),
                "reference_slot": list(range(14, 22)),
                "padding": [],
                "event_spans": [list(range(2 + 4 * i, 6 + 4 * i)) for i in range(3)],
                "input_ids": [101, 102] + action + reference + [103, 104],
            }
        )
    return rows


def test_entire_wordbag_and_frozen_design():
    from scripts import run_grain_person_configuration_hm146 as R

    rows = fake_rows()
    bags = {tuple(sorted(Counter(r["input_ids"]).items())) for r in rows}
    assert len(bags) == 1
    design = R.fixed_design(rows)
    assert len(design) == 164
    assert Counter(d["mode"] for d in design) == {
        "capture": 32,
        "noop": 4,
        "action_k": 32,
        "action_v": 32,
        "action_kv": 32,
        "reference_kv": 32,
    }
    for entry in design[36:]:
        r, d = (rows[entry[k]] for k in ("row", "donor"))
        assert r["bank"] == d["bank"]
        if entry["mode"] == "reference_kv":
            assert r["b"] == d["b"] and r["c"] == -d["c"]
        else:
            assert r["c"] == d["c"] and r["b"] == [-x for x in d["b"]]


def test_bag_or_event_scope_violation_fails():
    from scripts import run_grain_person_configuration_hm146 as R

    with pytest.raises(ValueError):
        R.validate_rows(fake_rows()[:-1])
    rows = fake_rows()
    rows[0]["input_ids"][2] = 909
    with pytest.raises(ValueError):
        R.validate_rows(rows)
    rows = fake_rows()
    rows[0]["event_spans"][0] = []
    with pytest.raises(ValueError):
        R.validate_rows(rows)


def test_independent_full_selection_audit():
    from scripts import run_grain_person_configuration_hm146 as R

    audit = R.law_audit()
    assert audit["maximum_error"] < 1e-12
    assert audit["checks"] > 600
    for law in audit["selected_laws"]:
        assert law["class_correlation"] == pytest.approx(-0.36)


def test_complete_outputs_keep_all_signed_components():
    from scripts import run_grain_person_configuration_hm146 as R

    cube = np.empty((2, 2, 2, 2, 2, 5, 3, 2, 4))
    native = truth()
    for bank, mode, family, unit in itertools.product(
        range(2), range(5), range(3), range(4)
    ):
        cube[bank, :, :, :, :, mode, family, :, unit] = native
    rows, summary, error = R.outputs(cube)
    assert len(rows) == 3840 and len(summary) == 24 and error < 1e-12
    for row in summary:
        assert row["joint_person_coefficient"] == pytest.approx(0.3)
        assert row["predicted_personal_reversal"]
