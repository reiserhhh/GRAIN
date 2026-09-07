"""Arithmetic counterexamples and failure-mode tests for the output-only audit."""

import pandas as pd
import pytest

from grain_research.vector_mechanisms.reciprocity_audit import (
    check_panel,
    component_summary,
    gap_changes,
    polarity_components,
)


def fixture_panel(*, semantic=2.0, bias=3.0):
    """Create explicit semantic and answer-bias effects, without model simulation."""
    rows, base, effects = [], [], []
    config = {"sites": ["block_16"], "doses": [-1, 1], "random_replicates": 1}
    for group in ("g1", "g2"):
        for variant in ("R_direct", "R_negative"):
            for polarity in (-1, 1):
                case = f"{group}_{variant}_{polarity}"
                row = {
                    "case_id": case,
                    "group_id": group,
                    "variant": variant,
                    "polarity": polarity,
                    "split": "evaluation",
                    "task": "R",
                    "expected_sign": polarity,
                }
                rows.append(row)
                base.append(row | {"margin": polarity * 4.0})
                for source in ("G", "R"):
                    for kind, replica in (("signal", -1), ("random", 0)):
                        slope = (
                            (semantic + (bias if variant == "R_direct" else -bias))
                            if kind == "signal"
                            else 0.0
                        )
                        for dose in (-1.0, 1.0):
                            effects.append(
                                row
                                | {
                                    "source": source,
                                    "site": "block_16",
                                    "kind": kind,
                                    "replica": replica,
                                    "dose": dose,
                                    "margin": polarity * 4.0 + dose * slope,
                                    "delta": dose * slope,
                                }
                            )
    return rows, config, pd.DataFrame(base), pd.DataFrame(effects)


def test_output_coverage_requires_every_cell_and_no_duplicates():
    rows, config, base, effects = fixture_panel()
    assert check_panel(rows, config, base, effects)["maximum_delta_error"] == 0
    for bad in (effects.iloc[1:], pd.concat([effects, effects.iloc[:1]])):
        with pytest.raises(ValueError, match="intervention cell"):
            check_panel(rows, config, base, bad)


def test_wrong_delta_and_truth_metadata_are_rejected():
    rows, config, base, effects = fixture_panel()
    effects.loc[0, "delta"] += 1
    with pytest.raises(ValueError, match="delta"):
        check_panel(rows, config, base, effects)
    _, _, _, effects = fixture_panel()
    base.loc[0, "expected_sign"] *= -1
    with pytest.raises(ValueError, match="metadata"):
        check_panel(rows, config, base, effects)


@pytest.mark.parametrize("semantic,bias", [(2.0, 3.0), (0.0, 3.0), (2.0, 0.0)])
def test_even_odd_exact_for_an_explicit_additive_counterexample(semantic, bias):
    _, _, _, effects = fixture_panel(semantic=semantic, bias=bias)
    components = polarity_components(effects)
    signal = components[components.kind == "signal"]
    assert (signal.polarity_even == semantic).all()
    assert (signal.polarity_odd == bias).all()
    summary = component_summary(components)
    assert set(summary.n_scenarios) == {2}
    assert (
        summary[summary.component == "polarity_even"].paired_ci_low == semantic
    ).all()
    assert (summary[summary.component == "polarity_odd"].paired_ci_low == bias).all()


def test_unpaired_question_forms_are_not_silently_dropped():
    _, _, _, effects = fixture_panel()
    with pytest.raises(ValueError, match="Unpaired"):
        polarity_components(effects[effects.variant == "R_direct"])


def test_shifting_both_classes_does_not_improve_discrimination():
    _, _, baseline, effects = fixture_panel()
    result = gap_changes(baseline, effects)
    assert (result.native_gap == 8).all()
    assert (result.plus_direction_gap == 8).all()
    assert (result.gap_change == 0).all()
