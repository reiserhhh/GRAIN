"""Independent output checks and query-polarity decomposition for the P1 panel."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .analysis import grouped_bootstrap

KEYS = ["case_id", "source", "site", "kind", "replica", "dose"]


def check_panel(rows, config, baselines, effects):
    """Require exactly the declared Cartesian panel and correct recorded deltas."""
    cases = {r["case_id"]: r for r in rows}
    if len(cases) != len(rows) or baselines.case_id.duplicated().any():
        raise ValueError("Duplicate material or baseline")
    if set(baselines.case_id) != set(cases):
        raise ValueError("Baseline coverage mismatch")
    expected = set()
    arms = [("signal", -1, float(d)) for d in config["doses"]] + [
        ("random", replica, dose)
        for replica in range(config["random_replicates"])
        for dose in (-1.0, 1.0)
    ]
    for row, source, site, (kind, replica, dose) in itertools.product(
        [r for r in rows if r["split"] == "evaluation"],
        ("G", "R"),
        config["sites"],
        arms,
    ):
        expected.add((row["case_id"], source, site, kind, replica, dose))
    actual = set(effects[KEYS].itertuples(index=False, name=None))
    if effects.duplicated(KEYS).any() or actual != expected:
        raise ValueError("Incomplete, duplicate, or unexpected intervention cell")
    for frame in (baselines, effects):
        numeric = frame.select_dtypes(include="number")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("Non-finite panel data")
        for row in frame.itertuples(index=False):
            reference = cases[row.case_id]
            keys = ["group_id", "variant", "polarity", "task"]
            keys += [k for k in ("expected_sign", "split") if k in frame.columns]
            for key in keys:
                if getattr(row, key) != reference[key]:
                    raise ValueError("Material metadata changed")
    base = baselines.set_index("case_id").margin
    expected_delta = effects.margin.to_numpy() - effects.case_id.map(base).to_numpy()
    error = float(np.max(np.abs(expected_delta - effects.delta.to_numpy())))
    if error > 1e-9:
        raise ValueError("Stored intervention delta does not match baseline")
    return {
        "baseline_rows": len(baselines),
        "effect_rows": len(effects),
        "maximum_delta_error": error,
    }


def signed_slopes(effects):
    """Compute each paired +/- effect without mixing opposite question codings."""
    index = [
        "case_id",
        "group_id",
        "variant",
        "polarity",
        "source",
        "site",
        "kind",
        "replica",
    ]
    wide = effects[effects.dose.isin([-1, 1])].pivot(
        index=index, columns="dose", values="margin"
    )
    if wide.isna().any().any() or not {-1, 1} <= set(wide.columns):
        raise ValueError("Missing signed intervention")
    return ((wide[1] - wide[-1]) / 2).rename("slope").reset_index()


def polarity_components(effects):
    """Decompose paired query effects into even/odd components, not latent causes.

    The interpretation as semantic vs answer bias is only a diagnostic model:
    other computations can also be sensitive to question polarity.
    """
    slopes = signed_slopes(effects)
    records = []
    for positive, negative, family in (
        ("R_direct", "R_negative", "action"),
        ("R_trait", "R_trait_negative", "trait_attribution"),
    ):
        sub = slopes[slopes.variant.isin([positive, negative])]
        if sub.empty:
            continue
        index = ["group_id", "polarity", "source", "site", "kind", "replica"]
        pairs = sub.pivot(index=index, columns="variant", values="slope")
        if set(pairs.columns) != {positive, negative} or pairs.isna().any().any():
            raise ValueError("Unpaired query polarity")
        pairs["polarity_even"] = (pairs[positive] + pairs[negative]) / 2
        pairs["polarity_odd"] = (pairs[positive] - pairs[negative]) / 2
        pairs["family"] = family
        records.extend(
            pairs.reset_index()[
                index + ["polarity_even", "polarity_odd", "family"]
            ].to_dict("records")
        )
    return pd.DataFrame(records)


def component_summary(components):
    """Cluster on scenarios, with random controls paired inside each material."""
    rows = []
    for (family, source, site), sub in components.groupby(["family", "source", "site"]):
        index = ["group_id", "polarity"]
        signal = sub[sub.kind == "signal"].set_index(index)
        random = (
            sub[sub.kind == "random"]
            .groupby(index)[["polarity_even", "polarity_odd"]]
            .mean()
        )
        random = random.reindex(signal.index)
        if random.isna().any().any():
            raise ValueError("Missing paired random query control")
        groups = signal.index.get_level_values("group_id")
        for component in ("polarity_even", "polarity_odd"):
            raw = grouped_bootstrap(signal[component].to_numpy(), groups)
            delta = grouped_bootstrap(
                (signal[component] - random[component]).to_numpy(), groups
            )
            rows.append(
                {
                    "family": family,
                    "source": source,
                    "site": site,
                    "component": component,
                    "signal": raw["mean"],
                    "signal_ci_low": raw["ci_low"],
                    "signal_ci_high": raw["ci_high"],
                    "random": float(random[component].mean()),
                    "signal_minus_random": delta["mean"],
                    "paired_ci_low": delta["ci_low"],
                    "paired_ci_high": delta["ci_high"],
                    "n_scenarios": raw["clusters"],
                }
            )
    return pd.DataFrame(rows)


def gap_changes(baselines, effects):
    """Distinguish shifting every answer from changing high/low discrimination."""
    base = baselines[baselines.split == "evaluation"].pivot(
        index=["group_id", "variant"], columns="polarity", values="margin"
    )
    base_gap = base[1] - base[-1]
    signal = effects[(effects.kind == "signal") & (effects.dose == 1)]
    rows = []
    for (variant, source, site), sub in signal.groupby(["variant", "source", "site"]):
        pair = sub.pivot(index="group_id", columns="polarity", values="margin")
        gap = pair[1] - pair[-1]
        native = base_gap.xs(variant, level="variant").reindex(gap.index)
        ci = grouped_bootstrap((gap - native).to_numpy(), gap.index)
        rows.append(
            {
                "variant": variant,
                "source": source,
                "site": site,
                "native_gap": float(native.mean()),
                "plus_direction_gap": float(gap.mean()),
                "gap_change": ci["mean"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "n_scenarios": ci["clusters"],
            }
        )
    return pd.DataFrame(rows)
