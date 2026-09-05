"""Official source membership with explicit bridge-priority exclusion."""

from __future__ import annotations


def source_partitions(sets):
    """Do not assume a prepared bridge is identical to an official-task intersection."""
    big5, bridge = set(sets["big5"]), set(sets["bridge"])
    axes = {k: set(v) for k, v in sets.items() if k.endswith("_cont")}
    if len(axes) != 4:
        raise ValueError("Need all four official MBTI source memberships")
    mbti = set().union(*axes.values())
    sources = {
        "big5_all": big5,
        "big5_only": big5 - mbti - bridge,
        "bridge_strict_report": bridge,
        "any_both_membership": big5 & mbti,
        "both_nonreport_reserved": (big5 & mbti) - bridge,
        "bridge_outside_official_mbti": bridge - mbti,
    }
    sources.update({k + "_only": v - big5 - bridge for k, v in axes.items()})
    if sources["big5_only"] & bridge or any(
        sources[k + "_only"] & bridge for k in axes
    ):
        raise ValueError("Bridge contamination in source-head members")
    return sources


def exclusive_source_groups(sets):
    """Return disjoint source categories; all other both-label members stay reserved."""
    parts = source_partitions(sets)
    return {
        "big5_only": parts["big5_only"],
        "mbti_only": set().union(
            *(v for k, v in parts.items() if k.endswith("_cont_only"))
        ),
        "bridge": parts["bridge_strict_report"],
        "both_nonreport_reserved": parts["both_nonreport_reserved"],
    }
