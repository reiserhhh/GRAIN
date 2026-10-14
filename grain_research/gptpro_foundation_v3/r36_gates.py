"""R36 stage gates for the phenomenon pipeline (P03 p.32; P04 pp.27-31)."""

from __future__ import annotations

STAGES = ("replicated_regularity", "structural_reliability", "admission", "shortcut_control",
          "rival_explanations", "discriminating_intervention", "reader_gauge",
          "transport", "prospective_validation")
STATUSES = ("PASS", "FAIL", "BLOCKED", "NOT_RUN")


def gates(record: dict) -> dict:
    """Return per-stage status and the first gate that is not PASS (stages are ordered)."""
    out, first = {}, None
    for s in STAGES:
        st = record.get(s, "NOT_RUN")
        if st not in STATUSES:
            raise ValueError(f"unknown status {st} for {s}")
        out[s] = st
        if first is None and st != "PASS":
            first = s
    return {"stages": out, "first_non_pass": first, "complete": first is None}
