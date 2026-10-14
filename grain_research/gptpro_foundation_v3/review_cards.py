#!/usr/bin/env python3
"""Review/acceptance card builder: rules bound to machine-checked evidence.

A review card states production rules. Every rule cites evidence selectors that
are resolved against the actual result files; the stated value must match the
file value within the declared rounding tolerance, otherwise the card build
fails. Implementation, protocol and science statuses are kept separate. Passing
a review build is bookkeeping consistency, not scientific validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

VERDICTS = {
    "SUPPORTED_FINITE",
    "SUPPORTED_LOWER_LOSS",
    "HIGHER_LOSS",
    "INCONCLUSIVE",
    "EXACT_CERTIFICATE",
    "COUNTEREXAMPLE",
    "KNOWN_METHOD_SUFFICIENT",
    "NO_INCREMENT",
    "INFORMATION_LOSS",
    "NOT_IDENTIFIED",
    "BOUNDS_ONLY",
    "BLOCKED",
    "HUMAN_PENDING",
    "OPEN",
}
STATUS = {"DONE", "PARTIAL", "OPEN", "BLOCKED", "FAILED"}
SCIENCE = {
    "SUPPORTED_FINITE",
    "FINITE_CERTIFICATE",
    "PARTIAL",
    "INCONCLUSIVE",
    "OPEN",
    "HUMAN_PENDING",
    "BLOCKED",
}


@dataclass
class Evidence:
    """One value in one file. `select` filters CSV rows; `key` walks JSON."""

    path: str
    field: str | None = None
    select: dict | None = None
    key: list | None = None
    expect: float | str | bool | None = None
    tol: float = 5e-4
    note: str = ""


@dataclass
class Rule:
    """One production rule with scope, requirement flags and verdict."""

    rule_id: str
    scope: str
    statement: str
    verdict: str
    evidence: list[Evidence]
    needs: list[str] = field(default_factory=list)
    standard_method: str = ""
    information: str = ""


@dataclass
class Card:
    """A review card specification."""

    card_id: str
    package: str
    artifact: str
    requirements: list[str]
    depends_on: list[str]
    rules: list[Rule]
    statuses: dict
    gaps: list[dict]
    non_claims: list[str]
    derived: list[Callable] = field(default_factory=list)
    handoffs: list[dict] = field(default_factory=list)
    extra_checks: list[Callable] = field(default_factory=list)


def sha256(path: Path) -> str:
    """File digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def resolve(root: Path, ev: Evidence, derived: list[dict] | None = None):
    """Return (value, sha256) for one evidence selector; raise on ambiguity.

    Paths `derived:<quantity>` resolve against this build's derived rows.
    """
    if ev.path.startswith("derived:"):
        rows = [r for r in derived or [] if r.get("quantity") == ev.path[8:]]
        if len(rows) != 1:
            raise ValueError(f"{ev.path}: {len(rows)} derived rows")
        return _number(rows[0][ev.field]) if not isinstance(rows[0][ev.field], bool) else rows[0][ev.field], "derived"
    path = root / ev.path
    if not path.exists():
        raise FileNotFoundError(ev.path)
    if path.suffix == ".csv":
        with path.open() as f:
            rows = [
                r
                for r in csv.DictReader(f)
                if all(str(r.get(k)) == str(v) for k, v in (ev.select or {}).items())
            ]
        if len(rows) != 1:
            raise ValueError(f"{ev.path} {ev.select}: {len(rows)} rows")
        value = _number(rows[0][ev.field])
    elif path.suffix == ".json":
        value = json.loads(path.read_text())
        for k in ev.key or []:
            if isinstance(value, list) and isinstance(k, dict):
                matches = [v for v in value if all(v.get(a) == b for a, b in k.items())]
                if len(matches) != 1:
                    raise ValueError(f"{ev.path} {k}: {len(matches)} matches")
                value = matches[0]
            else:
                value = value[k]
        value = _number(value) if not isinstance(value, (list, dict)) else value
    else:
        value = None
    return value, sha256(path)


def check(value, ev: Evidence) -> bool:
    """Stated value equals file value within tolerance (or exactly for text)."""
    if ev.expect is None:
        return True
    if isinstance(ev.expect, bool) or isinstance(value, bool):
        return value == ev.expect
    if isinstance(ev.expect, (int, float)) and isinstance(value, (int, float)):
        return math.isclose(value, ev.expect, abs_tol=ev.tol)
    return str(value) == str(ev.expect)


def build(card: Card, root: Path, output: Path, atomic: dict) -> dict:
    """Resolve evidence, run derived computations, write COMMON 8 + card files."""
    output.mkdir(parents=True, exist_ok=False)
    log = []
    started = time.time()
    definition = next(t for t in atomic["tasks"] if t["id"] == card.card_id)
    inputs, metrics, tests = {}, [], []
    derived_rows = []
    for fn in card.derived:
        rows, used = fn(root)
        derived_rows.extend(rows)
        for p in used:
            inputs[p] = sha256(root / p)
        log.append({"event": "derived", "function": fn.__name__, "rows": len(rows)})
    for rule in card.rules:
        if rule.verdict not in VERDICTS:
            raise ValueError(f"{rule.rule_id}: verdict {rule.verdict}")
        if not rule.evidence:
            raise ValueError(f"{rule.rule_id}: no evidence")
        for i, ev in enumerate(rule.evidence):
            value, digest = resolve(root, ev, derived_rows)
            if digest != "derived":
                inputs[ev.path] = digest
            ok = check(value, ev)
            tests.append(
                f"{'PASS' if ok else 'FAIL'} {rule.rule_id}#{i} {ev.path} "
                f"{ev.select or ev.key} {ev.field or ''} file={value!r} stated={ev.expect!r}"
            )
            metrics.append(
                {
                    "rule_id": rule.rule_id,
                    "evidence_index": i,
                    "path": ev.path,
                    "selector": json.dumps(ev.select or ev.key, ensure_ascii=False),
                    "field": ev.field or "",
                    "file_value": json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value,
                    "stated_value": ev.expect,
                    "tolerance": ev.tol,
                    "match": ok,
                    "note": ev.note,
                }
            )
            log.append({"event": "evidence", "rule": rule.rule_id, "index": i, "match": ok})
    for fn in card.extra_checks:
        name, ok, detail = fn(root)
        tests.append(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
        log.append({"event": "check", "name": name, "ok": ok})
    failed = [t for t in tests if t.startswith("FAIL")]
    for key, allowed in (("implementation", STATUS), ("protocol", STATUS), ("science", SCIENCE)):
        if card.statuses[key] not in allowed:
            raise ValueError(f"status {key}={card.statuses[key]}")
    rules_json = {
        "card": card.card_id,
        "package": card.package,
        "definition": {k: definition[k] for k in ("instruction", "acceptance_check", "artifacts", "requirements", "depends_on")},
        "rules": [
            {
                "rule_id": r.rule_id,
                "scope": r.scope,
                "statement": r.statement,
                "verdict": r.verdict,
                "needs": r.needs,
                "standard_method": r.standard_method,
                "information": r.information,
                "evidence": [
                    {"path": e.path, "select": e.select, "key": e.key, "field": e.field, "value": e.expect, "note": e.note}
                    for e in r.evidence
                ],
            }
            for r in card.rules
        ],
        "handoffs": card.handoffs,
    }
    acceptance = {
        "card": card.card_id,
        "evidence_checks": {"total": len(tests), "failed": len(failed)},
        "statuses": card.statuses,
        "acceptance_decision": "REVIEW_BUILT_EVIDENCE_CONSISTENT" if not failed else "REVIEW_BUILD_FAILED",
        "gaps": card.gaps,
        "non_claims": card.non_claims,
        "parent_card_promoted": False,
        "note": "Evidence consistency is bookkeeping, not scientific validation.",
    }
    (output / card.artifact).write_text(json.dumps(rules_json, indent=1, ensure_ascii=False) + "\n")
    (output / "acceptance.json").write_text(json.dumps(acceptance, indent=1, ensure_ascii=False) + "\n")
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "card": card.card_id,
                "method": "review builder: every rule's stated numbers resolved against result files (rounding tolerance per selector); derived computations recorded in derived_metrics.csv",
                "atomic_definition": definition["instruction"],
                "acceptance_check": definition["acceptance_check"],
                "requirements": card.requirements,
                "depends_on": card.depends_on,
                "output_path_deviation": "COMMON root FXX/r1/<card> holds rejected legacy outputs; new revision written under FXX/review_r1/<card>",
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    (output / "input_manifest.json").write_text(json.dumps(dict(sorted(inputs.items())), indent=1) + "\n")
    with (output / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0]))
        w.writeheader()
        w.writerows(metrics)
    if derived_rows:
        with (output / "derived_metrics.csv").open("w", newline="") as f:
            keys = sorted({k for r in derived_rows for k in r})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(derived_rows)
    (output / "tests.txt").write_text("\n".join(tests) + "\n")
    lines = [
        f"# {card.card_id} review decision",
        "",
        f"Statuses: implementation {card.statuses['implementation']}, protocol {card.statuses['protocol']}, science {card.statuses['science']}.",
        f"Evidence checks: {len(tests) - len(failed)}/{len(tests)} PASS.",
        "",
        "## Rules",
        "",
    ]
    for r in card.rules:
        lines.append(f"- **{r.rule_id}** [{r.verdict}] ({r.scope}) {r.statement}")
        if r.needs:
            lines.append(f"  - needs: {', '.join(r.needs)}")
        if r.standard_method:
            lines.append(f"  - standard method: {r.standard_method}")
        if r.information:
            lines.append(f"  - information: {r.information}")
    if card.handoffs:
        lines += ["", "## Handoffs", ""] + [f"- {h['to']}: {h['what']}" for h in card.handoffs]
    lines += ["", "## Remaining gaps", ""] + [f"- {g['gap']} -> {g['next']}" for g in card.gaps]
    lines += ["", "## Non-claims", ""] + [f"- {n}" for n in card.non_claims]
    (output / "decision.md").write_text("\n".join(lines) + "\n")
    log.append({"event": "finished", "seconds": time.time() - started, "python": sys.version.split()[0], "platform": platform.platform()})
    (output / "run_log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
    (output / "summary.json").write_text(
        json.dumps(
            {
                "card": card.card_id,
                "rules": len(card.rules),
                "verdicts": {v: sum(r.verdict == v for r in card.rules) for v in sorted({r.verdict for r in card.rules})},
                "evidence_checks": len(tests),
                "failed_checks": len(failed),
                "derived_rows": len(derived_rows),
                "statuses": card.statuses,
            },
            indent=1,
        )
        + "\n"
    )
    outputs = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.name != "output_manifest.json"}
    (output / "output_manifest.json").write_text(json.dumps(outputs, indent=1) + "\n")
    return {"failed": failed, "tests": len(tests)}
