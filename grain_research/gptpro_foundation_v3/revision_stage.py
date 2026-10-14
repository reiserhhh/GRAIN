"""Revision registry and fail-closed stage modes (packet 12 Part C, repaired in packet 13 Part A; F00-04 / R35).

Modes, kept distinct:
- integrity: re-hash declared run-time manifests. Declared append-only amendments are verified by exact prefix hashes.
  A root without any run-time manifest may carry a dated current integrity snapshot. That authenticates today's
  bytes only, and never counts as historical provenance.
- readiness: dependency and evidence readiness for dependent use. It reports provenance, integrity basis and
  scientific status as separate fields. Rejected legacy output stays readable history and never grants readiness.
- reproduce / execute: bounded replays. A preflight runs BEFORE any child process and refuses unless every declared
  input and code file exists and matches its pin. Each input must map to a registered root (or be a declared external
  reference). The authorization and compute cap must match. The dependency contract must hold: scientific adapters
  need READY dependencies, and diagnostic adapters may read only explicitly listed, pinned history. A diagnostic PASS
  never changes readiness.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results/grain_gptpro_foundation_v3"
REGISTRY = REPO / "configs/grain_v3_revision_registry.json"
SNAPSHOTS = RES / "F00/integrity_snapshots_r1/snapshots"
READY = "READY_FOR_DEPENDENT_USE"
OK_STATES = ("OK", "OK_DECLARED_AMENDMENTS", "OK_CURRENT_SNAPSHOT")
# Packet 17 Part D: typed integrity evidence levels (ascending). A current snapshot is a LIMITED input: it authenticates
# today's bytes, never historical run-time provenance or preregistration. Consuming contracts must permit it explicitly.
EVIDENCE_LEVELS = ("none", "current_snapshot_limited", "runtime_manifest")
DEFAULT_PERMITTED_EVIDENCE = ("runtime_manifest",)  # fail-closed default for contracts that declare nothing


class StageRefusal(RuntimeError):
    """A fail-closed refusal; the CLI maps it to exit status 2."""


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("._") and not any(x.startswith("._") for x in p.relative_to(root).parts))


def tree_sha(root: Path) -> str:
    lines = "".join(f"{p.relative_to(root)}\t{sha(p)}\n" for p in tree_files(root))
    return hashlib.sha256(lines.encode()).hexdigest()


def resolve_root(root: str) -> Path:
    p = Path(root)
    if p.is_absolute():
        return p
    return REPO / root if root.startswith(("results/", "reports/", "docs/", "tests/", "configs/", "scripts/", "grain_research/")) else RES / root


def snapshot_path(root_key: str) -> Path:
    return SNAPSHOTS / (root_key.replace("/", "__") + ".json")


def load_registry(path: Path = REGISTRY) -> dict:
    reg = json.loads(path.read_text())
    if reg.get("schema") != "grain-v3-revision-registry-v1":
        raise StageRefusal("unknown registry schema")
    return reg


# ---------------------------------------------------------------- integrity
def manifest_entries(root: Path) -> list[tuple[str, Path, str]]:
    """(source manifest, file, recorded sha) from every declared run-time manifest under a root."""
    out = []
    om = root / "output_manifest.json"
    if om.exists():
        data = json.loads(om.read_text())
        if isinstance(data, dict) and isinstance(data.get("files"), list):
            items = [(f.get("path"), f.get("sha256")) for f in data["files"] if isinstance(f, dict)]
        elif isinstance(data, list):
            items = [(f.get("path"), f.get("sha256")) for f in data if isinstance(f, dict)]
        elif isinstance(data, dict):
            items = [(k, v) for k, v in data.items() if isinstance(v, str)]
        else:
            items = []
        for name, digest in items:
            if isinstance(name, str) and isinstance(digest, str) and len(digest) == 64:
                target = root / name if (root / name).exists() or not (REPO / name).exists() else REPO / name
                out.append(("output_manifest.json", target, digest))
    bi = root / "bundle_index.json"
    if bi.exists():
        for e in json.loads(bi.read_text()):
            if isinstance(e, dict) and isinstance(e.get("sha256"), str) and e.get("name"):
                out.append(("bundle_index.json", root / "bundles" / f"{e['name']}.json", e["sha256"]))
    cr = root / "common_record_r1"
    if (cr / "output_manifest.json").exists():
        for f in json.loads((cr / "output_manifest.json").read_text())["files"]:
            out.append(("common_record_r1/output_manifest.json", cr / f["path"], f["sha256"]))
    if (cr / "input_manifest.json").exists():
        for f in json.loads((cr / "input_manifest.json").read_text()).get("packet_files", []):
            out.append(("common_record_r1/input_manifest.json", root / f["path"], f["sha256"]))
    an = root / "analysis.json"
    if an.exists():
        data = json.loads(an.read_text())
        for key in ("output_sha256", "episode_files_sha256"):
            block = data.get(key) if isinstance(data, dict) else None
            if not isinstance(block, dict):
                continue
            for name, digest in block.items():
                if isinstance(name, str) and isinstance(digest, str) and len(digest) == 64:
                    out.append((f"analysis.json:{key}", root / name, digest))
    return out


def verify_amendment(path: Path, amendment: dict) -> bool:
    data = path.read_bytes()
    n = amendment["recorded_prefix_bytes"]
    return (hashlib.sha256(data[:n]).hexdigest() == amendment["recorded_sha256"]
            and hashlib.sha256(data).hexdigest() == amendment["current_sha256"] and len(data) > n)


def _snapshot_integrity(root_key: str, root: Path) -> dict | None:
    sp = snapshot_path(root_key)
    if not sp.exists():
        return None
    snap = json.loads(sp.read_text())
    problems = []
    listed = set(snap["files"])
    for rel, digest in snap["files"].items():
        p = root / rel
        if not p.exists():
            problems.append(f"MISSING {rel} (current snapshot {snap['snapshot_date']})")
        elif sha(p) != digest:
            problems.append(f"CHANGED {rel} (current snapshot {snap['snapshot_date']})")
    excluded = tuple(snap.get("excluded_prefixes", []))
    for p in tree_files(root):
        rel = str(p.relative_to(root))
        if rel not in listed and not rel.startswith(excluded):
            problems.append(f"NEW_AFTER_SNAPSHOT {rel}")
    return {"root": root_key, "integrity": "FAIL" if problems else "OK_CURRENT_SNAPSHOT", "checked": len(listed), "problems": problems,
            "declared_amendments_verified": [], "sidecars_ignored": 0, "integrity_basis": f"current_snapshot_{snap['snapshot_date']}",
            "historical_runtime_manifest_available": snap.get("historical_runtime_manifest_available", False)}


def integrity(root_key: str, entry: dict | None) -> dict:
    root = resolve_root(root_key)
    if not root.is_dir():
        return {"root": root_key, "integrity": "MISSING_ROOT", "checked": 0, "problems": [f"{root_key} absent"], "integrity_basis": "none"}
    amend = {a["file"]: a for a in (entry or {}).get("amendments", [])}
    checked, problems, amended, sidecars = 0, [], [], 0
    seen = set()
    for source, path, digest in manifest_entries(root):
        if path.name.startswith("._"):
            sidecars += 1
            continue
        key = (str(path), digest)
        if key in seen:
            continue
        seen.add(key)
        checked += 1
        if not path.exists():
            problems.append(f"MISSING {path} ({source})")
            continue
        if sha(path) == digest:
            continue
        rel = str(path.relative_to(root)) if root in path.parents else str(path)
        a = amend.get(rel)
        if a and a.get("authority") and a.get("recorded_prefix_bytes") is not None and a["recorded_sha256"] == digest and verify_amendment(path, a):
            amended.append(rel)
        else:
            problems.append(f"CHANGED {rel} ({source})")
    if checked == 0:
        snap = _snapshot_integrity(root_key, root)
        if snap is not None:
            return snap
        return {"root": root_key, "integrity": "NO_MANIFEST", "checked": 0, "problems": [], "integrity_basis": "none",
                "historical_runtime_manifest_available": False}
    status = "FAIL" if problems else ("OK_DECLARED_AMENDMENTS" if amended else "OK")
    return {"root": root_key, "integrity": status, "checked": checked, "problems": problems,
            "declared_amendments_verified": sorted(set(amended)), "sidecars_ignored": sidecars,
            "integrity_basis": "runtime_manifest" + ("+declared_append_only_amendments" if amended else ""),
            "historical_runtime_manifest_available": True}


def test_record_ok(rec: dict) -> bool:
    """A registered test-record reconciliation: file exists, live hash matches the pin, first line exit_code=0."""
    p = resolve_root(rec["path"])
    return p.is_file() and sha(p) == rec.get("sha256") and p.read_text().startswith("exit_code=0\n")


def own_evidence_level(integ: dict) -> str:
    """Evidence level of one root from its live integrity check (never from a recorded label)."""
    if integ.get("integrity") not in OK_STATES:
        return "none"
    basis = integ.get("integrity_basis") or "none"
    if basis.startswith("runtime_manifest"):
        return "runtime_manifest"
    if basis.startswith("current_snapshot"):
        return "current_snapshot_limited"
    return "none"


def evidence_level(reg: dict, root_key: str, _stack: tuple = (), icache: dict | None = None) -> str:
    """Weakest live evidence level over a root and its declared dependency closure."""
    if root_key in _stack:
        return "none"
    entry = reg["revisions"].get(root_key)
    if entry is None:
        return "none"
    integ = icache[root_key] if icache is not None and root_key in icache else integrity(root_key, entry)
    if icache is not None:
        icache[root_key] = integ
    level = own_evidence_level(integ)
    for dep in entry.get("depends_on_revisions", []):
        d = evidence_level(reg, dep, _stack + (root_key,), icache)
        if EVIDENCE_LEVELS.index(d) < EVIDENCE_LEVELS.index(level):
            level = d
    return level


# ---------------------------------------------------------------- supersession and readiness
def superseders(reg: dict, root_key: str) -> list[tuple[str, dict]]:
    out = []
    for other, e in reg["revisions"].items():
        for rel in e.get("relations", []):
            if rel["other"] == root_key and rel["type"].startswith("SUPERSEDES"):
                out.append((other, rel))
    return out


def root_readiness(reg: dict, root_key: str, run_tests: bool = False, _stack: tuple = (), icache: dict | None = None) -> dict:
    if root_key in _stack:
        return {"root": root_key, "state": "NOT_READY_DEPENDENCY_CYCLE", "reasons": ["cycle"]}
    entry = reg["revisions"].get(root_key)
    if entry is None:
        return {"root": root_key, "state": "NOT_READY_UNKNOWN_MAPPING", "reasons": ["root not in registry"]}
    integ = icache[root_key] if icache is not None and root_key in icache else integrity(root_key, entry)
    if icache is not None:
        icache[root_key] = integ
    base = {"integrity": integ["integrity"], "integrity_basis": integ.get("integrity_basis"),
            "provenance_basis": "runtime_manifest" if integ.get("historical_runtime_manifest_available") else "none (no historical run-time manifest)",
            "evidence_level_own": own_evidence_level(integ), "scientific_status": entry.get("status")}
    reasons = []
    if entry["kind"] == "REJECTED_LEGACY":
        return {"root": root_key, "state": "NOT_READY_REJECTED_LEGACY", **base,
                "history_readable": integ["integrity"] in OK_STATES, "reasons": [f"rejected by {entry.get('authority')}"]}
    if entry["kind"] != "CURRENT_REVISION":
        return {"root": root_key, "state": "NOT_READY_HISTORY_ONLY", **base, "history_readable": integ["integrity"] in OK_STATES,
                "reasons": [f"kind {entry['kind']} is history, not a current revision"]}
    if integ["integrity"] not in OK_STATES:
        reasons.append(f"integrity {integ['integrity']}: {integ['problems'][:3]}")
    ev = entry.get("evidence", {})
    if not ev.get("decision") or not resolve_root(ev["decision"]).exists():
        reasons.append("missing decision/evidence document")
    sup = [(o, r) for o, r in superseders(reg, root_key) if r["type"] == "SUPERSEDES_SCIENTIFIC_USE"]
    if len(sup) > 1:
        return {"root": root_key, "state": "NOT_READY_AMBIGUOUS_SUPERSESSION", **base, "reasons": [f"superseded for scientific use by {[o for o, _ in sup]}"]}
    if len(sup) == 1:
        reasons.append(f"superseded for scientific use by {sup[0][0]}")
    tests_txt = resolve_root(root_key) / "common_record_r1/tests.txt"
    if tests_txt.exists() and not tests_txt.read_text().startswith(("exit_code=0", "no packet-specific")):
        legacy = tests_txt.read_text()
        rec = entry.get("test_record_reconciliation")
        # Packet 18 Part D: a legacy record showing only passing pytest output (no exit_code line) may be reconciled by a
        # registered, hash-pinned re-record that starts with exit_code=0. A legacy record containing failures is never reconciled.
        legacy_passing_format = " passed" in legacy and "failed" not in legacy and "error" not in legacy.lower()
        if legacy_passing_format and rec and test_record_ok(rec):
            base["test_record_basis"] = f"legacy format reconciled by {rec['path']} (re-recorded {rec.get('recorded_at', '?')})"
        else:
            reasons.append("recorded packet tests did not exit 0")
    if run_tests and entry.get("tests"):
        res = subprocess.run([sys.executable, "-m", "pytest", "-q", *entry["tests"]], cwd=REPO, capture_output=True, text=True)
        if res.returncode != 0:
            return {"root": root_key, "state": "NOT_READY_CHILD_TESTS_FAILED", **base, "reasons": [res.stdout[-400:]]}
    deps = []
    for dep in entry.get("depends_on_revisions", []):
        d = root_readiness(reg, dep, run_tests=False, _stack=_stack + (root_key,), icache=icache)
        deps.append(d)
        if d["state"] != READY:
            reasons.append(f"dependency {dep}: {d['state']}")
    state = READY if not reasons else ("NOT_READY_SUPERSEDED" if any("superseded" in r for r in reasons) and len(reasons) == 1
                                         else "NOT_READY_DEPENDENCY" if all(r.startswith("dependency") for r in reasons)
                                         else "NOT_READY_EVIDENCE_OR_INTEGRITY")
    return {"root": root_key, "state": state, **base, "reasons": reasons, "dependencies": [{"root": d["root"], "state": d["state"]} for d in deps]}


def card_chain(plan: dict, card: str) -> list[str]:
    seen, stack = [], list(plan[card]["depends_on"])
    while stack:
        c = stack.pop()
        if c not in seen:
            seen.append(c)
            stack.extend(plan[c]["depends_on"])
    return sorted(seen)


def card_readiness(reg: dict, plan: dict, card: str, cache: dict | None = None, icache: dict | None = None) -> dict:
    cache = {} if cache is None else cache
    if card not in plan or card not in reg["cards"]:
        return {"card": card, "state": "NOT_READY_UNKNOWN_MAPPING", "reasons": ["card not in plan/registry"]}
    info = reg["cards"][card]

    def ready(r):
        if r not in cache:
            cache[r] = root_readiness(reg, r, icache=icache)
        return cache[r]
    evidence = info.get("current_evidence", [])
    if not evidence:
        own = {"state": "NOT_READY_NO_CURRENT_EVIDENCE", "reasons": ["only the rejected legacy root exists"]}
    else:
        bad = [f"{r}: {ready(r)['state']}" for r in evidence if ready(r)["state"] != READY]
        own = {"state": READY if not bad else "NOT_READY_EVIDENCE_OR_INTEGRITY", "reasons": bad}
    dep_states = {}
    for d in card_chain(plan, card):
        de = reg["cards"].get(d, {}).get("current_evidence", [])
        dep_states[d] = "NOT_READY_NO_CURRENT_EVIDENCE" if not de else (READY if all(ready(r)["state"] == READY for r in de) else "NOT_READY_EVIDENCE_OR_INTEGRITY")
    blocked = sorted(d for d, s in dep_states.items() if s != READY)
    state = own["state"] if own["state"] != READY else (READY if not blocked else "NOT_READY_DEPENDENCY")
    return {"card": card, "state": state, "own": own, "transitive_dependencies": len(dep_states), "blocked_dependencies": blocked,
            "legacy_root": info.get("legacy_root"), "note": "readiness is bookkeeping for dependent use, not scientific acceptance"}


# ---------------------------------------------------------------- preflight and replays
def owning_root(reg: dict, path: Path) -> str | None:
    best = None
    for key in reg["revisions"]:
        r = resolve_root(key)
        if path == r or r in path.parents:
            if best is None or len(str(resolve_root(best))) < len(str(r)):
                best = key
    return best


def _pin_ok(item: dict) -> tuple[bool, str]:
    p = resolve_root(item["path"])
    if not p.exists():
        return False, f"MISSING input {item['role']}: {item['path']}"
    got = tree_sha(p) if item.get("kind") == "tree" else sha(p)
    if got != item["sha256"]:
        return False, f"CHANGED input {item['role']}: {item['path']}"
    return True, ""


def preflight(reg: dict, adapter_id: str, authorization: Path | None, cap_seconds: float | None, output: Path) -> dict:
    """All checks that must pass before any child starts. Raises StageRefusal listing every failure."""
    ad = reg.get("adapters", {}).get(adapter_id)
    if ad is None:
        raise StageRefusal(f"unknown adapter {adapter_id}")
    fails = []
    if output.exists():
        fails.append("output directory must be fresh")
    if ad["mode"] == "execute" or authorization is not None:
        if authorization is None or not authorization.exists():
            fails.append("execute mode requires the registered authorization document")
        elif sha(authorization) != ad.get("authorization_sha256"):
            fails.append("authorization document hash does not match the registered authorization (stale or wrong)")
    if ad["mode"] == "execute" and (cap_seconds is None or cap_seconds < ad["compute_cap_seconds"]):
        fails.append("unapproved compute: requested cap below the adapter's declared compute")
    root_key = ad["root"]
    if root_key not in reg["revisions"]:
        fails.append(f"adapter root {root_key} is not registered")
    elif not resolve_root(root_key).is_dir():
        fails.append(f"adapter root {root_key} is missing")
    if root_key in reg["revisions"] and len([o for o, r in superseders(reg, root_key) if r["type"] == "SUPERSEDES_SCIENTIFIC_USE"]) > 1:
        fails.append(f"adapter root {root_key} has ambiguous supersession")
    if not ad.get("inputs"):
        fails.append("adapter declares no inputs; nothing can be pinned")
    history = set(ad.get("history_roots", []))
    used_roots = set()
    for item in ad.get("inputs", []):
        ok, msg = _pin_ok(item)
        if not ok:
            fails.append(msg)
        p = resolve_root(item["path"])
        if item.get("external"):
            continue
        owner = owning_root(reg, p)
        if owner is None:
            fails.append(f"input {item['role']} is not inside any registered root (unknown mapping)")
            continue
        used_roots.add(owner)
    for item in ad.get("code", []):
        ok, msg = _pin_ok({**item, "role": "code"})
        if not ok:
            fails.append(msg.replace("input code", "code"))
    mode = ad.get("dependency_mode")
    readiness = {}
    if mode == "scientific":
        for r in sorted(set(ad.get("requires_ready", [])) | used_roots):
            st = root_readiness(reg, r)["state"]
            readiness[r] = st
            if st != READY:
                fails.append(f"scientific dependency {r} is {st}")
    elif mode == "diagnostic":
        for r in sorted(used_roots):
            st = root_readiness(reg, r)["state"]
            readiness[r] = st
            if st != READY and r not in history:
                fails.append(f"diagnostic input root {r} is {st} and not listed in history_roots")
    else:
        fails.append("adapter declares no dependency_mode (scientific|diagnostic)")
    permitted = list(ad.get("evidence_levels_permitted", DEFAULT_PERMITTED_EVIDENCE))
    if any(lv not in EVIDENCE_LEVELS or lv == "none" for lv in permitted):
        fails.append(f"contract declares an invalid evidence level in {permitted}")
    levels = {}
    level_roots = sorted(set(ad.get("requires_ready", [])) | used_roots) if mode == "scientific" else sorted(r for r in used_roots if r not in history)
    for r in level_roots:
        lv = evidence_level(reg, r)
        levels[r] = lv
        if lv not in permitted:
            fails.append(f"contract does not permit evidence level {lv} for {r} (permitted: {permitted})")
    if fails:
        raise StageRefusal("; ".join(fails))
    return {"adapter": adapter_id, "dependency_mode": mode, "input_roots": sorted(used_roots), "readiness_at_preflight": readiness,
            "evidence_levels": levels, "evidence_levels_permitted": permitted}


def authorize(reg: dict, adapter_id: str, authorization: Path | None, cap_seconds: float | None) -> dict:
    """Authorization and compute checks only (kept for callers that do not launch anything)."""
    ad = reg.get("adapters", {}).get(adapter_id)
    if ad is None:
        raise StageRefusal(f"unknown adapter {adapter_id}")
    if ad["mode"] == "execute":
        if authorization is None or not authorization.exists():
            raise StageRefusal("execute mode requires an authorization document")
        if sha(authorization) != ad["authorization_sha256"]:
            raise StageRefusal("authorization document hash does not match the registered authorization")
        if cap_seconds is None or cap_seconds < ad["compute_cap_seconds"]:
            raise StageRefusal("unapproved compute: requested cap below the adapter's declared compute")
    return ad


def run_adapter(reg: dict, adapter_id: str, output: Path, authorization: Path | None = None, cap_seconds: float | None = None) -> dict:
    pre = preflight(reg, adapter_id, authorization, cap_seconds, output)      # raises before any child starts
    ad = reg["adapters"][adapter_id]
    target = resolve_root(ad["root"])
    before = {"root_tree": tree_sha(target), "inputs": {i["role"]: i["sha256"] for i in ad["inputs"]}}
    output.mkdir(parents=True)
    contract = {"adapter": adapter_id, "inputs": {i["role"]: str(resolve_root(i["path"])) for i in ad["inputs"]},
                "code": [str(resolve_root(c["path"])) for c in ad.get("code", [])], "dependency_mode": ad["dependency_mode"]}
    (output / "adapter_contract.json").write_text(json.dumps(contract, indent=1) + "\n")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
           "PYTHONPATH": f"{REPO}:{REPO / 'scripts'}"}
    t0 = time.time()
    try:
        done = subprocess.run([sys.executable, "-m", "grain_research.gptpro_foundation_v3.revision_adapters", adapter_id, str(output)],
                              cwd=REPO, env=env, capture_output=True, text=True, timeout=ad["compute_cap_seconds"])
        rc, err = done.returncode, (done.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        rc, err = -9, "timeout at declared compute cap"
    after_inputs = {}
    for i in ad["inputs"]:
        p = resolve_root(i["path"])
        after_inputs[i["role"]] = (tree_sha(p) if i.get("kind") == "tree" else sha(p)) if p.exists() else None
    after = {"root_tree": tree_sha(target), "inputs": after_inputs}
    result_path = output / "adapter_result.json"
    child = json.loads(result_path.read_text()) if result_path.exists() else {}
    out = {"adapter": adapter_id, "mode": ad["mode"], "dependency_mode": ad["dependency_mode"], "root": ad["root"], "preflight": pre,
           "child_exit": rc, "seconds": time.time() - t0, "source_artifacts_unchanged": before == after,
           "diagnostic_only": ad["dependency_mode"] == "diagnostic", "grants_readiness": False,
           "child_result": child, "stderr_tail": err if rc else ""}
    out["status"] = "PASS" if (rc == 0 and before == after and child.get("status") == "PASS") else "FAIL"
    (output / "stage_result.json").write_text(json.dumps(out, indent=1) + "\n")
    return out
