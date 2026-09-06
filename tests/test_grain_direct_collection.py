#!/usr/bin/env python3
"""Complete synthetic registries, native-readout replay and export boundary checks."""

import copy
import json
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_cache import (
    verify_author,
    verify_collection,
)
from grain_research.vector_mechanisms.direct_material import (
    archive_direct,
    load_registration,
    technical_cases,
    verify_prompts,
    verify_tokenizer,
)
from grain_research.vector_mechanisms.direct_traits import (
    AXES,
    BIG5,
    SYSTEM,
    collect_author,
    question,
    tasks_for_group,
)
from grain_research.vector_mechanisms.ready_comparison import digest
from tests.test_grain_direct_traits import Tokenizer


class Observer:
    tokenizer = Tokenizer()
    max_tokens = 3072

    def __init__(self):
        self.calls = 0

    def capture(self, ids):
        self.calls += 1
        return {}, np.sin(np.arange(128) + sum(ids) % 19)


def author_packet(group="mbti_only", uid="fixture"):
    """Generate only invented examples, never reading a local participant file."""
    arms = {
        a: [f"invented example {i}" for i in range(36)]
        for a in ("raw", "cue_mask", "placebo_mask")
    }
    tasks = tasks_for_group(group)
    rows, metrics = collect_author(arms, tasks, Observer())
    case = {
        "author_id": uid,
        "source_group": group,
        "tasks": list(tasks),
        "supported": True,
        "layouts": [
            {k: r[k] for k in ("arm", "view", "task", "prompt_tokens", "prompt_sha256")}
            | {"supported": True}
            for r in rows
        ],
    }
    data = {
        "author_id": uid,
        "source_group": group,
        "case_commitment": material_hash(case),
        "readouts": rows,
        "metrics": metrics,
        "view_means": {
            f"{a}__{t}": float(
                np.mean(
                    [
                        r["conditional_score"]
                        for r in rows
                        if r["arm"] == a and r["task"] == t
                    ]
                )
            )
            for a in arms
            for t in tasks
        },
    }
    return arms, case, data


def registry(tmp_path):
    """Sign a complete token support packet with the production material schema."""
    folder = tmp_path / "material"
    folder.mkdir()
    cases = [
        author_packet(g, str(i))[1]
        for i, g in enumerate(["big5_only"] * 3 + ["mbti_only"] * 5 + ["bridge"])
    ]
    source = [{k: c[k] for k in ("author_id", "source_group")} for c in cases]
    views = tmp_path / "views.json"
    views.write_text(json.dumps(source), encoding="utf-8")
    m = {
        "protocol": "VM05_DIRECT_TRAIT_MATERIAL_V1",
        "authors": len(source),
        "common_material_sha256": material_hash(source),
        "system": SYSTEM,
        "questions": {t: question(t) for t in (*BIG5, *AXES)},
        "max_tokens": 3072,
        "margin_tokens": 16,
        "neural_forwards": 0,
        "labels_read": False,
        "bridge_outcomes_read": False,
    }
    s = {
        "status": "COMPLETE",
        "authors": len(source),
        "logical_prompts": sum(len(c["layouts"]) for c in cases),
        "groups": dict(Counter(c["source_group"] for c in cases)),
        "supported_groups": dict(Counter(c["source_group"] for c in cases)),
        "unsupported_authors": 0,
        "neural_forwards": 0,
        "labels_read": False,
        "bridge_outcomes_read": False,
    }

    def seal():
        m["identity"] = material_hash({k: v for k, v in m.items() if k != "identity"})
        path = folder / "private_direct_cases.json"
        path.write_text(json.dumps(cases), encoding="utf-8")
        s.update(
            parent_identity=m["identity"], outputs_sha256={path.name: digest(path)}
        )
        (folder / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        (folder / "summary.json").write_text(json.dumps(s), encoding="utf-8")

    seal()
    return views, folder, cases, m, s, seal


def test_support_grid_and_role_cannot_be_changed_by_resigning(tmp_path):
    views, folder, cases, _m, _s, seal = registry(tmp_path)
    _, valid, _ = load_registration(views, folder)
    chosen = technical_cases(valid)
    assert len(chosen) == 6 and Counter(c["source_group"] for c in chosen) == {
        "big5_only": 2,
        "mbti_only": 4,
    }
    assert technical_cases(list(reversed(valid))) == chosen
    cases[0]["layouts"][0] = copy.deepcopy(cases[0]["layouts"][1])
    seal()
    with pytest.raises(ValueError, match="grid"):
        load_registration(views, folder)


def test_independent_arithmetic_and_exact_prompt_replay():
    arms, case, packet = author_packet()
    result = verify_author(packet, case, packet["metrics"]["alphabets"])
    assert result["rows"] == 36 and result["max_error"] < 1e-12
    assert len(verify_prompts(case, arms, Tokenizer())) == 36
    broken = copy.deepcopy(packet)
    broken["readouts"][0]["conditional_score"] += 0.1
    with pytest.raises(ValueError, match="reconstructed"):
        verify_author(broken, case, packet["metrics"]["alphabets"])
    broken = copy.deepcopy(case)
    broken["layouts"][0]["prompt_sha256"] = "wrong"
    with pytest.raises(ValueError, match="registered"):
        verify_prompts(broken, arms, Tokenizer())
    obs = Observer()
    with pytest.raises(TimeoutError):
        collect_author(arms, case["tasks"], obs, deadline=time.monotonic() - 1)
    assert obs.calls == 0


def test_direct_tokenizer_certificate_matches_actual_schema(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text("日本語", encoding="utf-8")
    material = {"tokenizer_sha256": {path.name: digest(path)}}
    verify_tokenizer(tmp_path, material)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_tokenizer(tmp_path, material)


def test_direct_archive_entrypoints_are_independently_importable(tmp_path):
    archive = tmp_path / "source.zip"
    archive_direct(Path(__file__).resolve().parents[1], archive)
    isolated = tmp_path / "isolated"
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == len(set(z.namelist()))
        z.extractall(isolated)
    for name in ("collect_grain_direct_traits.py", "prepare_grain_direct_traits.py"):
        result = subprocess.run(
            [sys.executable, "-I", str(isolated / "scripts" / name), "--help"],
            cwd=isolated,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_collector_full_synthetic_qualification_and_corrupt_cache(
    tmp_path, monkeypatch
):
    """Exercise run/qualification gates without weights or natural-person labels."""
    from scripts import collect_grain_direct_traits as entry

    fixtures = [
        author_packet(g, str(i))
        for i, g in enumerate(["big5_only"] * 2 + ["mbti_only"] * 4)
    ]
    whole = [c for _, c, _ in fixtures]
    by_author = {c["author_id"]: arms for arms, c, _ in fixtures}
    alphabets = (
        fixtures[0][2]["metrics"]["alphabets"] | fixtures[-1][2]["metrics"]["alphabets"]
    )
    material = {
        "identity": "synthetic",
        "qwen_revision": "synthetic-revision",
        "alphabets": alphabets,
    }

    class SyntheticObserver(Observer):
        def manifest(self):
            return {
                "revision": material["qwen_revision"],
                "dtype": "torch.bfloat16",
                "device": "cuda",
                "training": False,
                "attention": "eager",
                "pad_to_length": 3072,
            }

    monkeypatch.setattr(entry, "ROOT", tmp_path)
    for name in entry.CODE:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic fixture source\n", encoding="utf-8")
    monkeypatch.setattr(
        entry, "load_registration", lambda *args: (by_author, whole, material)
    )
    monkeypatch.setattr(entry, "verify_tokenizer", lambda *args: None)
    monkeypatch.setattr(
        entry.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer()
    )
    monkeypatch.setattr(entry, "load_observer", lambda *args: SyntheticObserver())
    monkeypatch.setattr(entry, "bundle_arms", lambda arms: arms)
    monkeypatch.setattr(
        entry,
        "archive_direct",
        lambda root, out: out.write_bytes(b"synthetic snapshot"),
    )
    args = SimpleNamespace(
        views=tmp_path / "unused",
        material_dir=tmp_path / "unused",
        tokenizer_dir=tmp_path,
        output_dir=tmp_path / "results/qual",
        qualification=True,
        preflight_only=False,
        shard_count=128,
        shard_index=0,
        max_seconds=60,
        max_cache_bytes=200_000_000,
    )
    entry.run(args)
    code = {name: digest(tmp_path / name) for name in entry.CODE}
    _, s = entry.qualified_parent(args.output_dir, material, code, whole)
    assert (
        s["authors"] == 6
        and s["logical_prompts"] == 234
        and s["all_qualification_rows_replayed"]
    )
    assert (
        verify_collection(args.output_dir, technical_cases(whole), material, code)[
            "max_error"
        ]
        < 1e-12
    )
    source = args.output_dir / "summary.json"
    changed = json.loads(source.read_text(encoding="utf-8"))
    changed["real_neural_collection"] = False
    source.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        entry.qualified_parent(args.output_dir, material, code, whole)
