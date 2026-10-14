#!/usr/bin/env python3
"""Finite licensed GUM interface check; no psychological labels or neural fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

from run_grain_math_response import ROOT, sha, write_csv, write_json

from grain_research.construction import math_observation_dag as D
from grain_research.construction import math_purpose_accounts as P
from grain_research.source_event_graph import stanza_words

REV = "d650dac3b46abb373c98f23c8568e1044024b992"
URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/"
    + REV
    + "/"
)
CACHE = ROOT / "results/grain_vector_mechanisms/resources/ud_gum_r217"
DEFAULT = ROOT / "results/grain_math_response_v1/observation_dag_natural_r1"


def source_cache():
    """Cache a pinned public release with its actual license and attribution."""
    names = ("README.md", "LICENSE.txt", "en_gum-ud-test.conllu")
    inventory = CACHE / "inventory.json"
    if inventory.exists():
        data = json.loads(inventory.read_text())
        for name, digest in data["files"].items():
            if sha(CACHE / name) != digest:
                raise ValueError("changed licensed source " + name)
        return data
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in names:
        urllib.request.urlretrieve(URL + name, CACHE / name)
    readme = (CACHE / "README.md").read_text()
    if "License: CC BY-NC-SA 4.0" not in readme or "Includes text: yes" not in readme:
        raise ValueError("source license/inclusion no longer agrees with contract")
    data = {
        "revision": REV,
        "repository": "https://github.com/UniversalDependencies/UD_English-GUM",
        "license": "CC BY-NC-SA 4.0",
        "use": "local noncommercial research",
        "files": {n: sha(CACHE / n) for n in names},
        "attribution": "GUM / Amir Zeldes and Siyao Peng; see retained README and LICENSE",
        "no_private_data_or_psychological_labels": True,
    }
    write_json(inventory, data)
    return data


def sentence_bank(content, limit=64):
    """Select source-order non-Reddit sentences; gold syntax selects shape, not score.

    Only original text and token offsets are supplied to the observed parser.
    Human UD annotations are a separate structural comparator. They are not
    gold event identity, modality or psychological annotations.
    """
    selected, metadata = [], {}
    for block in content.strip().split("\n\n"):
        comments = [x[2:] for x in block.splitlines() if x.startswith("# ")]
        for c in comments:
            if c.startswith("newdoc id = "):
                metadata = {"document": c.split(" = ", 1)[1]}
            elif c.startswith("meta::") and " = " in c:
                k, v = c.split(" = ", 1)
                metadata[k] = v
        text = next((c[7:] for c in comments if c.startswith("text = ")), None)
        sent_id = next((c[10:] for c in comments if c.startswith("sent_id = ")), None)
        fields = [
            x.split("\t") for x in block.splitlines() if x and not x.startswith("#")
        ]
        tokens = [x for x in fields if x[0].isdigit()]
        if (
            not text
            or metadata.get("meta::genre") == "reddit"
            or not 8 <= len(tokens) <= 50
        ):
            continue
        if not any(x[3] == "VERB" for x in tokens) or not any(
            x[7] == "obj" for x in tokens
        ):
            continue
        positions, cursor, covered = {}, 0, set()
        for x in fields:
            if "." in x[0]:
                continue
            if "-" in x[0]:
                a, b = map(int, x[0].split("-"))
                ids = list(range(a, b + 1))
            else:
                i = int(x[0])
                if i in covered:
                    continue
                ids = [i]
            start = text.find(x[1], cursor)
            if start < 0:
                raise ValueError(f"GUM raw token alignment failed: {sent_id}")
            end = start + len(x[1])
            for i in ids:
                positions[i] = (start, end, len(ids) > 1)
                covered.add(i)
            cursor = end
        words = []
        for x in tokens:
            i = int(x[0])
            start, end, expanded = positions[i]
            words.append(
                {
                    "id": i,
                    "text": x[1],
                    "lemma": x[2].lower(),
                    "upos": x[3],
                    "feats": x[5],
                    "head": int(x[6]),
                    "deprel": x[7],
                    "start": start,
                    "end": end,
                    "surface": text[start:end],
                    "sentence_index": 0,
                    "expanded_multiword": expanded,
                }
            )
        selected.append(
            {
                "material_id": sent_id,
                "text": text,
                "metadata": dict(metadata),
                "gold_words": [words],
                "author_id_verified": False,
                "selection": "first64 non-Reddit 8..50 tokens with VERB and obj in UD annotations; no score selection",
            }
        )
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError("finite selection incomplete")
    return selected


def main():
    """Compare new observed syntax to separate UD structure, preserving failures."""
    from fetch_grain_syntax_reference import MODELS as SYNTAX
    from fetch_grain_syntax_reference import PACKAGES, PROCESSORS
    from run_grain_source_event_graph import verify_models

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT)
    args = ap.parse_args()
    out = args.out
    inventory = source_cache()
    bank = sentence_bank((CACHE / "en_gum-ud-test.conllu").read_text())
    signature = {
        str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        "dag": sha(ROOT / "grain_research/construction/math_observation_dag.py"),
        "licensed_source": sha(CACHE / "inventory.json"),
    }
    protocol = out / "00/protocol_before_run.json"
    if protocol.exists() and json.loads(protocol.read_text())["signature"] != signature:
        raise ValueError("natural interface protocol changed")
    write_json(
        protocol,
        {
            "signature": signature,
            "source": inventory,
            "materials": len(bank),
            "psychological_annotations": False,
            "new_encoder_calls": 0,
            "new_fits": 0,
            "purpose": "natural typed observation interface, not author measurement acceptance",
        },
    )
    write_json(out / "00/materials.json", bank)
    verify_models()
    sys.path.insert(0, str(PACKAGES))
    import stanza
    import torch

    torch.set_num_threads(2)
    parser = stanza.Pipeline(
        "en",
        dir=str(SYNTAX),
        package=None,
        processors=PROCESSORS,
        use_gpu=False,
        download_method=None,
        verbose=False,
    )
    rows, nodes, calls, start = [], [], 0, time.monotonic()
    for i, row in enumerate(bank):
        path = out / f"01/parses/{i:04d}.json"
        digest = hashlib.sha256(row["text"].encode()).hexdigest()
        if path.exists():
            data = json.loads(path.read_text())
            if data["text_sha256"] != digest:
                raise ValueError("natural parse cache changed")
        else:
            words = stanza_words(parser(row["text"]), row["text"])
            graph = D.observe_dag(words, row["text"])
            gold = D.observe_dag(row["gold_words"], row["text"])
            data = {
                "text_sha256": digest,
                "words": words,
                "graph": graph,
                "gold_graph": gold,
                "flat": P.observe_modes(words, row["text"]),
            }
            write_json(path, data)
            calls += 1
        observed, gold = data["graph"]["events"], data["gold_graph"]["events"]
        by_span = {tuple(e["action_span"]): e for e in gold}
        matched = [e for e in observed if tuple(e["action_span"]) in by_span]
        rows.append(
            {
                "material_id": row["material_id"],
                "genre": row["metadata"].get("meta::genre"),
                "flat_heads": len(data["flat"]),
                "dag_heads": len(observed),
                "gold_ud_dag_heads": len(gold),
                "matched_heads": len(matched),
                "same_actor_conceiver": sum(
                    (e["actor"], e["conceiver"])
                    == (
                        by_span[tuple(e["action_span"])]["actor"],
                        by_span[tuple(e["action_span"])]["conceiver"],
                    )
                    for e in matched
                ),
                "context_nodes": sum(
                    n["type"].endswith("context") for n in data["graph"]["nodes"]
                ),
                "unresolved_main": sum(e["status"] != "ok" for e in observed),
                "author_score": None,
                "condition": None,
            }
        )
        nodes.extend(data["graph"]["nodes"])
    write_csv(out / "02/interface_comparison.csv", rows)
    summary = {
        "materials": len(bank),
        "documents": len({r["metadata"]["document"] for r in bank}),
        "genres": dict(Counter(r["genre"] for r in rows)),
        "observed_nodes": len(nodes),
        "node_types": dict(Counter(n["type"] for n in nodes)),
        "flat_heads": sum(r["flat_heads"] for r in rows),
        "dag_heads": sum(r["dag_heads"] for r in rows),
        "gold_ud_dag_heads": sum(r["gold_ud_dag_heads"] for r in rows),
        "matched_heads": sum(r["matched_heads"] for r in rows),
        "same_actor_conceiver": sum(r["same_actor_conceiver"] for r in rows),
        "context_nodes": sum(r["context_nodes"] for r in rows),
        "unresolved_main": sum(r["unresolved_main"] for r in rows),
        "new_stanza_calls": calls,
        "elapsed_seconds": time.monotonic() - start,
        "actual_author_scores": 0,
        "new_encoder_calls": 0,
        "private_reads_or_api_calls_or_training": 0,
        "scope": "UD structure comparator, not independent semantic gold or human personality verification",
    }
    write_json(out / "03/summary.json", summary)
    write_json(
        out / "04/artifact_hashes.json",
        {
            str(p.relative_to(out)): sha(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
            and not p.name.startswith("._")
            and p != out / "04/artifact_hashes.json"
        },
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
