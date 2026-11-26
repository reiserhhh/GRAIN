#!/usr/bin/env python3
"""Exercise the panel on existing authorized forum counts, essays, and choice logs."""

import argparse
import csv
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from grain_core.evidence_panel import (
    VERSION,
    Coverage,
    EvidenceEvent,
    Opportunity,
    run_panel,
    run_queries,
)
from grain_core.evidence_stream import read_panel, text_csv, write_stream

CHANNELS = {"forum": ["words", "units"], "essay": ["characters"], "button": ["left"]}


def key(domain, identity):
    return domain + ":" + hashlib.blake2b(identity.encode(), digest_size=12).hexdigest()


def prepare(args, directory):
    bank = json.loads(args.forum_counts.read_text())
    events, opportunity = [], []
    counts = {}
    for row in bank["rows"]:
        actor = key("forum", row["author"])
        for w, communities in enumerate(row["communities"]):
            for ctx, value in communities.items():
                z = value["text"]
                events.append(
                    EvidenceEvent(
                        f"f{len(events)}",
                        f"forum:{w}",
                        actor,
                        ctx,
                        "forum",
                        [z["words"], z["units"]],
                    )
                )
                if w == 0:
                    counts[ctx] = counts.get(ctx, 0) + 1
    target = max(counts, key=lambda ctx: (counts[ctx], ctx))
    with args.essays.open(encoding="utf-8", errors="replace", newline="") as source:
        for row in text_csv(
            source,
            columns={"id": "#AUTHID", "actor": "#AUTHID", "text": "TEXT"},
            constants={"window": "essays", "context": "writing"},
        ):
            actor = key("essay", row.actor)
            events.append(
                EvidenceEvent(
                    actor,
                    row.window,
                    actor,
                    row.context,
                    "essay",
                    [len(row.text)] if row.text.strip() else None,
                )
            )
    with args.choices.open(encoding="utf-8", newline="") as source:
        for r in csv.DictReader(source):
            actor = key("choice", r["subject"])
            w = "choice:" + r["block_num"]
            button = int(r["subj_choice"])
            menu = [float(r[x]) for x in ("mine_prob_win_left", "mine_prob_win_right")]
            if (
                button not in (1, 2)
                or sorted(menu) != [0.3, 0.7]
                or menu[button - 1] != float(r["chosen_mine_prob"])
            ):
                raise ValueError("recorded choice and menu disagree")
            id_ = f"{actor}:{w}:{r['trial_num']}"
            events.append(
                EvidenceEvent(id_, w, actor, "menu", "button", [int(button == 1)])
            )
            opportunity.append(
                Opportunity(
                    id_, w, actor, "menu", "choose-70", True, menu[button - 1] == 0.7
                )
            )
    blocks = sorted({r.window for r in opportunity})
    # Forum windows are disjoint archival samples, not chronological or complete.
    coverage = [
        Coverage("essays", True, "writing"),
        *(Coverage(w, True, "menu") for w in blocks),
    ]
    queries = [
        {
            "id": "forum-overlap",
            "op": "participation_flow",
            "before": "forum:0",
            "after": "forum:1",
            "target": target,
        },
        {
            "id": "action-pairs",
            "op": "paired_opportunities",
            "before": blocks[0],
            "after": blocks[1],
            "origin": "menu",
            "destination": "menu",
            "measure": "choose-70",
            "channel": "button",
        },
    ]
    records = directory / "records.jsonl"
    with records.open("x", encoding="utf-8") as target:
        write_stream(
            target,
            CHANNELS,
            (r for group in (events, coverage, opportunity) for r in group),
        )
    (directory / "queries.json").write_text(json.dumps(queries))
    whole = {
        "schema_version": VERSION,
        "channels": CHANNELS,
        "events": list(map(asdict, events)),
        "opportunities": list(map(asdict, opportunity)),
        "coverage": list(map(asdict, coverage)),
        "queries": queries,
    }
    (directory / "whole.json").write_text(json.dumps(whole, separators=(",", ":")))
    return {
        "events": len(events),
        "opportunities": len(opportunity),
        "events_by_channel": {c: sum(r.channel == c for r in events) for c in CHANNELS},
        "whole_json_bytes": (directory / "whole.json").stat().st_size,
        "jsonl_bytes": records.stat().st_size,
    }


def worker(mode, directory):
    t0 = time.perf_counter()
    if mode == "stream":
        with (directory / "records.jsonl").open(encoding="utf-8") as source:
            panel = read_panel(source)
        result = run_queries(
            panel, json.loads((directory / "queries.json").read_text())
        )
    else:
        result = run_panel(json.loads((directory / "whole.json").read_text()))
    took = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if platform.system() == "Darwin" else peak * 1024
    check = json.dumps(result, sort_keys=True, allow_nan=False)
    return {
        "seconds": took,
        "peak_rss_bytes": peak_bytes,
        "result_digest": hashlib.sha256(check.encode()).hexdigest(),
        "paired_eligible": result["results"]["action-pairs"]["paired_known_eligible"][
            "actors"
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forum-counts", type=Path)
    p.add_argument("--essays", type=Path)
    p.add_argument("--choices", type=Path)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--worker", choices=("stream", "whole"))
    p.add_argument("--directory", type=Path)
    args = p.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.directory), allow_nan=False))
        return
    if not all((args.forum_counts, args.essays, args.choices)) or args.repeat < 1:
        p.error("supply all three local input paths and a positive repeat count")
    with tempfile.TemporaryDirectory() as temp:
        data = prepare(args, Path(temp))
        modes = {m: [] for m in ("whole", "stream")}
        for i in range(args.repeat):
            for m in list(modes)[:: (-1 if i % 2 else 1)]:
                result = subprocess.check_output(
                    [
                        sys.executable,
                        "-m",
                        "examples.evidence_real_replay",
                        "--worker",
                        m,
                        "--directory",
                        temp,
                    ],
                    text=True,
                )
                modes[m].append(json.loads(result))
        digests = {r["result_digest"] for rs in modes.values() for r in rs}
        if len(digests) != 1:
            raise AssertionError("the two ingestion paths changed analysis results")
    report = {
        "platform": f"{platform.system()} {platform.machine()}",
        "data": data,
        "cold_results_equal": True,
        "modes": {
            m: {
                "median_seconds": statistics.median(x["seconds"] for x in rows),
                "median_peak_rss_bytes": statistics.median(
                    x["peak_rss_bytes"] for x in rows
                ),
                "samples": rows,
            }
            for m, rows in modes.items()
        },
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
