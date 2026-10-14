#!/usr/bin/env python3
"""Index historical plan obligations without treating prose as completion evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PLAN_NAME = re.compile(
    r"PLAN|PROGRAM|PROTOCOL|GOAL|BACKLOG|TODO|PENDING|GAP|ROADMAP|TASK|计划|計画",
    re.IGNORECASE,
)
PLAN_TITLE = re.compile(
    r"计划|計画|研究路线|research (?:plan|program)|execution plan", re.IGNORECASE
)
PENDING = re.compile(
    r"pending|尚未|待实现|未实现|未完成|下一步|next step|proposed", re.IGNORECASE
)
LINK = re.compile(r"\]\(([^)]+)\)")


def markdown_files(root):
    """Read documentation only, excluding AppleDouble files and data/output trees."""
    files = set(root.glob("*.md"))
    for directory in ("docs", "research_notes", "_archive/docs"):
        base = root / directory
        if base.exists():
            files.update(base.rglob("*.md"))
    return sorted(
        p
        for p in files
        if not any(s.startswith(".") for s in p.relative_to(root).parts)
    )


def candidates(text):
    """Capture headings, table rows and proposed actions; none are auto-accepted."""
    in_code = False
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped:
            continue
        kind = None
        if re.match(r"#{2,4} ", stripped):
            kind = "section"
        elif stripped.startswith("|") and not re.match(r"\|[ :|\-]+$", stripped):
            kind = "table_row"
        elif re.match(r"[-*] \[[ xX]\]", stripped):
            kind = "checkbox_claim"
        elif PENDING.search(stripped):
            kind = "prospective_or_status_claim"
        if kind:
            yield line_number, kind, stripped


def local_links(path, text, root):
    """Resolve explicit local references, not infer results from filename similarity."""
    rows = []
    for value in LINK.findall(text):
        value = unquote(value.split("#", 1)[0].split(' "', 1)[0])
        if not value or re.match(r"[a-z]+://", value, re.IGNORECASE):
            continue
        target = (path.parent / value).resolve()
        try:
            relative = str(target.relative_to(root.resolve()))
        except ValueError:
            relative = str(target)
        rows.append({"target": relative, "exists": target.is_file()})
    return rows


def inventory(roots):
    """Retain copies and hashes so deduplication cannot discard changed plans."""
    documents, entries = [], []
    for label, root in roots.items():
        if not root.is_dir():
            raise ValueError("Missing requested repository: " + str(root))
        for path in markdown_files(root):
            content = path.read_text(encoding="utf-8")
            title = next(
                (s.lstrip("# ") for s in content.splitlines() if s.startswith("# ")), ""
            )
            if not PLAN_NAME.search(path.stem) and not PLAN_TITLE.search(title):
                continue
            relative = str(path.relative_to(root))
            if path.name.startswith("GRAIN_ALL_PLAN_RESEARCH_LOOP"):
                continue  # This coordinator is not its own research obligation.
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            document_id = f"{label}:{relative}"
            document = {
                "document_id": document_id,
                "repo": label,
                "path": relative,
                "sha256": digest,
                "title": title,
                "lines": len(content.splitlines()),
                "state": "UNREVIEWED",
                "local_links": local_links(path, content, root),
            }
            documents.append(document)
            for number, kind, text in candidates(content):
                entries.append(
                    {
                        "entry_id": f"{document_id}:L{number}",
                        "document_id": document_id,
                        "source_sha256": digest,
                        "line": number,
                        "kind": kind,
                        "text": text,
                        "review_state": "UNREVIEWED",
                        "evidence": "",
                    }
                )
    return documents, entries


def write_csv(path, rows):
    """Write an inspectable catalogue, including a header for an empty result."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]) if rows else ["entry_id"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Create a fresh snapshot; this command never executes referenced scripts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persona-root", type=Path, default=ROOT.parent / "project persona"
    )
    parser.add_argument("--grain-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    documents, entries = inventory(
        {"grain": args.grain_root, "persona": args.persona_root}
    )
    args.output.mkdir(parents=True)
    (args.output / "documents.json").write_text(
        json.dumps(documents, ensure_ascii=False, indent=2) + "\n"
    )
    write_csv(args.output / "candidate_obligations.csv", entries)
    hashes = Counter(d["sha256"] for d in documents)
    summary = {
        "documents": len(documents),
        "documents_by_repo": dict(Counter(d["repo"] for d in documents)),
        "unique_contents": len(hashes),
        "candidate_entries": len(entries),
        "reviewed_entries": 0,
        "experiment_count": None,
        "scope": "root/docs/research_notes/archive-docs markdown; name/title candidate selection",
        "not_scanned": "private datasets, generated outputs, attachments outside repos, non-markdown plans",
        "interpretation": "Candidate index, not a count of experiments or verified completion. Historical status claims are not authoritative.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
