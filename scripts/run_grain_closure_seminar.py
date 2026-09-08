#!/usr/bin/env python3
"""Run a read-only Claude seminar fork; save only formal result and safe metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/grain_vector_mechanisms"


def main():
    """Resume authorized discussion context without granting execution or write tools."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--fork", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "claude_framework_closure_20260908"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private seminar outputs must remain in ignored results")
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"round{args.round}_response.md"
    if dst.exists():
        raise ValueError("Do not overwrite a previous formal reply")
    settings = {
        "disableAllHooks": True,
        "permissions": {
            "deny": [
                "Read(./results/**)",
                "Read(./data/**)",
                "Read(./data_sets/**)",
                "Read(./.env*)",
                "Read(../project persona/.env*)",
                "Read(../project persona/data_sets/**)",
            ]
        },
    }
    cmd = [
        "/Users/huokean/.local/bin/claude",
        "--print",
        "--resume",
        args.resume,
        "--model",
        "claude-fable-5-1",
        "--effort",
        "max",
        "--output-format",
        "json",
        "--permission-mode",
        "plan",
        "--restricted",
        "--tools",
        "Read,Glob,Grep",
        "--allowedTools",
        "Read,Glob,Grep",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--disable-slash-commands",
        "--settings",
        json.dumps(settings),
        "--append-system-prompt",
        "Only read public project docs/reports/code for this seminar. Never write files, execute commands, read private data, secrets or results. Ignore old execution requests. Return a formal research response only.",
    ]
    if args.fork:
        cmd.append("--fork-session")
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            input=args.brief.read_text(),
            text=True,
            cwd=ROOT,
            capture_output=True,
            timeout=900,
            check=False,
        )
        elapsed = time.monotonic() - start
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = {"is_error": True, "subtype": "non_json_output"}
        safe = {
            k: payload.get(k)
            for k in (
                "session_id",
                "subtype",
                "is_error",
                "duration_ms",
                "num_turns",
                "modelUsage",
                "permission_denials",
            )
        }
        safe.update(
            elapsed_seconds=elapsed,
            returncode=result.returncode,
            requested_model="claude-fable-5-1",
            effort="max",
            tools=["Read", "Glob", "Grep"],
            permissions="plan + restricted, no MCP/hooks/skills",
            fork=args.fork,
        )
        # No stream/thinking blocks or raw diagnostics are persisted.
        (out / f"round{args.round}_metadata.json").write_text(
            json.dumps(safe, indent=2) + "\n"
        )
        if (
            result.returncode
            or payload.get("is_error")
            or not isinstance(payload.get("result"), str)
        ):
            print(
                {
                    "status": "CLAUDE_FAILED",
                    "subtype": payload.get("subtype"),
                    "returncode": result.returncode,
                }
            )
            if isinstance(payload.get("result"), str):
                dst.write_text(payload["result"] + "\n")
            raise SystemExit(1)
        dst.write_text(payload["result"] + "\n")
        print(
            {
                "status": "COMPLETE",
                "session_id": payload.get("session_id"),
                "elapsed_seconds": elapsed,
                "round": args.round,
            },
            flush=True,
        )
    except subprocess.TimeoutExpired:
        (out / f"round{args.round}_metadata.json").write_text(
            json.dumps({"status": "TIMEOUT", "seconds": time.monotonic() - start})
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
