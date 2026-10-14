#!/usr/bin/env python3
"""Archive inspectable upstream R sources and provenance, without installing packages."""

import argparse
import hashlib
import json
from pathlib import Path

import requests


def fetch(output):
    """Resolve one upstream commit and bind downloaded public-source bytes to it."""
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "upstream_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, info in manifest["files"].items():
            if (
                hashlib.sha256((output / name).read_bytes()).hexdigest()
                != info["sha256"]
            ):
                raise ValueError("Changed upstream source archive")
        print("Verified existing upstream archive")
        return
    if any(output.iterdir()):
        raise ValueError("Incomplete upstream archive: use a new output")
    repo = "lockEF/r.jive"
    response = requests.get(
        f"https://api.github.com/repos/{repo}/commits/master", timeout=30
    )
    response.raise_for_status()
    commit = response.json()["sha"]
    files = {}
    for path in ("R/sJIVE_functions.r", "DESCRIPTION"):
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        name = Path(path).name
        (output / name).write_bytes(r.content)
        files[name] = {"url": url, "sha256": hashlib.sha256(r.content).hexdigest()}
    manifest = {
        "repo": repo,
        "commit": commit,
        "files": files,
        "modifications": "none",
        "execution": "not_executed_by_fetcher",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print({"repo": repo, "commit": commit, "files": list(files)})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    fetch(p.parse_args().output)
