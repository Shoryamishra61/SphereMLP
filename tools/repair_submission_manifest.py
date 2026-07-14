"""Normalize Whest package manifest paths for portable tarball submission.

WhestBench 0.12.0rc3 on Windows writes ``manifest.json`` file names with
backslashes while the generated tar members use POSIX forward slashes.  The
remote grader compares those names literally.  This utility rewrites only the
manifest file names, preserves all payload bytes, and verifies the resulting
archive before it is used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def repair(source: Path, output: Path) -> dict[str, object]:
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        manifest_member = next(
            (member for member in members if member.name == "manifest.json"), None
        )
        if manifest_member is None:
            raise ValueError("source archive has no manifest.json")
        raw_manifest = archive.extractfile(manifest_member)
        if raw_manifest is None:
            raise ValueError("cannot read manifest.json")
        manifest = json.loads(raw_manifest.read().decode("utf-8"))
        for record in manifest["files"]:
            record["name"] = record["name"].replace("\\", "/")
        manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

        with tarfile.open(output, "w:gz") as repaired:
            for member in members:
                if member.name == "manifest.json":
                    replacement = tarfile.TarInfo("manifest.json")
                    replacement.size = len(manifest_bytes)
                    replacement.mode = member.mode
                    replacement.mtime = member.mtime
                    repaired.addfile(replacement, io.BytesIO(manifest_bytes))
                else:
                    payload = archive.extractfile(member) if member.isfile() else None
                    repaired.addfile(member, payload)
    return verify(output)


def verify(path: Path) -> dict[str, object]:
    with tarfile.open(path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        raw_manifest = archive.extractfile(members["manifest.json"])
        if raw_manifest is None:
            raise ValueError("cannot read repaired manifest")
        manifest = json.loads(raw_manifest.read().decode("utf-8"))
        for record in manifest["files"]:
            name = record["name"]
            if "\\" in name or name not in members:
                raise ValueError(f"manifest member mismatch: {name!r}")
            payload = archive.extractfile(members[name])
            if payload is None:
                raise ValueError(f"cannot read member: {name}")
            actual = hashlib.sha256(payload.read()).hexdigest()
            if actual != record["sha256"]:
                raise ValueError(f"hash mismatch: {name}")
    return {"archive": str(path), "file_count": len(manifest["files"]), "verified": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(repair(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
