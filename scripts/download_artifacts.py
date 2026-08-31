#!/usr/bin/env python3
"""Download, hash-check, and optionally extract ScienceArena release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release_assets.json"


def headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    result = {"Accept": accept, "User-Agent": "science-arena-reproducer"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        result["Authorization"] = f"Bearer {token}"
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_assets(repo: str, tag: str) -> dict[str, dict]:
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers())) as response:
        release = json.load(response)
    return {asset["name"]: asset for asset in release["assets"]}


def download(asset: dict, remote: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        remote["url"], headers=headers("application/octet-stream")
    )
    digest = hashlib.sha256()
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != asset["sha256"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {asset['filename']}")
    partial.replace(destination)


def safe_target(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if target != root.resolve() and root.resolve() not in target.parents:
        raise RuntimeError(f"unsafe archive path: {name}")
    return target


def extract(path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                safe_target(output, member.filename)
            archive.extractall(output)
        return
    if path.name.endswith(".tar.zst"):
        import zstandard

        with path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    safe_target(output, member.name)
                    if not (member.isfile() or member.isdir()):
                        raise RuntimeError(
                            f"unsupported non-regular archive member: {member.name}"
                        )
                    archive.extract(member, output)
        return
    if path.name.endswith(".json.zst"):
        import zstandard

        target = output / path.name[: -len(".zst")]
        with path.open("rb") as source, target.open("wb") as destination:
            zstandard.ZstdDecompressor().copy_stream(source, destination)
        return
    raise RuntimeError(f"unsupported extraction format: {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--download-dir", type=Path, default=ROOT / "artifacts/downloads")
    parser.add_argument("--extract-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assets = {row["name"]: row for row in manifest["assets"]}
    if args.list or (not args.name and not args.all):
        for row in assets.values():
            print(f"{row['name']:<32} {row['bytes'] / 1024 / 1024:8.1f} MiB  {row['purpose']}")
        return 0
    selected = list(assets) if args.all else args.name
    unknown = sorted(set(selected) - set(assets))
    if unknown:
        raise SystemExit(f"unknown asset name(s): {', '.join(unknown)}")
    remote = release_assets(manifest["repository"], manifest["tag"])
    for name in selected:
        row = assets[name]
        if row["filename"] not in remote:
            raise RuntimeError(f"release is missing {row['filename']}")
        target = args.download_dir / row["filename"]
        if not target.is_file() or sha256(target) != row["sha256"]:
            print(f"downloading {name} -> {target}")
            download(row, remote[row["filename"]], target)
        else:
            print(f"verified existing {target}")
        if args.extract:
            destination = args.extract_dir / row["extract_to"]
            print(f"extracting {target.name} -> {destination}")
            extract(target, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
