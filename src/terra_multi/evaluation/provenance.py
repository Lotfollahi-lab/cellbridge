"""Deterministic identities for source files, code, and artifact settings."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def stable_json(payload) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    )


def stable_signature(payload) -> str:
    return hashlib.sha256(stable_json(payload).encode()).hexdigest()


def file_identity(path, *, checksum=False) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    identity = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if checksum:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        identity["sha256"] = digest.hexdigest()
    return identity


def path_identity(path) -> list[dict] | dict:
    path = Path(path)
    if path.is_file():
        return file_identity(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return [file_identity(item) for item in sorted(path.iterdir()) if item.is_file()]


def git_identity(repository) -> dict:
    repository = Path(repository).resolve()
    if not (repository / ".git").exists():
        raise ValueError(f"not a git checkout: {repository}")

    def run_bytes(*arguments):
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout

    def run(*arguments):
        return run_bytes(*arguments).decode().strip()

    status = run("status", "--porcelain")
    digest = hashlib.sha256(run_bytes("diff", "--binary", "HEAD", "--"))
    untracked = run("ls-files", "--others", "--exclude-standard").splitlines()
    for relative in sorted(untracked):
        path = repository / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        if path.is_file():
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
    return {
        "path": str(repository),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "working_tree_signature": digest.hexdigest(),
    }
