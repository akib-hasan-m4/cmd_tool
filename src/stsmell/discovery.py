"""Locate Tonel source files on disk."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable

TONEL_SUFFIXES = (".class.st", ".trait.st", ".extension.st")

DEFAULT_EXCLUDES = (
    "*/.git/*",
    "*/node_modules/*",
    "*/.venv/*",
    "*/pharo-local/*",
)


def is_tonel_file(path: Path) -> bool:
    name = path.name
    if name == "package.st":
        return False  # package marker, carries no class or method
    return any(name.endswith(suffix) for suffix in TONEL_SUFFIXES)


def _excluded(path: Path, patterns: Iterable[str]) -> bool:
    text = str(path)
    return any(fnmatch.fnmatch(text, pattern) for pattern in patterns)


def discover(root: Path, excludes: Iterable[str] = ()) -> list[Path]:
    """Return Tonel files under ``root`` (or ``[root]`` if it is itself one)."""
    patterns = list(DEFAULT_EXCLUDES) + list(excludes)
    root = root.resolve()

    if root.is_file():
        return [root] if is_tonel_file(root) else []

    found = [
        p
        for p in root.rglob("*.st")
        if is_tonel_file(p) and not _excluded(p, patterns)
    ]
    return sorted(found)
