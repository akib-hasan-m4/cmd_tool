"""Human-readable terminal output."""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

from ..analysis import AnalysisResult

_COLORS = {
    "critical": "\033[1;31m",
    "major": "\033[31m",
    "minor": "\033[33m",
    "info": "\033[36m",
}
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _use_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def render_text(result: AnalysisResult, stream=None, root: Path | None = None) -> str:
    stream = stream or sys.stdout
    color = _use_color(stream)

    def paint(text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if color else text

    lines: list[str] = []
    root = root or result.project.root

    def rel(path: str | None) -> str:
        if not path:
            return "<unknown>"
        try:
            return str(Path(path).relative_to(root))
        except ValueError:
            return path

    if not result.findings:
        lines.append(paint("No smells found.", _BOLD))
    else:
        current_file = object()
        for finding in result.findings:
            if finding.file != current_file:
                current_file = finding.file
                lines.append("")
                lines.append(paint(rel(finding.file), _BOLD))
            severity = paint(
                f"{finding.severity:>8}", _COLORS.get(finding.severity, "")
            )
            lines.append(
                f"  {finding.line:>5}  {severity}  {finding.message}"
                f"  {paint('[' + finding.rule + ']', _DIM)}"
            )
            lines.append(f"         {paint(finding.entity, _DIM)}")

    counts = Counter(f.severity for f in result.findings)
    summary = ", ".join(
        f"{counts[s]} {s}" for s in ("critical", "major", "minor", "info") if counts[s]
    )
    lines.append("")
    lines.append(
        paint(
            f"{len(result.findings)} finding(s)"
            + (f" ({summary})" if summary else "")
            + f" across {result.files_scanned} file(s), "
            f"{len(result.suite.classes)} class(es), "
            f"{len(result.suite.methods)} method(s)",
            _BOLD,
        )
    )
    if result.parse_errors:
        lines.append(
            paint(
                f"{len(result.parse_errors)} file(s) had parse errors "
                "(results for those files may be incomplete)",
                _COLORS["minor"],
            )
        )
    if result.read_errors:
        lines.append(
            paint(f"{len(result.read_errors)} file(s) could not be read", _COLORS["major"])
        )
    return "\n".join(lines)
