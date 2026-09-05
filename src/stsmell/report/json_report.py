"""Machine-readable JSON output."""

from __future__ import annotations

import json

from ..analysis import AnalysisResult


def render_json(result: AnalysisResult) -> str:
    payload = {
        "root": str(result.project.root),
        "summary": {
            "files_scanned": result.files_scanned,
            "classes": len(result.suite.classes),
            "methods": len(result.suite.methods),
            "findings": len(result.findings),
            "parse_errors": len(result.parse_errors),
            "read_errors": len(result.read_errors),
        },
        "parse_errors": [str(p) for p in result.parse_errors],
        "findings": [f.as_dict() for f in result.findings],
    }
    return json.dumps(payload, indent=2)
