"""SARIF 2.1.0 output, for GitHub code scanning."""

from __future__ import annotations

import json
from pathlib import Path

from .. import __version__
from ..analysis import AnalysisResult
from ..rules import registry

# SARIF has three levels; map our four severities onto them.
_LEVEL = {"critical": "error", "major": "error", "minor": "warning", "info": "note"}


def _uri(path: str | None, root: Path) -> str:
    if not path:
        return "unknown"
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()


def render_sarif(result: AnalysisResult) -> str:
    root = result.project.root
    used = {f.rule for f in result.findings}
    rules = [
        {
            "id": rule.id,
            "name": rule.name,
            "shortDescription": {"text": rule.name},
            "fullDescription": {"text": rule.description},
            "defaultConfiguration": {"level": _LEVEL.get(rule.severity, "warning")},
            "properties": {"tags": ["code-smell", "smalltalk"]},
        }
        for rule in registry()
        if rule.id in used
    ]

    results = [
        {
            "ruleId": f.rule,
            "level": _LEVEL.get(f.severity, "warning"),
            # Keyword selectors already end in ':', so a colon separator would
            # render as 'classify:: ...'.
            "message": {"text": f"{f.entity} \u2014 {f.message}"},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": _uri(f.file, root)},
                        "region": {"startLine": max(f.line, 1)},
                    }
                }
            ],
        }
        for f in result.findings
    ]

    return json.dumps(
        {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "stsmell",
                            "version": __version__,
                            "informationUri": "https://github.com/",
                            "rules": rules,
                        }
                    },
                    "results": results,
                }
            ],
        },
        indent=2,
    )
