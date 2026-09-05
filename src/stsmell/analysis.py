"""Top-level pipeline: discovery -> parse -> extract -> metrics -> rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import Config
from .discovery import discover
from .extract import extract_file
from .metrics import MetricSuite, compute_metrics
from .model import Project
from .rules import Finding, registry, run_rules


@dataclass
class AnalysisResult:
    project: Project
    suite: MetricSuite
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    parse_errors: list[Path] = field(default_factory=list)
    read_errors: list[tuple[Path, str]] = field(default_factory=list)


def build_project(root: Path, config: Config) -> tuple[Project, list[Path], list[tuple[Path, str]]]:
    paths = discover(root, config.excludes)
    project = Project(root=root)
    parse_errors: list[Path] = []
    read_errors: list[tuple[Path, str]] = []

    for path in paths:
        try:
            parsed = extract_file(path)
        except (OSError, UnicodeDecodeError) as exc:
            read_errors.append((path, str(exc)))
            continue
        if parsed.parse_error:
            # Keep the file: tree-sitter recovers, so partial results are still
            # useful. Surface the count rather than failing silently.
            parse_errors.append(path)
        project.files.append(parsed)

    return project, parse_errors, read_errors


def analyze(root: Path, config: Optional[Config] = None) -> AnalysisResult:
    config = (config or Config()).with_defaults(registry())
    project, parse_errors, read_errors = build_project(root, config)
    suite = compute_metrics(project)
    findings = run_rules(suite, config)
    return AnalysisResult(
        project=project,
        suite=suite,
        findings=findings,
        files_scanned=len(project.files),
        parse_errors=parse_errors,
        read_errors=read_errors,
    )
