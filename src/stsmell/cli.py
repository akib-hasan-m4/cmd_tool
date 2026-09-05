"""Command line interface."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .analysis import analyze
from .config import SEVERITY_ORDER, Config, ConfigError, find_config, load_config
from .report import render_json, render_sarif, render_text
from .rules import registry

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2


def _load(args) -> Config:
    path = Path(args.config) if getattr(args, "config", None) else find_config(Path(args.path))
    config = load_config(path)
    if getattr(args, "exclude", None):
        config.excludes.extend(args.exclude)
    if getattr(args, "min_severity", None):
        config.min_severity = args.min_severity
    return config


def _load_baseline(path: Optional[str]) -> set[tuple]:
    if not path:
        return set()
    data = json.loads(Path(path).read_text())
    return {(entry["rule"], entry["entity"]) for entry in data.get("findings", [])}


def cmd_analyze(args) -> int:
    config = _load(args)
    result = analyze(Path(args.path).resolve(), config)

    baseline = _load_baseline(args.baseline)
    if baseline:
        result.findings = [f for f in result.findings if f.key() not in baseline]

    floor = SEVERITY_ORDER[config.min_severity]
    result.findings = [f for f in result.findings if f.severity_rank >= floor]

    if args.format == "json":
        print(render_json(result))
    elif args.format == "sarif":
        print(render_sarif(result))
    else:
        print(render_text(result, sys.stdout, root=Path(args.path).resolve()))

    if result.read_errors:
        for path, message in result.read_errors:
            print(f"error: cannot read {path}: {message}", file=sys.stderr)
    return EXIT_FINDINGS if result.findings else EXIT_OK


def cmd_metrics(args) -> int:
    config = _load(args)
    result = analyze(Path(args.path).resolve(), config)

    scope = args.scope
    if scope == "class":
        rows = [cm.as_dict() for cm in result.suite.classes]
    else:
        rows = [mm.as_dict() for mm in result.suite.methods]

    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        if not rows:
            return EXIT_OK
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        print(buffer.getvalue(), end="")
    return EXIT_OK


def cmd_rules(args) -> int:
    config = load_config(Path(args.config) if args.config else None).with_defaults(registry())
    if args.format == "json":
        print(
            json.dumps(
                [
                    {
                        "id": r.id,
                        "name": r.name,
                        "severity": r.severity,
                        "description": r.description,
                        "options": config.options(r.id),
                    }
                    for r in registry()
                ],
                indent=2,
            )
        )
        return EXIT_OK

    for rule in registry():
        options = {k: v for k, v in config.options(rule.id).items() if k != "enabled"}
        state = "" if config.enabled(rule.id) else "  (disabled)"
        print(f"{rule.id}  [{rule.severity}]{state}")
        print(f"    {rule.description}")
        if options:
            print(f"    thresholds: {options}")
        print()
    return EXIT_OK


def cmd_baseline(args) -> int:
    config = _load(args)
    result = analyze(Path(args.path).resolve(), config)
    payload = {
        "version": __version__,
        "findings": [
            {"rule": f.rule, "entity": f.entity, "message": f.message}
            for f in result.findings
        ],
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
        print(
            f"wrote baseline with {len(result.findings)} finding(s) to {args.output}",
            file=sys.stderr,
        )
    else:
        print(text)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stsmell",
        description="Detect code smells in Pharo/Smalltalk source (Tonel format).",
    )
    parser.add_argument("--version", action="version", version=f"stsmell {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub):
        sub.add_argument("path", help="Tonel file or directory to analyse")
        sub.add_argument("-c", "--config", help="path to a stsmell.toml")
        sub.add_argument(
            "-e", "--exclude", action="append", default=[], help="glob to exclude (repeatable)"
        )

    analyze_parser = subparsers.add_parser("analyze", help="report code smells")
    add_common(analyze_parser)
    analyze_parser.add_argument(
        "-f", "--format", choices=("text", "json", "sarif"), default="text"
    )
    analyze_parser.add_argument(
        "--min-severity", choices=sorted(SEVERITY_ORDER, key=SEVERITY_ORDER.get)
    )
    analyze_parser.add_argument("--baseline", help="suppress findings listed in this file")
    analyze_parser.set_defaults(func=cmd_analyze)

    metrics_parser = subparsers.add_parser("metrics", help="dump raw metrics")
    add_common(metrics_parser)
    metrics_parser.add_argument("-f", "--format", choices=("csv", "json"), default="csv")
    metrics_parser.add_argument("-s", "--scope", choices=("method", "class"), default="method")
    metrics_parser.set_defaults(func=cmd_metrics)

    rules_parser = subparsers.add_parser("rules", help="list rules and thresholds")
    rules_parser.add_argument("-c", "--config", help="path to a stsmell.toml")
    rules_parser.add_argument("-f", "--format", choices=("text", "json"), default="text")
    rules_parser.set_defaults(func=cmd_rules)

    baseline_parser = subparsers.add_parser(
        "baseline", help="freeze current findings so only new ones are reported"
    )
    add_common(baseline_parser)
    baseline_parser.add_argument("-o", "--output", help="write baseline here (default stdout)")
    baseline_parser.set_defaults(func=cmd_baseline)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except BrokenPipeError:  # e.g. piping into head
        return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
