"""Configuration: rule toggles and thresholds, overridable via TOML."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - fallback for 3.10
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:  # pragma: no cover
        tomllib = None  # type: ignore

CONFIG_FILENAMES = ("stsmell.toml", ".stsmell.toml")

SEVERITY_ORDER = {"info": 0, "minor": 1, "major": 2, "critical": 3}


class ConfigError(Exception):
    pass


@dataclass
class Config:
    #: rule id -> {"enabled": bool, <threshold name>: value}
    rules: dict[str, dict[str, Any]] = field(default_factory=dict)
    excludes: list[str] = field(default_factory=list)
    min_severity: str = "info"
    source: Optional[Path] = None

    def options(self, rule_id: str) -> dict[str, Any]:
        return self.rules.get(rule_id, {})

    def enabled(self, rule_id: str) -> bool:
        return bool(self.options(rule_id).get("enabled", True))

    def threshold(self, rule_id: str, name: str, default: Any) -> Any:
        return self.options(rule_id).get(name, default)

    def with_defaults(self, registry) -> "Config":
        """Fill in each rule's declared defaults for anything unset."""
        for rule in registry:
            merged = dict(rule.defaults)
            merged.setdefault("enabled", True)
            merged.update(self.rules.get(rule.id, {}))
            self.rules[rule.id] = merged
        return self


def find_config(start: Path) -> Optional[Path]:
    """Search ``start`` and its parents for a config file."""
    start = start.resolve()
    candidates = [start] if start.is_dir() else [start.parent]
    candidates.extend(candidates[0].parents)
    for directory in candidates:
        for name in CONFIG_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def load_config(path: Optional[Path]) -> Config:
    if path is None:
        return Config()
    if tomllib is None:  # pragma: no cover
        raise ConfigError("TOML support requires Python 3.11+ or the 'tomli' package")
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    except Exception as exc:  # tomllib.TOMLDecodeError
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    config = Config(source=path)
    config.excludes = list(data.get("exclude", []) or [])
    min_severity = data.get("min_severity")
    if min_severity:
        if min_severity not in SEVERITY_ORDER:
            raise ConfigError(
                f"min_severity must be one of {sorted(SEVERITY_ORDER)}, got {min_severity!r}"
            )
        config.min_severity = min_severity

    rules = data.get("rules", {})
    if not isinstance(rules, dict):
        raise ConfigError("[rules] must be a table")
    for rule_id, options in rules.items():
        if not isinstance(options, dict):
            raise ConfigError(f"[rules.{rule_id}] must be a table")
        config.rules[rule_id] = dict(options)
    return config
