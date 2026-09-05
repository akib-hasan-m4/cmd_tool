"""Rule base class, finding record, and the registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..config import SEVERITY_ORDER, Config
from ..metrics import MetricSuite
from ..metrics.klass import ClassMetrics
from ..metrics.method import MethodMetrics


@dataclass
class Finding:
    rule: str
    severity: str
    message: str
    entity: str
    file: Optional[str] = None
    line: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def severity_rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, 0)

    def key(self) -> tuple:
        """Stable identity for baseline matching (line-number independent)."""
        return (self.rule, self.entity)

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "entity": self.entity,
            "file": self.file,
            "line": self.line,
            "metrics": self.metrics,
        }


@dataclass
class RuleContext:
    suite: MetricSuite
    config: Config


class Rule:
    """Base class for all rules.

    Rules read computed metrics and the model only -- never the CST -- which
    keeps them trivially unit-testable.
    """

    id: str = ""
    name: str = ""
    severity: str = "minor"
    description: str = ""
    defaults: dict[str, Any] = {}

    def opt(self, ctx: RuleContext, name: str) -> Any:
        return ctx.config.threshold(self.id, name, self.defaults.get(name))

    def check_method(
        self, mm: MethodMetrics, cm: ClassMetrics, ctx: RuleContext
    ) -> Iterable[Finding]:
        return ()

    def check_class(self, cm: ClassMetrics, ctx: RuleContext) -> Iterable[Finding]:
        return ()

    def check_project(self, ctx: RuleContext) -> Iterable[Finding]:
        return ()

    # -- helpers -------------------------------------------------------
    def method_finding(self, mm: MethodMetrics, message: str, **metrics) -> Finding:
        return Finding(
            rule=self.id,
            severity=self.severity,
            message=message,
            entity=mm.method.qualified_name,
            file=str(mm.method.path) if mm.method.path else None,
            line=mm.method.start_line,
            metrics=metrics,
        )

    def class_finding(self, cm: ClassMetrics, message: str, **metrics) -> Finding:
        return Finding(
            rule=self.id,
            severity=self.severity,
            message=message,
            entity=cm.klass.name,
            file=str(cm.klass.path) if cm.klass.path else None,
            line=cm.klass.start_line,
            metrics=metrics,
        )


_REGISTRY: list[Rule] = []


def register(rule_cls):
    _REGISTRY.append(rule_cls())
    return rule_cls


def registry() -> list[Rule]:
    return list(_REGISTRY)


def run_rules(suite: MetricSuite, config: Config) -> list[Finding]:
    ctx = RuleContext(suite=suite, config=config)
    active = [r for r in registry() if config.enabled(r.id)]
    findings: list[Finding] = []

    for rule in active:
        findings.extend(rule.check_project(ctx))

    for cm in suite.classes:
        for rule in active:
            findings.extend(rule.check_class(cm, ctx))
        for mm in cm.method_metrics:
            for rule in active:
                findings.extend(rule.check_method(mm, cm, ctx))

    findings.sort(key=lambda f: (-f.severity_rank, f.file or "", f.line, f.rule))
    return findings
