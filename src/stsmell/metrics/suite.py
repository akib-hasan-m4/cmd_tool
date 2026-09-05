"""Run every metric over a project."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Project
from .hierarchy import Hierarchy
from .klass import ClassMetrics, compute_class_metrics
from .method import MethodMetrics, compute_method_metrics


@dataclass
class MetricSuite:
    project: Project
    hierarchy: Hierarchy
    classes: list[ClassMetrics] = field(default_factory=list)

    @property
    def methods(self) -> list[MethodMetrics]:
        return [m for c in self.classes for m in c.method_metrics]

    def class_by_name(self, name: str):
        for c in self.classes:
            if c.klass.name == name:
                return c
        return None


def compute_metrics(project: Project) -> MetricSuite:
    hierarchy = Hierarchy.build(project)
    project_classes = project.class_names()
    suite = MetricSuite(project=project, hierarchy=hierarchy)

    for parsed in project.files:
        for klass in parsed.classes:
            method_metrics = [
                compute_method_metrics(method, klass, parsed.source)
                for method in klass.methods
            ]
            suite.classes.append(
                compute_class_metrics(klass, method_metrics, hierarchy, project_classes)
            )
    return suite
