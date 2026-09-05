"""Per-class metrics, aggregated from method metrics plus the hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

from ..model import ClassDef
from .hierarchy import Hierarchy
from .method import MethodMetrics


@dataclass
class ClassMetrics:
    klass: ClassDef
    nom: int = 0
    nocm: int = 0
    wmc: int = 0
    niv: int = 0
    cloc: int = 0
    dit: int = 0
    noc: int = 0
    cbo: int = 0
    tcc: Optional[float] = None
    acc_ratio: float = 0.0
    coupled_classes: set[str] = field(default_factory=set)
    method_metrics: list[MethodMetrics] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "class": self.klass.name,
            "kind": self.klass.kind,
            "package": self.klass.package,
            "file": str(self.klass.path) if self.klass.path else None,
            "line": self.klass.start_line,
            "NOM": self.nom,
            "NOCM": self.nocm,
            "WMC": self.wmc,
            "NIV": self.niv,
            "CLOC": self.cloc,
            "DIT": self.dit,
            "NOC": self.noc,
            "CBO": self.cbo,
            "TCC": round(self.tcc, 3) if self.tcc is not None else None,
            "ACC_RATIO": round(self.acc_ratio, 3),
        }


def _tight_class_cohesion(metrics: list[MethodMetrics]) -> Optional[float]:
    """Fraction of eligible method pairs that touch a common instance variable.

    Accessors and abstract methods are excluded (they trivially touch one or
    zero variables and would otherwise dominate the score). Undefined -- and so
    ``None`` -- for classes with fewer than two eligible methods.
    """
    eligible = [
        m for m in metrics if not m.is_accessor and not m.is_abstract and not m.method.class_side
    ]
    if len(eligible) < 2:
        return None
    pairs = list(combinations(eligible, 2))
    if not pairs:
        return None
    connected = sum(1 for a, b in pairs if a.accessed_ivars & b.accessed_ivars)
    return connected / len(pairs)


def compute_class_metrics(
    klass: ClassDef,
    method_metrics: list[MethodMetrics],
    hierarchy: Hierarchy,
    project_classes: set[str],
) -> ClassMetrics:
    cm = ClassMetrics(klass=klass, method_metrics=method_metrics)

    cm.nom = len(klass.methods)
    cm.nocm = len(klass.class_methods)
    cm.niv = len(klass.inst_vars)
    cm.wmc = sum(m.cyclo for m in method_metrics)
    cm.cloc = sum(m.mloc for m in method_metrics)
    cm.dit = hierarchy.depth.get(klass.name, 0)
    cm.noc = hierarchy.num_children(klass.name)

    referenced: set[str] = set()
    for m in method_metrics:
        referenced |= m.referenced_classes
    cm.coupled_classes = (referenced & project_classes) - {klass.name}
    cm.cbo = len(cm.coupled_classes)

    cm.tcc = _tight_class_cohesion(method_metrics)
    if cm.nom:
        cm.acc_ratio = sum(1 for m in method_metrics if m.is_accessor) / cm.nom

    return cm
