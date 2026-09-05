"""Class-level smells."""

from __future__ import annotations

from typing import Iterable

from .base import Finding, Rule, register


@register
class GodClass(Rule):
    id = "god-class"
    name = "God Class"
    severity = "critical"
    description = (
        "A class that does too much: high total complexity, low cohesion and "
        "lots of state. Composite detection strategy after Lanza & Marinescu."
    )
    defaults = {"min_wmc": 47, "max_tcc": 0.33, "min_niv": 5}

    def check_class(self, cm, ctx) -> Iterable[Finding]:
        if cm.klass.is_extension or cm.tcc is None:
            return
        min_wmc = self.opt(ctx, "min_wmc")
        max_tcc = self.opt(ctx, "max_tcc")
        min_niv = self.opt(ctx, "min_niv")
        if cm.wmc >= min_wmc and cm.tcc < max_tcc and cm.niv >= min_niv:
            yield self.class_finding(
                cm,
                f"god class: WMC={cm.wmc} (>= {min_wmc}), TCC={cm.tcc:.2f} "
                f"(< {max_tcc}), {cm.niv} instance variables (>= {min_niv})",
                WMC=cm.wmc,
                TCC=round(cm.tcc, 3),
                NIV=cm.niv,
            )


@register
class LargeClass(Rule):
    id = "large-class"
    name = "Large Class"
    severity = "major"
    description = "Class defines too many methods."
    defaults = {"max_methods": 40}

    def check_class(self, cm, ctx) -> Iterable[Finding]:
        limit = self.opt(ctx, "max_methods")
        if cm.nom > limit:
            yield self.class_finding(
                cm,
                f"defines {cm.nom} methods (limit {limit})",
                NOM=cm.nom,
                limit=limit,
            )


@register
class DataClass(Rule):
    id = "data-class"
    name = "Data Class"
    severity = "minor"
    description = (
        "Little more than accessors around instance variables; behaviour that "
        "uses this data probably lives elsewhere."
    )
    defaults = {"min_accessor_ratio": 0.8, "min_methods": 5, "min_ivars": 2}

    def check_class(self, cm, ctx) -> Iterable[Finding]:
        if cm.klass.is_extension:
            return
        if (
            cm.nom >= self.opt(ctx, "min_methods")
            and cm.niv >= self.opt(ctx, "min_ivars")
            and cm.acc_ratio >= self.opt(ctx, "min_accessor_ratio")
        ):
            yield self.class_finding(
                cm,
                f"{cm.acc_ratio:.0%} of its {cm.nom} methods are plain accessors",
                ACC_RATIO=round(cm.acc_ratio, 3),
                NOM=cm.nom,
                NIV=cm.niv,
            )


@register
class RefusedBequest(Rule):
    id = "refused-bequest"
    name = "Refused Bequest"
    severity = "major"
    description = (
        "A subclass rejects what it inherits by overriding an inherited method "
        "with subclassResponsibility."
    )
    defaults: dict = {}

    def check_class(self, cm, ctx) -> Iterable[Finding]:
        klass = cm.klass
        if klass.is_extension or not klass.superclass:
            return
        hierarchy = ctx.suite.hierarchy
        if klass.superclass not in hierarchy.by_name:
            return  # superclass outside the analysed source; cannot judge

        parent = hierarchy.by_name[klass.superclass]
        parent_selectors = {m.selector for m in parent.methods if not m.class_side}

        refused = [
            mm
            for mm in cm.method_metrics
            if mm.is_abstract and mm.method.selector in parent_selectors
        ]
        # Deliberately *not* also flagging subclasses that never touch inherited
        # instance variables. That heuristic works in Java but misfires badly in
        # Smalltalk, where idiomatic code reaches inherited state through
        # accessors (``self origin``) rather than direct variable reference. On
        # the Pharo corpus it produced 3707 findings against 16 real ones.
        if refused:
            names = ", ".join(sorted(m.method.selector for m in refused)[:3])
            yield self.class_finding(
                cm,
                f"overrides inherited method(s) with subclassResponsibility: {names}",
                superclass=klass.superclass,
                refused=len(refused),
            )


@register
class DeepInheritance(Rule):
    id = "deep-inheritance"
    name = "Deep Inheritance"
    severity = "minor"
    description = "Long inheritance chain within the analysed source."
    defaults = {"max_depth": 8}

    def check_class(self, cm, ctx) -> Iterable[Finding]:
        limit = self.opt(ctx, "max_depth")
        if cm.dit > limit:
            yield self.class_finding(
                cm,
                f"inheritance depth {cm.dit} (limit {limit})",
                DIT=cm.dit,
                limit=limit,
            )


@register
class HighCoupling(Rule):
    id = "high-coupling"
    name = "High Coupling"
    severity = "minor"
    description = "References many other classes in the project."
    defaults = {"max_cbo": 15}

    def check_class(self, cm, ctx) -> Iterable[Finding]:
        limit = self.opt(ctx, "max_cbo")
        if cm.cbo > limit:
            yield self.class_finding(
                cm,
                f"references {cm.cbo} other project classes (limit {limit})",
                CBO=cm.cbo,
                limit=limit,
            )
