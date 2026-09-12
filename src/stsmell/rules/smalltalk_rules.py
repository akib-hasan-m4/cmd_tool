"""Smells specific to Smalltalk/Pharo, with no direct Java analogue."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .base import Finding, Rule, register


@register
class MissingClassComment(Rule):
    """Class ships without the documentation Pharo expects every class to have.

    No metric and no threshold: reads ``klass.comment``, the string literal
    preceding the ``Class {`` definition in the Tonel file, and fires when it is
    absent or blank. Classes and traits only -- an extension file documents
    nothing of its own.
    """

    id = "missing-class-comment"
    name = "Missing Class Comment"
    severity = "minor"
    description = (
        "Pharo treats the class comment as the primary documentation of a "
        "class's purpose, and the community enforces its presence."
    )
    defaults: dict = {}

    def check_class(self, cm, ctx) -> Iterable[Finding]:
        klass = cm.klass
        if klass.kind not in ("class", "trait"):
            return
        if not klass.name:
            return
        if not (klass.comment or "").strip():
            yield self.class_finding(cm, f"class '{klass.name}' has no class comment")


@register
class GodClassSide(Rule):
    """The metaclass has grown into a utility bag of static-style helpers.

    Metric: NOCM (number of class-side methods -- those declared
    ``Foo class >> bar``). Counted separately from NOM because instance-side
    size and class-side size are different design problems.
    Breaks when NOCM > max_class_methods.
    """

    id = "god-class-side"
    name = "God Class Side"
    severity = "minor"
    description = (
        "Too many class-side methods -- the Smalltalk shape of a utility class "
        "full of static methods."
    )
    defaults = {"max_class_methods": 15}

    def check_class(self, cm, ctx) -> Iterable[Finding]:
        limit = self.opt(ctx, "max_class_methods")
        if cm.nocm > limit:
            yield self.class_finding(
                cm,
                f"defines {cm.nocm} class-side methods (limit {limit})",
                NOCM=cm.nocm,
                limit=limit,
            )


@register
class ExtensionSprawl(Rule):
    """A package monkey-patches foreign classes on a scale that hides behaviour.

    The one project-scoped rule: it implements ``check_project`` rather than
    ``check_class``, since the unit of judgement is the package, not any single
    file. Tallies, per package, how many distinct extended classes and how many
    extension methods it contributes.
    Breaks when either count exceeds its limit -- max_extended_classes or
    max_extension_methods.
    """

    id = "extension-sprawl"
    name = "Extension Sprawl"
    severity = "major"
    description = (
        "A package attaches many methods to classes it does not own. Monkey "
        "patching at scale makes behaviour hard to locate and is a recurring "
        "maintainability problem in Pharo."
    )
    defaults = {"max_extended_classes": 10, "max_extension_methods": 30}

    def check_project(self, ctx) -> Iterable[Finding]:
        max_classes = self.opt(ctx, "max_extended_classes")
        max_methods = self.opt(ctx, "max_extension_methods")

        per_package: dict[str, Counter] = {}
        locations: dict[str, tuple] = {}
        for cm in ctx.suite.classes:
            klass = cm.klass
            if not klass.is_extension:
                continue
            package = klass.package or (
                str(klass.path.parent.name) if klass.path else "<unknown>"
            )
            counts = per_package.setdefault(package, Counter())
            counts[klass.name] += len(klass.methods)
            locations.setdefault(
                package, (str(klass.path) if klass.path else None, klass.start_line)
            )

        for package, counts in sorted(per_package.items()):
            extended = len(counts)
            methods = sum(counts.values())
            if extended <= max_classes and methods <= max_methods:
                continue
            path, line = locations.get(package, (None, 0))
            yield Finding(
                rule=self.id,
                severity=self.severity,
                message=(
                    f"package '{package}' extends {extended} foreign class(es) "
                    f"with {methods} method(s) "
                    f"(limits {max_classes} classes / {max_methods} methods)"
                ),
                entity=package,
                file=path,
                line=line,
                metrics={"extended_classes": extended, "extension_methods": methods},
            )
