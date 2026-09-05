"""Rule definitions. Importing this package populates the registry."""

from .base import Finding, Rule, RuleContext, registry, run_rules
from . import method_rules  # noqa: F401  (registers rules)
from . import class_rules  # noqa: F401
from . import smalltalk_rules  # noqa: F401

__all__ = ["Finding", "Rule", "RuleContext", "registry", "run_rules"]
