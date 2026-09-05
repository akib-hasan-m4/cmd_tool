"""Output formats."""

from .text import render_text
from .json_report import render_json
from .sarif import render_sarif

__all__ = ["render_text", "render_json", "render_sarif"]
