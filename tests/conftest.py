from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


def analyze_dir(path: Path, config=None):
    from stsmell.analysis import analyze

    return analyze(path, config)


def metrics_for(path: Path):
    """Return {selector: MethodMetrics} and {name: ClassMetrics} for one file."""
    from stsmell.extract import extract_file
    from stsmell.metrics import compute_metrics
    from stsmell.model import Project

    parsed = extract_file(path)
    suite = compute_metrics(Project(root=path.parent, files=[parsed]))
    methods = {m.method.selector: m for m in suite.methods}
    classes = {c.klass.name: c for c in suite.classes}
    return methods, classes
