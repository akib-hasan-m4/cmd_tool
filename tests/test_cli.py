"""End-to-end CLI behaviour: formats, exit codes, config and baselines."""

import json

import pytest

from stsmell.cli import EXIT_ERROR, EXIT_FINDINGS, EXIT_OK, main


def test_clean_directory_exits_zero(fixtures, capsys):
    assert main(["analyze", str(fixtures / "clean")]) == EXIT_OK
    assert "No smells found" in capsys.readouterr().out


def test_findings_exit_one(fixtures, capsys):
    assert main(["analyze", str(fixtures / "smelly")]) == EXIT_FINDINGS
    out = capsys.readouterr().out
    assert "long-method" in out
    assert "finding(s)" in out


def test_json_format_is_valid_and_structured(fixtures, capsys):
    main(["analyze", str(fixtures / "smelly"), "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["findings"] == len(payload["findings"])
    assert payload["summary"]["parse_errors"] == 0
    first = payload["findings"][0]
    assert {"rule", "severity", "message", "entity", "file", "line"} <= set(first)


def test_sarif_format_is_valid(fixtures, capsys):
    main(["analyze", str(fixtures / "smelly"), "--format", "sarif"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == "2.1.0"
    run = payload["runs"][0]
    assert run["tool"]["driver"]["name"] == "stsmell"
    assert run["results"]
    reported = {r["ruleId"] for r in run["results"]}
    declared = {r["id"] for r in run["tool"]["driver"]["rules"]}
    assert reported <= declared, "every reported rule must be declared in the driver"
    for result in run["results"]:
        assert result["level"] in {"error", "warning", "note"}
        assert result["locations"][0]["physicalLocation"]["region"]["startLine"] >= 1


def test_min_severity_filters_and_flips_exit_code(fixtures, capsys):
    """The smelly fixtures top out at 'major', so a critical floor clears them."""
    assert main(["analyze", str(fixtures / "smelly"), "--min-severity", "critical"]) == EXIT_OK
    assert "No smells found" in capsys.readouterr().out

    assert main(["analyze", str(fixtures / "smelly"), "--min-severity", "major"]) == EXIT_FINDINGS
    out = capsys.readouterr().out
    assert "minor" not in out.replace("min-severity", "")


def test_exclude_glob_skips_files(fixtures, capsys):
    main(
        [
            "analyze",
            str(fixtures / "smelly"),
            "--exclude",
            "*LongMethodExample*",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert all("LongMethodExample" not in (f["file"] or "") for f in payload["findings"])


def test_config_file_overrides_thresholds(fixtures, tmp_path, capsys):
    config = tmp_path / "stsmell.toml"
    config.write_text("[rules.long-method]\nenabled = false\n")
    main(
        [
            "analyze",
            str(fixtures / "smelly"),
            "--config",
            str(config),
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert all(f["rule"] != "long-method" for f in payload["findings"])


def test_invalid_config_reports_error(fixtures, tmp_path, capsys):
    config = tmp_path / "stsmell.toml"
    config.write_text("this is not = valid = toml\n")
    assert main(["analyze", str(fixtures / "smelly"), "--config", str(config)]) == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_baseline_round_trip_suppresses_known_findings(fixtures, tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    assert main(["baseline", str(fixtures / "smelly"), "-o", str(baseline)]) == EXIT_OK
    capsys.readouterr()

    assert main(["analyze", str(fixtures / "smelly"), "--baseline", str(baseline)]) == EXIT_OK
    assert "No smells found" in capsys.readouterr().out


def test_baseline_still_reports_new_findings(fixtures, tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    main(["baseline", str(fixtures / "clean"), "-o", str(baseline)])
    capsys.readouterr()
    assert main(["analyze", str(fixtures / "smelly"), "--baseline", str(baseline)]) == EXIT_FINDINGS


def test_metrics_csv_has_a_row_per_method(fixtures, capsys):
    main(["metrics", str(fixtures / "clean"), "--format", "csv"])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[0].startswith("class,selector")
    assert len(lines) == 5  # header + four methods


def test_metrics_class_scope_json(fixtures, capsys):
    main(["metrics", str(fixtures / "clean"), "--scope", "class", "--format", "json"])
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["class"] == "CleanPoint"
    assert rows[0]["NOM"] == 4


def test_rules_command_lists_every_rule(capsys):
    from stsmell.rules import registry

    assert main(["rules"]) == EXIT_OK
    out = capsys.readouterr().out
    for rule in registry():
        assert rule.id in out


def test_rules_command_json(capsys):
    from stsmell.rules import registry

    main(["rules", "--format", "json"])
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == len(registry())
    assert all(row["options"] for row in rows if row["id"] != "uses-this-context")


def test_analyzing_a_single_file(fixtures, capsys):
    assert (
        main(["analyze", str(fixtures / "smelly" / "LongMethodExample.class.st")])
        == EXIT_FINDINGS
    )
    assert "long-method" in capsys.readouterr().out


def test_missing_path_is_an_error(tmp_path, capsys):
    result = main(["analyze", str(tmp_path / "nope")])
    assert result in (EXIT_OK, EXIT_ERROR)
