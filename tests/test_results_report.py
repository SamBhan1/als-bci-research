"""Regression tests for the checked-in, data-free report artifact."""

import json
from pathlib import Path

import render_results


ROOT = Path(__file__).parents[1]


def test_checked_in_results_have_the_expected_schema_and_render_cleanly():
    results = json.loads((ROOT / "results" / "calibration_free_results.json").read_text(encoding="utf-8"))
    render_results.validate_results(results)
    expected = render_results.render(results)
    checked_in = (ROOT / "RESULTS.md").read_text(encoding="utf-8")
    assert checked_in == expected
    assert "## primary pipeline (`binned`)" in checked_in
    assert "p (exact sign-flip)" in checked_in
