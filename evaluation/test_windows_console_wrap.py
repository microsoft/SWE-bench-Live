from evaluation.evaluation import (
    _build_windows_test_name_lookup,
    _normalize_windows_test_name_artifacts,
    _resolve_test_status,
    _summarize_expected_tests,
)


def test_wrap_normalization_removes_inserted_space_and_duplicate_boundary():
    assert _normalize_windows_test_name_artifacts("ct ty") == "cty"
    assert _normalize_windows_test_name_artifacts("NumberIntVal(1 12)") == "NumberIntVal(12)"


def test_exact_match_precedes_windows_fallback():
    status = {"ct ty": "pass", "cty": "fail"}
    lookup = _build_windows_test_name_lookup(list(status))
    assert _resolve_test_status("ct ty", status, "windows", lookup) == "pass"


def test_ambiguous_windows_fallback_fails_closed():
    status = {"cty": "pass", "ct ty": "fail"}
    lookup = _build_windows_test_name_lookup(list(status))
    assert _resolve_test_status("ct  ty", status, "windows", lookup) is None


def test_linux_keeps_exact_matching_and_does_not_normalize():
    status = {"cty": "pass"}
    assert _resolve_test_status("ct ty", status, "linux", {}) is None
    assert _resolve_test_status("cty", status, "linux", {}) == "pass"


def test_classification_reports_missing_and_skipped_without_guessing():
    status = {"cty": "pass", "skipped": "skip"}
    assert _summarize_expected_tests(
        ["ct ty", "skipped", "unknown"], status, "linux"
    ) == {
        "success": [],
        "failure": [],
        "skipped": ["skipped"],
        "missing": ["ct ty", "unknown"],
    }
    assert _summarize_expected_tests(
        ["ct ty"], status, "windows"
    )["success"] == ["ct ty"]