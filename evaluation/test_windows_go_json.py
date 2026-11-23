import json

from evaluation.windows_go_json import (
    extract_go_json_test_status,
    filter_windows_go_json_status,
)


def _event(package: str, action: str, test: str | None = None) -> str:
    event = {"Time": "2026-01-01T00:00:00Z", "Action": action, "Package": package}
    if test is not None:
        event["Test"] = test
    return json.dumps(event)


def test_extracts_terminal_status_and_filters_ghosts():
    log = "\n".join(
        [
            _event("example/reportsizes", "pass", "TestString"),
            _event("example/ko", "fail", "TestPublishPipeError"),
            _event("example/shell", "fail"),
        ]
    )
    parsed = {"TestString": "pass", "TestDescription": "fail", "TestPublishPipeError": "fail"}
    assert filter_windows_go_json_status(parsed, log, "windows") == {
        "TestString": "pass",
        "TestPublishPipeError": "fail",
    }


def test_recovers_objects_split_across_physical_windows_lines():
    raw = _event("example/conf", "pass", "TestBadStringEscape")
    wrapped = raw[:62] + "\n" + raw[62:]
    assert extract_go_json_test_status(wrapped) == {"TestBadStringEscape": "pass"}


def test_later_time_boundary_recovers_after_malformed_object():
    malformed = '{"Time":"2026-01-01T00:00:00Z","Action":"output","Package":"example","Output":"unterminated'
    recovered = _event("example/conf", "pass", "TestEmptyStringDQ")
    statuses = extract_go_json_test_status(malformed + "\n" + recovered)
    assert statuses == {"TestEmptyStringDQ": "pass"}


def test_non_windows_and_non_go_logs_keep_existing_status():
    status = {"TestSomething": "fail"}
    assert filter_windows_go_json_status(status, "FAILED TestSomething", "linux") == status
    assert filter_windows_go_json_status(status, "FAILED TestSomething", "windows") == status


