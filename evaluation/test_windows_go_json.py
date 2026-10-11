from .windows_go_json import (
    extract_go_json_test_keys,
    extract_go_json_test_status,
    filter_unsupported_go_json_status,
)


def event(package, action, test=None):
    fields = {"Package": package, "Action": action}
    if test is not None:
        fields["Test"] = test
    import json
    return json.dumps(fields)


def test_extracts_only_explicit_terminal_test_events():
    log = "\n".join(
        [
            event("example/reportsizes", "run", "TestString"),
            event("example/reportsizes", "pass", "TestString"),
            event("example/reportsizes", "fail"),
            event("example/shell", "pass", "TestRunCommand"),
        ]
    )
    assert extract_go_json_test_keys(log) == {
        "example/reportsizes::TestString",
        "example/shell::TestRunCommand",
    }


def test_windows_filter_drops_ghost_parser_statuses_and_keeps_real_failures():
    log = "\n".join(
        [
            event("example/reportsizes", "pass", "TestString"),
            event("example/ko", "fail", "TestPublishPipeError"),
        ]
    )
    status = {
        "example/reportsizes::TestDescription": "fail",
        "example/reportsizes::TestString": "pass",
        "example/ko::TestPublishPipeError": "fail",
    }
    assert filter_unsupported_go_json_status(status, log, "windows") == {
        "example/reportsizes::TestString": "pass",
        "example/ko::TestPublishPipeError": "fail",
    }


def test_keeps_same_test_name_separate_by_package():
    log = "\n".join(
        [
            event("example/a", "pass", "TestSame"),
            event("example/b", "fail", "TestSame"),
        ]
    )
    assert extract_go_json_test_status(log) == {
        "example/a::TestSame": "pass",
        "example/b::TestSame": "fail",
    }


def test_wrapped_events_override_a_conflicting_parser_status():
    wrapped_log = (
        '{"Package":"example/test",\n'
        '"Action":"pass","Test":"TestGatewayTLSMixedIPAndDNS"}'
    )
    status = {"example/test::TestGatewayTLSMixedIPAndDNS": "fail", "Ghost": "fail"}
    assert extract_go_json_test_status(wrapped_log) == {
        "example/test::TestGatewayTLSMixedIPAndDNS": "pass"
    }
    assert filter_unsupported_go_json_status(status, wrapped_log, "windows") == {
        "example/test::TestGatewayTLSMixedIPAndDNS": "pass"
    }


def test_non_windows_and_non_go_logs_are_unchanged():
    status = {"TestSomething": "fail"}
    assert filter_unsupported_go_json_status(status, "FAILED TestSomething", "linux") == status
    assert filter_unsupported_go_json_status(status, "FAILED TestSomething", "windows") == status