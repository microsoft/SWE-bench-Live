from evaluation.windows_validation import (
    is_corrupted_windows_test_name,
    normalize_windows_validation_status,
    parse_windows_go_json_status,
)


def test_windows_go_json_replaces_fragment_prone_custom_parser_statuses():
    log = "\n".join(
        [
            '{"Action":"pass","Package":"example/pkg","Test":"TestStable"}',
            '{"Action":"fail","Package":"example/pkg","Test":"TestRegression"}',
            '{"Action":"pass","Package":"example/pkg","Test":"TestWrapped"',
            '  (0.00s)\\n"}',
        ]
    )
    parser_status = {
        "TestStable": "pass",
        'TestWrapped (0.00s)\\n"}': "pass",
        "TestFallbackOnly": "pass",
    }

    assert parse_windows_go_json_status(log) == {
        "example/pkg::TestStable": "pass",
        "example/pkg::TestRegression": "fail",
    }
    assert normalize_windows_validation_status(
        "custom parser",
        log,
        parser_status,
        "windows",
    ) == {"example/pkg::TestStable": "pass", "example/pkg::TestRegression": "fail"}


def test_package_qualified_identities_do_not_collapse_same_named_go_tests():
    log = "\n".join(
        [
            '{"Action":"pass","Package":"example/first","Test":"TestSame"}',
            '{"Action":"fail","Package":"example/second","Test":"TestSame"}',
        ]
    )
    assert parse_windows_go_json_status(log) == {
        "example/first::TestSame": "pass",
        "example/second::TestSame": "fail",
    }


def test_wrapped_go_stream_fails_closed_instead_of_emitting_fragment_name():
    log = '{"Action":"pass","Test":"TestWrapped"\n'
    parser_status = {'TestWrapped (0.00s)\\n"}': "pass"}

    assert parse_windows_go_json_status(log) == {}
    assert normalize_windows_validation_status(
        "custom parser",
        log,
        parser_status,
        "windows",
    ) == {}


def test_windows_non_go_log_keeps_instance_parser_result():
    parser_status = {"TestFromCustomParser": "pass"}
    assert normalize_windows_validation_status(
        "custom parser",
        "--- PASS: TestFromCustomParser (0.00s)",
        parser_status,
        "windows",
    ) == parser_status


def test_linux_keeps_instance_parser_result_even_when_go_json_is_present():
    parser_status = {"fragment": "pass"}
    log = '{"Action":"pass","Test":"TestStable"}'
    assert normalize_windows_validation_status(
        "custom parser",
        log,
        parser_status,
        "linux",
    ) == parser_status


def test_detects_observed_json_suffix_artifact_without_rejecting_normal_name():
    assert is_corrupted_windows_test_name('TestVarzHandler (0.00s)\\n"}')
    assert not is_corrupted_windows_test_name("TestVarzHandler")