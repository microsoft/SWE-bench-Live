from .windows_tap_capture import assert_complete_numbered_tap_capture


def test_accepts_complete_single_plan_tap_capture():
    log = "# suite\nok 1 - first\nnot ok 2 - second\n1..2\n"

    assert assert_complete_numbered_tap_capture(log) is True


def test_rejects_missing_assertion_id_even_when_plan_is_present():
    log = "ok 1 - first\nok 3 - third\n1..3\n"

    try:
        assert_complete_numbered_tap_capture(log)
    except ValueError as error:
        assert "missing=[2]" in str(error)
    else:
        raise AssertionError("incomplete TAP capture was accepted")


def test_rejects_duplicate_assertion_id():
    log = "ok 1 - first\nok 1 - duplicate\n1..1\n"

    try:
        assert_complete_numbered_tap_capture(log)
    except ValueError as error:
        assert "duplicates=[1]" in str(error)
    else:
        raise AssertionError("duplicate TAP assertion was accepted")


def test_ignores_non_tap_and_multiple_plan_logs():
    assert assert_complete_numbered_tap_capture("ordinary test log\n") is False
    assert assert_complete_numbered_tap_capture("ok 1 - nested\n1..1\nok 1 - outer\n1..1\n") is False