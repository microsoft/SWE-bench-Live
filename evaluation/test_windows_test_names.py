from .windows_test_names import (
    build_windows_repeated_char_index,
    classify_expected_tests,
    collapse_adjacent_repeated_chars,
    resolve_expected_test_name,
)


def test_collapses_adjacent_console_duplicates():
    assert collapse_adjacent_repeated_chars("github.coom/hashicorp") == "github.com/hashicorp"
    assert collapse_adjacent_repeated_chars("TestLLoad") == "TestLoad"
    assert collapse_adjacent_repeated_chars("deefault") == "default"
    assert collapse_adjacent_repeated_chars("NNumberIntVal") == "NumberIntVal"


def test_windows_fallback_is_unique_and_exact_match_wins():
    status = {
        "github.com/hashicorp/packer/command/TestBuild": "pass",
        "TestLoad": "pass",
        "default": "pass",
        "TestLLoad": "fail",
    }
    index = build_windows_repeated_char_index(status)

    assert resolve_expected_test_name(
        "github.coom/hashicorp/packer/command/TestBuild",
        status,
        "windows",
        index,
    ) == "github.com/hashicorp/packer/command/TestBuild"
    # The exact key is preferred, even though its normalized form collides.
    assert resolve_expected_test_name("TestLLoad", status, "windows", index) == "TestLLoad"
    assert resolve_expected_test_name("deefault", status, "windows", index) == "default"


def test_ambiguous_normalization_fails_closed_and_linux_stays_exact():
    status = {"TestLoad": "pass", "TestLLoad": "pass"}
    index = build_windows_repeated_char_index(status)

    assert "TestLoad" not in index
    assert resolve_expected_test_name("TestLLLoad", status, "windows", index) is None
    assert resolve_expected_test_name("TestLLoad", status, "linux", index) == "TestLLoad"
    assert resolve_expected_test_name("TestLLLoad", status, "linux", index) is None


def test_windows_fallback_decodes_go_unicode_escapes():
    status = {
        "TestImportSubscriptionPartialOverlapWithPrefix/myprefix.>": "pass",
        "TestJSONCompat/should_support_JSON_not_prettified_with_final_bracket_after_newline": "pass",
    }
    assert resolve_expected_test_name(
        r"TestImportSubscriptionPartialOverrlapWithPrefix/myprefix.\u003e",
        status,
        "windows",
    ) == "TestImportSubscriptionPartialOverlapWithPrefix/myprefix.>"


def test_classification_assigns_collision_group_by_unique_minimum_cost():
    status = {
        "TestGetStashEntries/Severall_stash_entries_found": "pass",
        "TestGetStashEntries/Several_stassh_entries_found": "pass",
    }
    expected = [
        "TestGetStashEntries/Several_staash_entries_found",
        "TestGetStashEntries/Severaal_stash_entries_found",
    ]
    assert classify_expected_tests(expected, status, "windows") == {
        "success": expected,
        "failure": [],
    }


def test_classification_preserves_expected_names_and_statuses():
    status = {
        "github.com/hashicorp/packer/command/TestBuild": "pass",
        "TestLoad": "fail",
        "default": "skip",
    }
    assert classify_expected_tests(
        [
            "github.coom/hashicorp/packer/command/TestBuild",
            "TestLLoad",
            "deefault",
            "missing",
        ],
        status,
        "windows",
    ) == {
        "success": ["github.coom/hashicorp/packer/command/TestBuild"],
        "failure": ["TestLLoad"],
    }