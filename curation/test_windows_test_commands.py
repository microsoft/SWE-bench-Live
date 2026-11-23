from .windows_test_commands import (
    normalize_windows_go_test_file_globs,
    normalize_windows_test_commands,
)


def test_normalizes_unquoted_relative_go_file_glob_for_windows():
    command = "go test pkg/integration/clients/*.go -v -json > reports/integration.json"

    assert normalize_windows_go_test_file_globs(command) == (
        "go test ./pkg/integration/clients -v -json > reports/integration.json"
    )


def test_normalizes_only_go_test_segments_and_preserves_other_commands():
    command = (
        "$env:GOFLAGS = '-mod=vendor'; "
        "go test ./pkg/unit/*.go -json; "
        "echo pkg/integration/clients/*.go"
    )

    assert normalize_windows_go_test_file_globs(command) == (
        "$env:GOFLAGS = '-mod=vendor'; "
        "go test ./pkg/unit -json; "
        "echo pkg/integration/clients/*.go"
    )


def test_preserves_quoted_or_absolute_file_globs():
    command = (
        "go test 'pkg/integration/clients/*.go' -json; "
        "go test C:\\repo\\pkg\\integration\\clients\\*.go -json"
    )

    assert normalize_windows_go_test_file_globs(command) == command


def test_normalizes_command_sequence_without_mutating_input():
    commands = [
        "go test pkg/a/*.go -json",
        "go test ./pkg/b -json",
    ]

    assert normalize_windows_test_commands(commands) == [
        "go test ./pkg/a -json",
        "go test ./pkg/b -json",
    ]
    assert commands == [
        "go test pkg/a/*.go -json",
        "go test ./pkg/b -json",
    ]