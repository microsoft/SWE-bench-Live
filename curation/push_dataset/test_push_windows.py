from curation.windows_test_commands import normalize_windows_test_commands


def test_windows_dataset_publisher_normalizes_test_commands():
    assert normalize_windows_test_commands([
        "go test pkg/integration/clients/*.go -v -json",
    ]) == ["go test ./pkg/integration/clients -v -json"]