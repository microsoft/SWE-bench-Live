import unittest

from .evaluation import default_pytest_parser, parse_log_pytest


class PytestIdentifierParserTests(unittest.TestCase):
    def test_parameterized_node_ids_keep_spaces(self):
        log = "\n".join(
            [
                "PASSED tests/test_values.py::test_value[LINESTRING (0 0)]",
                "FAILED tests/test_values.py::test_value[NOT A WKT STRING] - assertion failed",
                "SKIPPED [1] tests/test_values.py::test_value[POLYGON ((0 0, 0 0))] - optional dependency",
            ]
        )

        parsed = parse_log_pytest(log, None)

        self.assertEqual(
            parsed,
            {
                "tests/test_values.py::test_value[LINESTRING (0 0)]": "PASSED",
                "tests/test_values.py::test_value[NOT A WKT STRING]": "FAILED",
                "tests/test_values.py::test_value[POLYGON ((0 0, 0 0))]": "SKIPPED",
            },
        )

    def test_skipped_summary_without_node_id_is_ignored(self):
        self.assertEqual(
            parse_log_pytest(
                "SKIPPED [1] tests/test_values.py:42: optional dependency", None
            ),
            {},
        )

    def test_default_parser_normalizes_statuses_without_truncating_ids(self):
        log = "\n".join(
            [
                "PASSED tests/test_values.py::test_value[Valid value with spaces]",
                "XFAIL tests/test_values.py::test_value[expected failure value] - known issue",
                "XPASS tests/test_values.py::test_value[unexpected pass value]",
            ]
        )

        parsed = default_pytest_parser(log)

        self.assertEqual(
            parsed,
            {
                "tests/test_values.py::test_value[Valid value with spaces]": "pass",
                "tests/test_values.py::test_value[expected failure value]": "pass",
                "tests/test_values.py::test_value[unexpected pass value]": "pass",
            },
        )


if __name__ == "__main__":
    unittest.main()