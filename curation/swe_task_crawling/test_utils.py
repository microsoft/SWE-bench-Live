import unittest
from types import SimpleNamespace
from unittest.mock import patch

from curation.swe_task_crawling.utils import extract_patches, is_test_artifact_path


RUSTLS_2586_STYLE_PATCH = """\
diff --git a/rustls/benches/benchmarks.rs b/rustls/benches/benchmarks.rs
index 1111111..2222222 100644
--- a/rustls/benches/benchmarks.rs
+++ b/rustls/benches/benchmarks.rs
@@ -1 +1 @@
-use rustls_test::FailsReads;
+use rustls_test::TestNonBlockIo;
diff --git a/rustls/src/conn.rs b/rustls/src/conn.rs
index 3333333..4444444 100644
--- a/rustls/src/conn.rs
+++ b/rustls/src/conn.rs
@@ -1 +1 @@
-fn complete_io() {}
+fn complete_io() { return; }
diff --git a/rustls/tests/api.rs b/rustls/tests/api.rs
index 5555555..6666666 100644
--- a/rustls/tests/api.rs
+++ b/rustls/tests/api.rs
@@ -1 +1 @@
-fn existing_test() {}
+fn client_complete_io_with_nonblocking_io() {}
"""


class ExtractPatchesTests(unittest.TestCase):
    def test_recognizes_existing_test_markers_and_benchmark_paths(self):
        self.assertTrue(is_test_artifact_path("rustls/tests/api.rs"))
        self.assertTrue(is_test_artifact_path("rustls/benches/benchmarks.rs"))
        self.assertTrue(is_test_artifact_path("project/benchmark/main.go"))
        self.assertTrue(is_test_artifact_path("project\\benches\\main.rs"))
        self.assertFalse(is_test_artifact_path("src/benchmarking.py"))
        self.assertFalse(is_test_artifact_path("rustls/src/conn.rs"))

    @patch("curation.swe_task_crawling.utils.wrapped_requests_get")
    def test_keeps_benchmark_support_with_test_patch(self, request_get):
        request_get.return_value = SimpleNamespace(text=RUSTLS_2586_STYLE_PATCH)

        fix_patch, test_patch = extract_patches({"diff_url": "https://example.invalid/pr.diff"}, None)

        self.assertIn("rustls/src/conn.rs", fix_patch)
        self.assertNotIn("rustls/benches/benchmarks.rs", fix_patch)
        self.assertIn("rustls/benches/benchmarks.rs", test_patch)
        self.assertIn("TestNonBlockIo", test_patch)
        self.assertIn("rustls/tests/api.rs", test_patch)


if __name__ == "__main__":
    unittest.main()