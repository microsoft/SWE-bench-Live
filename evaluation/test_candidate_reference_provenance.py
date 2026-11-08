import unittest

from . import evaluation


class CandidateReferenceProvenanceTests(unittest.TestCase):
    def test_exact_gold_patch_is_reported_as_review_signal(self):
        instance = {"patch": "diff --git a/a b/a\n+gold\n", "pred_patch": "diff --git a/a b/a\n+gold\n"}
        result = evaluation.validate_candidate_provenance(instance)
        self.assertTrue(result["candidate_equals_gold"])
        self.assertTrue(result["candidate_equals_gold_after_crlf_normalization"])
        self.assertEqual(result["reference_match_kind"], "exact_bytes")
        self.assertEqual(result["provenance_status"], "exact_reference_match_unattributed")
        self.assertTrue(result["review_required"])

    def test_crlf_only_transport_match_is_reported_as_review_signal(self):
        instance = {"patch": "diff --git a/a b/a\n+gold\n", "pred_patch": "diff --git a/a b/a\r\n+gold\r\n"}
        result = evaluation.validate_candidate_provenance(instance)
        self.assertFalse(result["candidate_equals_gold"])
        self.assertTrue(result["candidate_equals_gold_after_crlf_normalization"])
        self.assertEqual(result["reference_match_kind"], "crlf_normalized")
        self.assertTrue(result["review_required"])

    def test_karmada_style_public_upstream_match_is_not_auto_rejected(self):
        # Karmada PR #7365 is the motivating case: a public upstream fix can
        # reproduce the benchmark reference without proving hidden-reference use.
        instance = {"patch": "diff --git a/a b/a\n+gold\n", "pred_patch": "diff --git a/a b/a\n+gold\n"}
        result = evaluation.validate_candidate_provenance(instance)
        self.assertTrue(result["review_required"])
        self.assertEqual(result["provenance_status"], "exact_reference_match_unattributed")

    def test_independent_candidate_is_accepted_with_hashes(self):
        instance = {"patch": "diff --git a/a b/a\n+gold\n", "pred_patch": "diff --git a/a b/a\n+candidate\n"}
        result = evaluation.validate_candidate_provenance(instance)
        self.assertFalse(result["candidate_equals_gold"])
        self.assertFalse(result["candidate_equals_gold_after_crlf_normalization"])
        self.assertEqual(result["provenance_status"], "not_exact_reference_match")
        self.assertFalse(result["review_required"])
        self.assertNotEqual(result["candidate_patch_sha256"], result["gold_patch_sha256"])


if __name__ == "__main__":
    unittest.main()
