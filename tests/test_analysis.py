"""Scientific accounting checks independent of GPU and model dependencies."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from lookagain.analysis import analyze_records, write_report


def record(example, action, correct, elapsed=1.0, memory=2.0, image=None, **extra):
    return {
        "example_id": example, "image_id": image or example,
        "action": action, "correct": correct, "elapsed_s": elapsed,
        "peak_memory_gib": memory, "status": "ok", **extra,
    }


class AnalysisTests(unittest.TestCase):
    def test_followup_cost_includes_direct_but_highres_is_standalone(self):
        rows = [
            record("a", "direct", False, elapsed=10, memory=7),
            record("a", "highres", True, elapsed=15, memory=12),
            record("a", "crop_tl", True, elapsed=3, memory=4),
        ]
        result = analyze_records(rows, expected_actions=("direct", "highres", "crop_tl"))
        crop = result["actions"]["crop_tl"]
        self.assertEqual(crop["total_latency_s"]["mean"], 13)
        self.assertEqual(crop["policy_peak_memory_gib"]["max"], 7)
        self.assertEqual(result["actions"]["highres"]["total_latency_s"]["mean"], 15)
        self.assertIsNone(crop["delta_ci_95_pp"])

    def test_oracle_stops_when_correct_and_excludes_highres(self):
        actions = ("direct", "highres", "think", "crop_tl")
        outcomes = {
            "a": (True, True, False, True),
            "b": (False, True, True, True),
            "c": (False, True, False, False),
        }
        rows = [record(example, action, correct, elapsed={"direct": 10, "highres": 1, "think": 2, "crop_tl": 3}[action])
                for example, correctness in outcomes.items() for action, correct in zip(actions, correctness)]
        result = analyze_records(rows, bootstrap_samples=100, expected_actions=actions)
        oracle = result["baselines"]["hindsight_oracle"]
        self.assertEqual(oracle["accuracy"], 2 / 3)
        self.assertEqual(oracle["selected_action_counts"], {"direct": 2, "think": 1})
        self.assertEqual(oracle["right_to_wrong_count"], 0)
        self.assertEqual(oracle["wrong_to_right_count"], 1)
        self.assertEqual(oracle["total_latency_s"]["mean"], 32 / 3)
        self.assertEqual(result["actions"]["think"]["wrong_to_right_count"], 1)
        self.assertEqual(result["actions"]["think"]["right_to_wrong_count"], 1)
        self.assertEqual(result["actions"]["think"]["delta_accuracy_pp"], 0)

    def test_missing_failed_duplicate_and_invalid_examples_excluded_from_all_methods(self):
        rows = [record("complete", "direct", False), record("complete", "think", True),
                record("missing", "direct", True),
                record("failed", "direct", True), record("failed", "think", False, status="oom"),
                record("duplicate", "direct", True), record("duplicate", "think", True), record("duplicate", "think", False),
                record("invalid", "direct", True), record("invalid", "think", True, elapsed=math.nan)]
        result = analyze_records(rows, expected_actions=("direct", "think"))
        self.assertEqual(result["status"], "incomplete_coverage")
        self.assertEqual(result["coverage"]["complete_examples"], 1)
        self.assertEqual(result["coverage"]["excluded_examples"], 4)
        self.assertEqual(result["actions"]["direct"]["accuracy"], 0)
        self.assertEqual(result["actions"]["think"]["accuracy"], 1)
        self.assertEqual(result["coverage"]["per_action"]["think"], {
            "ok_records": 3, "failed_records": 1, "invalid_records": 1,
            "missing_examples": 1, "duplicate_examples": 1,
        })

    def test_default_actions_not_inferred_from_surviving_records(self):
        result = analyze_records([record("a", "direct", True)])
        self.assertEqual(result["status"], "no_complete_groups")
        self.assertEqual(result["actions"], {})
        self.assertIn("highres", result["coverage"]["excluded_groups"][0]["problems"])

    def test_cluster_bootstrap_keeps_questions_on_same_image_together(self):
        # Four gains share one image; one harm belongs to another. Resampling
        # two images can only yield -100, +60, or +100 pp, never question-level
        # combinations. Their probabilities are 1/4, 1/2, and 1/4, so the
        # percentile interval must reach both extreme cluster outcomes.
        rows = []
        for index in range(4):
            rows.extend([record(f"gain{index}", "direct", False, image="image_a"),
                         record(f"gain{index}", "think", True, image="image_a")])
        rows.extend([record("harm", "direct", True, image="image_b"),
                     record("harm", "think", False, image="image_b")])
        samples, seed = 500, 27
        result = analyze_records(rows, bootstrap_samples=samples, seed=seed, expected_actions=("direct", "think"))
        self.assertAlmostEqual(result["actions"]["think"]["delta_accuracy_pp"], 60)
        self.assertEqual(result["actions"]["think"]["delta_ci_95_pp"], [-100.0, 100.0])
        self.assertEqual(result["actions"]["direct"]["delta_ci_95_pp"], [0.0, 0.0])

    def test_random_baseline_and_intervals_are_record_order_invariant(self):
        actions = ("direct", "crop_tl", "crop_tr")
        rows = [record(str(i), action, correct=(i + j) % 2 == 0)
                for i in range(10) for j, action in enumerate(actions)]
        first = analyze_records(rows, bootstrap_samples=100, expected_actions=actions)
        second = analyze_records(list(reversed(rows)), bootstrap_samples=100, expected_actions=actions)
        self.assertEqual(first, second)
        self.assertEqual(sum(first["baselines"]["uniform_random_crop"]["selected_action_counts"].values()), 10)

    def test_inconsistent_image_ids_and_nonboolean_correct_do_not_leak_into_metrics(self):
        rows = [record("a", "direct", True, image="first"), record("a", "think", True, image="second"),
                record("b", "direct", True), record("b", "think", 1)]
        result = analyze_records(rows, expected_actions=("direct", "think"))
        self.assertEqual(result["coverage"]["excluded_examples"], 2)
        self.assertIn("image_id", result["coverage"]["excluded_groups"][0]["problems"])
        self.assertEqual(result["actions"], {})

    def test_missing_identity_and_invalid_configuration_raise(self):
        with self.assertRaises(ValueError):
            analyze_records([{"action": "direct"}])
        with self.assertRaises(ValueError):
            analyze_records([], expected_actions=("think",))
        with self.assertRaises(ValueError):
            analyze_records([], bootstrap_samples=0)

    def test_empty_report_contains_coverage_and_no_fabricated_metrics(self):
        result = analyze_records([])
        # Exercise report formatting without relying on platform-specific
        # temporary-directory ACLs or creating files during a unit test.
        with patch.object(Path, "mkdir"), patch.object(Path, "write_text") as write:
            write_report(result, Path("report.md"))
            contents = write.call_args.args[0]
        self.assertIn("0/0 examples", contents)
        self.assertIn("No complete paired groups", contents)
        self.assertNotIn("| Method |", contents)


if __name__ == "__main__":
    unittest.main()
