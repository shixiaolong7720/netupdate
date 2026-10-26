"""Run with: python -m unittest discover -s full -p 'test_fah_metrics_v2.py'."""

import json
import unittest

import numpy as np

from fah_metrics_v2 import (
    calibrate_operating_points,
    evaluate_operating_points,
    exact_threshold_for_budget,
)


class ExactBudgetTests(unittest.TestCase):
    def test_integer_budget_uses_all_available_activations(self):
        for dtype in (np.float32, np.float64):
            scores = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=dtype)
            for budget in range(5):
                threshold = exact_threshold_for_budget(scores, budget)
                self.assertEqual(int(np.sum(scores.astype(np.float64) >= threshold)), budget)

    def test_ties_cannot_spend_fraction_of_tied_group(self):
        scores = np.asarray([0.9, 0.8, 0.8, 0.4])
        threshold = exact_threshold_for_budget(scores, 2)
        self.assertEqual(int(np.sum(scores >= threshold)), 1)
        self.assertEqual(int(np.sum(scores >= exact_threshold_for_budget(scores, 3))), 3)

    def test_zero_budget_saturated_score_and_empty(self):
        self.assertGreater(exact_threshold_for_budget(np.asarray([1.0]), 0), 1.0)
        self.assertEqual(exact_threshold_for_budget(np.asarray([]), 0), 0.0)
        self.assertEqual(exact_threshold_for_budget(np.asarray([0.0]), 1), 0.0)

    def test_invalid_inputs(self):
        for budget in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                exact_threshold_for_budget(np.asarray([0.5]), budget)
        for scores in ([np.nan], [np.inf], [-0.1], [1.1]):
            with self.assertRaises(ValueError):
                exact_threshold_for_budget(np.asarray(scores), 0)


class OperatingPointTests(unittest.TestCase):
    def setUp(self):
        self.scores = np.asarray([0.9, 0.6, 0.8, 0.4, 0.7, 0.2])
        self.gate = np.asarray([0.9, 0.9, 0.1, 0.8, 0.8, 0.3])
        self.cids = np.asarray([0, 0, 21, 21, 25, 25])
        self.duration = np.asarray([1.5, 1.5, 1800.0, 1800.0, 1800.0, 1800.0])

    def test_both_protocols_and_shared_respect_budgets(self):
        result = calibrate_operating_points(self.scores, self.gate, self.cids, self.duration)
        for p in ("real", "real_cw"):
            self.assertTrue(result[p]["feasible"])
            self.assertTrue(result["shared"][p]["feasible"])
        self.assertEqual(result["real"]["negative_hours"], 1.0)
        self.assertEqual(result["real_cw"]["negative_hours"], 2.0)
        json.dumps(result, allow_nan=False)

    def test_frozen_thresholds_do_not_recalibrate_on_test(self):
        scores = np.asarray([0.8, 0.7, 0.6, 0.5])
        gate = np.ones(4)
        cids = np.asarray([0, 0, 21, 25])
        durations = np.full(4, 1.5)
        dev = calibrate_operating_points(scores, gate, cids, durations, use_gate=False)
        test_scores = np.asarray([0.8, 0.7, 0.95, 0.95])
        test = evaluate_operating_points(test_scores, gate, cids, durations, dev)
        self.assertEqual(test["real"]["kws_threshold"], dev["real"]["kws_threshold"])
        self.assertEqual(test["real"]["fp"], 1)
        self.assertFalse(test["real"]["feasible"])
        self.assertEqual(test["threshold_source"], "frozen_development_thresholds")

    def test_gate_rejected_zero_scores_stay_rejected_at_zero_kws_threshold(self):
        frozen = {
            "real": {"gate_threshold": 0.5, "kws_threshold": 0.0},
            "real_cw": {"gate_threshold": 0.5, "kws_threshold": 0.0},
            "shared": {"gate_threshold": 0.5, "kws_threshold": 0.0},
        }
        result = evaluate_operating_points(
            [0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0, 21, 25], [1.5]*3, frozen,
        )
        self.assertEqual(result["real"]["fp"], 0)
        self.assertEqual(result["real"]["fn"], 1)
        self.assertEqual(result["real"]["gate_rejected_positive"], 1)
        self.assertEqual(result["real"]["kws_rejected_positive"], 0)

    def test_fallback_and_human_cw_protocol_shift_are_explicit(self):
        cids = np.where(self.cids == 25, 1, self.cids)
        dev = calibrate_operating_points(self.scores, self.gate, cids, self.duration)
        self.assertTrue(dev["real_cw"]["uses_confusable_fallback"])
        test = evaluate_operating_points(self.scores, self.gate, self.cids, self.duration, dev)
        self.assertTrue(test["real_cw"]["protocol_shift"])

    def test_missing_protocol_members_and_invalid_durations_fail(self):
        for cids in ([21, 21, 25], [0, 1, 25], [0, 21, 21]):
            with self.assertRaises(ValueError):
                calibrate_operating_points([0.5]*3, [0.5]*3, cids, [1.5]*3)
        with self.assertRaises(ValueError):
            calibrate_operating_points(self.scores, self.gate, self.cids, [0.0]*6)
        with self.assertRaises(ValueError):
            calibrate_operating_points(self.scores, self.gate, self.cids, self.duration,
                                       negative_hours_override={"unknown": 2.0})

    def test_duration_overrides_are_explicit_and_positive(self):
        result = calibrate_operating_points(self.scores, self.gate, self.cids, self.duration,
                                            negative_hours_override={"real": 10.0, "real_cw": 20.0})
        self.assertEqual(result["real"]["negative_hours"], 10.0)
        self.assertEqual(result["real"]["hours_source"], "override")
        with self.assertRaises(ValueError):
            calibrate_operating_points(self.scores, self.gate, self.cids, self.duration,
                                       negative_hours_override={"real": -1.0})

    def test_fractional_budget_never_rounds_up(self):
        scores = np.asarray([0.95, 0.8, 0.7])
        result = calibrate_operating_points(
            scores, np.ones(3), [0, 21, 25], [1.5]*3, use_gate=False,
            negative_hours_override={"real": 0.999999, "real_cw": 0.099999},
        )
        self.assertEqual(result["real"]["fp"], 0)
        self.assertLessEqual(result["real_cw"]["fp"], 1)
        self.assertLessEqual(result["real"]["fa_per_hour"], 1.0)
        self.assertLessEqual(result["real_cw"]["fa_per_hour"], 20.0)

    def test_exhaustive_search_matches_bruteforce_on_random_small_datasets(self):
        rng = np.random.default_rng(42)
        for _ in range(40):
            cids = np.asarray([0]*4 + [21]*4 + [25]*4)
            scores = rng.choice([0.0, 0.2, 0.4, 0.8, 1.0], size=12)
            gates = rng.choice([0.0, 0.3, 0.6, 1.0], size=12)
            duration = np.full(12, float(rng.choice([30.0, 60.0, 450.0, 900.0, 1800.0])))
            real_hours = float(np.sum(duration[cids == 21]) / 3600.0)
            cw_hours = float(np.sum(duration[cids != 0]) / 3600.0)
            result = calibrate_operating_points(scores, gates, cids, duration, gate_candidates=100)
            gate_thresholds = np.unique(np.r_[0.0, gates, np.nextafter(1.0, np.inf)])
            kws_thresholds = np.unique(np.r_[0.0, scores, np.nextafter(np.unique(scores), np.inf)])
            best = {"real": 1.0, "real_cw": 1.0, "shared": 1.0}
            for gt in gate_thresholds:
                for kt in kws_thresholds:
                    pred = (gates >= gt) & (scores >= kt)
                    frr = float(np.mean(~pred[cids == 0]))
                    real_fp = int(np.sum(pred[cids == 21]))
                    cw_fp = int(np.sum(pred[cids != 0]))
                    if real_fp / real_hours <= 1.0:
                        best["real"] = min(best["real"], frr)
                    if cw_fp / cw_hours <= 20.0:
                        best["real_cw"] = min(best["real_cw"], frr)
                    if real_fp / real_hours <= 1.0 and cw_fp / cw_hours <= 20.0:
                        best["shared"] = min(best["shared"], frr)
            self.assertEqual(result["real"]["frr"], best["real"])
            self.assertEqual(result["real_cw"]["frr"], best["real_cw"])
            self.assertEqual(result["shared"]["real"]["frr"], best["shared"])


if __name__ == "__main__":
    unittest.main()
