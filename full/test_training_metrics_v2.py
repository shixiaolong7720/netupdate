"""Normal train/val/test low-FAR metrics, distinct from post-quantization FAH."""
import json
import math
import unittest

import numpy as np

from training_metrics_v2 import calibrate_training_points, evaluate_training_points


class TrainingProxyTests(unittest.TestCase):
    def test_no_id25_requirement_and_all_ordinary_classes_participate(self):
        ids = np.asarray([0, 0, 1, 20, 21, 22, 23, 24])
        scores = np.asarray([.9, .8, .7, .6, .5, .4, .3, .2])
        result = calibrate_training_points(scores, np.ones(8), ids, use_gate=False)
        self.assertEqual(result["real"]["n_neg"], 4)
        self.assertEqual(result["real_cw"]["n_neg"], 6)
        self.assertEqual(result["real"]["ordinary_negative_ids"], [21, 22, 23, 24])
        self.assertEqual(result["real_cw"]["confusable_ids"], [1, 20])
        self.assertEqual(result["mode"], "training_sample_far_proxy")
        self.assertEqual(result["measurement"], "sample_false_accept_rate")
        self.assertNotIn("fa_per_hour", result["real"])
        self.assertNotIn("negative_hours", result["real"])
        json.dumps(result, allow_nan=False)

    def test_exact_integer_budget_and_ties(self):
        ids = [0, 21, 22, 23, 24]
        result = calibrate_training_points([.95, .9, .8, .7, .6], np.ones(5), ids,
                                           ordinary_far=.25, mixed_far=.25, use_gate=False)
        self.assertEqual(result["real"]["fp"], 1)
        tied = calibrate_training_points([.95, .9, .8, .8, .6], np.ones(5), ids,
                                         ordinary_far=.5, mixed_far=.5, use_gate=False)
        self.assertEqual(tied["real"]["fp"], 1)
        self.assertLessEqual(tied["real"]["far"], .5)

    def test_frozen_thresholds_can_fail_test_budget(self):
        ids = [0, 0, 21, 1]
        validation = calibrate_training_points([.9, .8, .4, .5], np.ones(4), ids, use_gate=False)
        test = evaluate_training_points([.9, .8, .99, 1.0], np.ones(4), ids, validation)
        self.assertEqual(test["real"]["kws_threshold"], validation["real"]["kws_threshold"])
        self.assertEqual(test["real"]["fp"], 1)
        self.assertFalse(test["real"]["feasible"])
        self.assertEqual(test["threshold_source"], "frozen_validation_thresholds")

    def test_ordinary_only_dataset_is_valid_and_clearly_marked(self):
        ids = [0, 0, 22, 23, 24]
        scores = [.9, .8, .1, .2, .3]
        validation = calibrate_training_points(scores, np.ones(5), ids, use_gate=False)
        self.assertFalse(validation["real_cw"]["has_confusable_classes"])
        self.assertIn("no confusable classes", validation["real_cw"]["definition"])
        test = evaluate_training_points(scores, np.ones(5), ids, validation)
        self.assertEqual(test["real_cw"]["frr"], 0)

    def test_gate_rejection_is_not_undone_at_zero_kws_threshold(self):
        points = calibrate_training_points([0, 0, 0], [0, 0, 0], [0, 21, 1], ordinary_far=1, mixed_far=1)
        for p in ("real", "real_cw"):
            points[p]["gate_threshold"], points[p]["kws_threshold"] = .5, 0
        points["shared"]["gate_threshold"], points["shared"]["kws_threshold"] = .5, 0
        result = evaluate_training_points([0, 0, 0], [0, 0, 0], [0, 21, 1], points)
        self.assertEqual(result["real"]["fp"], 0)
        self.assertEqual(result["real"]["gate_rejected_positive"], 1)
        self.assertEqual(result["shared"]["group_metrics"]["21"]["gate_rejected"], 1)

    def test_shared_group_breakdown_and_accuracy_are_complete(self):
        ids = [0, 0, 1, 21, 22, 23, 24, 25, 99]
        points = calibrate_training_points([.9, .8, .7, .6, .5, .4, .3, .2, .1], np.ones(9), ids,
                                           ordinary_far=.4, mixed_far=.5, use_gate=False)
        shared = points["shared"]
        self.assertEqual(set(shared["group_metrics"]), {"0", "1", "21", "22", "23", "24", "25", "99"})
        full = shared["real_cw"]
        self.assertEqual(full["n_pos"] + full["n_neg"], len(ids))
        self.assertEqual(full["tp"] + full["tn"] + full["fp"] + full["fn"], len(ids))
        self.assertEqual(full["accuracy"], (full["tp"] + full["tn"]) / len(ids))

    def test_validation_target_rates_are_reused_on_different_test_composition(self):
        points = calibrate_training_points([.9, .8, .5, .4], np.ones(4), [0, 21, 1, 25],
                                           ordinary_far=.2, mixed_far=.4, use_gate=False)
        test = evaluate_training_points([.9, .3, .4], np.ones(3), [0, 22, 23], points)
        self.assertEqual(test["real"]["target_far"], .2)
        self.assertEqual(test["real_cw"]["target_far"], .4)
        self.assertTrue(test["real_cw"]["protocol_shift"])

    def test_invalid_inputs_and_missing_groups_fail_explicitly(self):
        for ids in ([0, 1, 25], [1, 21, 22]):
            with self.assertRaises(ValueError):
                calibrate_training_points([.5]*3, [1]*3, ids)
        for far in (-.1, 1.1, np.nan, True):
            with self.assertRaises(ValueError):
                calibrate_training_points([.9, .5], [1, 1], [0, 21], ordinary_far=far)
        for scores in ([np.nan, .5], [1.1, .5]):
            with self.assertRaises(ValueError):
                calibrate_training_points(scores, [1, 1], [0, 21])
        with self.assertRaises(ValueError):
            evaluate_training_points([.9, .5], [1, 1], [0, 21], {"measurement": "FA/h"})

    def test_exhaustive_joint_search_matches_bruteforce(self):
        rng = np.random.default_rng(2026)
        ids = np.asarray([0]*3 + [1]*2 + [21, 22, 23, 24])
        ordinary = (ids >= 21)
        mixed = ids != 0
        for _ in range(40):
            scores = rng.choice([0, .2, .6, 1], len(ids))
            gates = rng.choice([0, .3, .8, 1], len(ids))
            ordinary_far, mixed_far = rng.choice([0, .25, .5, 1], size=2)
            result = calibrate_training_points(scores, gates, ids, ordinary_far, mixed_far, gate_candidates=100)
            targets = {"real": ordinary_far, "real_cw": mixed_far}
            masks = {"real": ordinary, "real_cw": mixed}
            best = {"real": 1.0, "real_cw": 1.0, "shared": 1.0}
            for gt in np.unique(np.r_[0, gates, np.nextafter(np.max(gates), np.inf)]):
                for kt in np.unique(np.r_[0, scores, np.nextafter(np.unique(scores), np.inf)]):
                    prediction = (gates >= gt) & (scores >= kt)
                    frr = float(np.mean(~prediction[ids == 0]))
                    feasible = {}
                    for p in targets:
                        fp = int(np.sum(prediction & masks[p]))
                        feasible[p] = fp / int(np.sum(masks[p])) <= targets[p]
                        if feasible[p]:
                            best[p] = min(best[p], frr)
                    if all(feasible.values()):
                        best["shared"] = min(best["shared"], frr)
            self.assertEqual(result["real"]["frr"], best["real"])
            self.assertEqual(result["real_cw"]["frr"], best["real_cw"])
            self.assertEqual(result["shared"]["real"]["frr"], best["shared"])


if __name__ == "__main__":
    unittest.main()
