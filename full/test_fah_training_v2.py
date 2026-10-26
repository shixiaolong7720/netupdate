"""CPU integration test of training, checkpoint restoration, and frozen testing.

Run with the PyTorch interpreter used for training. Synthetic random features
exercise execution and persistence only; these checks make no accuracy claim.
"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for inference tests")
class InferenceBoundaryTests(unittest.TestCase):
    def test_checkpoint_restore_preserves_centroid_without_renormalizing(self):
        import torch
        from train_mymodel_fah_v2 import restore

        model = torch.nn.Linear(2, 3)
        centroid = torch.tensor([0.57735, 0.57735, 0.57735])
        self.assertNotEqual(centroid.norm().item(), 1.0)
        with tempfile.TemporaryDirectory(prefix="fah_v2_centroid_") as temporary:
            path = Path(temporary) / "model.pt"
            torch.save({"kws_model": model.state_dict(), "gate_model": None,
                        "wake_centroid": centroid, "operating_points": {}}, path)
            _, restored, _ = restore(path, model, torch.device("cpu"))
            self.assertTrue(torch.equal(restored, centroid))

    def test_saved_float64_threshold_does_not_round_down_to_float32_score(self):
        import torch
        from train_mymodel_fah_v2 import predict_batch

        class Encoder(torch.nn.Module):
            def forward(self, mfcc, logmel):
                return torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        dummy = torch.zeros(1, 1, 1, 1)
        prediction = predict_batch(Encoder(), None, dummy, dummy, dummy, torch.tensor([1.0, 0.0]),
                                   {"gate_threshold": 0.0, "kws_threshold": float(np.nextafter(1.0, np.inf))})
        self.assertFalse(prediction.item())

    def test_gate_is_still_enforced_when_kws_threshold_is_zero(self):
        import torch
        from train_mymodel_fah_v2 import predict_batch

        class Encoder(torch.nn.Module):
            def forward(self, mfcc, logmel):
                return torch.tensor([[1.0, 0.0]])

        class Gate(torch.nn.Module):
            def forward(self, features):
                return torch.zeros(1, 1)  # sigmoid == 0.5 exactly

        dummy = torch.zeros(1, 1, 1, 1)
        prediction = predict_batch(Encoder(), Gate(), dummy, dummy, dummy, torch.tensor([1.0, 0.0]),
                                   {"gate_threshold": float(np.nextafter(0.5, np.inf)), "kws_threshold": 0.0})
        self.assertFalse(prediction.item())


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for the CPU integration test")
class TrainingWorkflowTests(unittest.TestCase):
    def test_synthetic_training_restores_and_evaluates_without_overwriting(self):
        import torch

        script = Path(__file__).with_name("train_mymodel_fah_v2.py").resolve()
        with tempfile.TemporaryDirectory(prefix="fah_v2_integration_") as temporary:
            root = Path(temporary)
            train_output, evaluation_output = root / "train", root / "evaluation"
            trained = subprocess.run(
                [sys.executable, str(script), "--smoke-test", "--output", str(train_output)],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(trained.returncode, 0, msg=trained.stdout + trained.stderr)

            stage1 = torch.load(train_output / "stage1_best.pt", map_location="cpu", weights_only=True)
            self.assertTrue(stage1["stage"].startswith("stage1/"))
            self.assertNotEqual(stage1["stage"], "stage1/initial")
            self.assertGreater(int(stage1["stage"].split("/")[-1]), 0)
            history = [json.loads(line) for line in (train_output / "history.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
            stages = {entry["stage"] for entry in history}
            self.assertTrue({"stage1", "gate", "cl_01", "gate_cl_01"}.issubset(stages))

            for filename in ("best_system.pt", "operating_points.json", "test_frozen_metrics.json",
                             "best_ge2e_kws.pth", "best_tiny_gate.pth", "best_ge2e_kws_tiny_gate_system.pth",
                             "export_manifest.json", "data_roles.json",
                             "val_scores.npz", "test_scores.npz"):
                self.assertGreater((train_output / filename).stat().st_size, 0)
            expected = json.loads((train_output / "test_frozen_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(expected["threshold_source"], "frozen_validation_thresholds")
            self.assertEqual(expected["mode"], "training_sample_far_proxy")
            self.assertNotIn("fa_per_hour", expected["real"])
            self.assertEqual(set(expected["shared"]["group_metrics"]), {"0", "1", "2", "21", "22", "23", "24"})
            roles = json.loads((train_output / "data_roles.json").read_text(encoding="utf-8"))
            self.assertFalse(roles["post_quantization_board_test"]["loaded_by_this_script"])
            with np.load(train_output / "test_scores.npz") as scores:
                self.assertNotIn(25, scores["cids"])
            checkpoint = torch.load(train_output / "best_system.pt", map_location="cpu", weights_only=True)
            self.assertIsNotNone(checkpoint["gate_model"])
            self.assertEqual(checkpoint["quantization_status"], "pending_external_pipeline")
            self.assertEqual(checkpoint["kws_threshold"], checkpoint["operating_points"]["shared"]["kws_threshold"])

            command = [
                sys.executable, str(script), "--evaluate-checkpoint", str(train_output / "best_system.pt"),
                "--test-dir", str(train_output / "synthetic_features" / "test"),
                "--device", "cpu", "--threads", "2", "--output", str(evaluation_output),
            ]
            evaluated = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(evaluated.returncode, 0, msg=evaluated.stdout + evaluated.stderr)
            metric_path = evaluation_output / "test_frozen_metrics.json"
            saved_bytes = metric_path.read_bytes()
            self.assertEqual(json.loads(saved_bytes.decode("utf-8")), expected)

            overwrite_attempt = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(overwrite_attempt.returncode, 0)
            self.assertIn("Output directory is not empty", overwrite_attempt.stderr)
            self.assertEqual(metric_path.read_bytes(), saved_bytes)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for training configuration tests")
class NormalTrainingConfigurationTests(unittest.TestCase):
    def test_no_argument_defaults_use_normal_three_splits_and_original_folder_labels(self):
        from train_mymodel_fah_v2 import parse_args
        args = parse_args([])
        root = r"D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake"
        self.assertEqual(args.train_dir, root + r"\train")
        self.assertEqual(args.val_dir, root + r"\val")
        self.assertEqual(args.test_dir, root + r"\test")
        self.assertEqual(args.label_policy, "folder")
        self.assertFalse(args.skip_test)
        self.assertFalse(hasattr(args, "adapt_dir"))

    def test_external_board_dataset_is_not_accepted_in_any_training_role(self):
        import contextlib
        import io
        from train_mymodel_fah_v2 import parse_args
        for split in ("train", "val", "test"):
            with self.subTest(split=split), contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as caught:
                    parse_args([f"--{split}-dir", r"D:\data\FUKE_FAH\test"])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("post-quantization", error.getvalue())

    def test_ordinary_background_loss_thresholds_include_mask_and_tts(self):
        from train_mymodel_fah_v2 import thresholds_by_cid
        points = {"real": {"kws_threshold": .6}, "real_cw": {"kws_threshold": .8},
                  "shared": {"kws_threshold": .9}}
        thresholds = thresholds_by_cid(points)
        self.assertTrue(all(thresholds[cid] == .6 for cid in (21, 22, 23, 24)))
        self.assertTrue(all(thresholds[cid] == .8 for cid in (*range(1, 21), 25)))


if __name__ == "__main__":
    unittest.main()
