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
                             "val_scores.npz", "test_scores.npz"):
                self.assertGreater((train_output / filename).stat().st_size, 0)
            expected = json.loads((train_output / "test_frozen_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(expected["threshold_source"], "frozen_development_thresholds")

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


if __name__ == "__main__":
    unittest.main()
