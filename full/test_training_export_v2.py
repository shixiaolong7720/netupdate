"""FP32 export key compatibility and metadata tests; no quantization claim."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from fah_model_v2 import DualStream_DSCNN_TCN_AttPool_GE2E, TinyMFCCGate
from training_export_v2 import export_legacy_checkpoints


class TrainingExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="training_export_v2_")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.config = {"base_ch": 4, "embedding_dim": 3, "dropout": 0.0,
                       "tcn_layers": 1, "tcn_dilations": (1,), "att_hidden": 8}
        self.model = DualStream_DSCNN_TCN_AttPool_GE2E(**self.config)
        self.gate = TinyMFCCGate()
        self.centroid = torch.tensor([0.57735, 0.57735, 0.57735])
        self.points = {"shared": {"gate_threshold": 0.125, "kws_threshold": 0.876543210123}}

    def export(self, gate=True):
        return export_legacy_checkpoints(self.output, self.model, self.gate if gate else None,
                                         self.centroid, self.points, self.config)

    def load(self, path):
        return torch.load(path, map_location="cpu", weights_only=True)

    def test_old_model_gate_and_system_keys_load_strictly(self):
        paths = self.export()
        kws, gate, system = (self.load(paths[key]) for key in ("kws", "gate", "system"))
        self.assertIn("model", kws)
        self.assertIn("gate", gate)
        self.assertTrue({"kws_model", "gate_model", "wake_centroid", "gate_threshold", "kws_threshold"}.issubset(system))
        fresh_model = DualStream_DSCNN_TCN_AttPool_GE2E(**self.config)
        fresh_model.load_state_dict(kws["model"], strict=True)
        TinyMFCCGate().load_state_dict(gate["gate"], strict=True)
        fresh_model.load_state_dict(system["kws_model"], strict=True)
        self.assertEqual(system["kws_threshold"], self.points["shared"]["kws_threshold"])
        self.assertEqual(system["gate_threshold"], 0.125)

    def test_centroid_and_source_weights_are_not_modified(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        centroid_before = self.centroid.clone()
        self.assertNotEqual(self.centroid.norm().item(), 1.0)
        system = self.load(self.export()["system"])
        self.assertTrue(torch.equal(system["wake_centroid"], centroid_before))
        self.assertTrue(torch.equal(self.centroid, centroid_before))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_float_weights_export_as_cpu_fp32_without_mutating_module(self):
        self.model.double()
        paths = self.export()
        exported = self.load(paths["kws"])
        self.assertTrue(all(value.device.type == "cpu" and value.dtype == torch.float32
                            for value in exported["model"].values()))
        self.assertTrue(all(value.dtype == torch.float64 for value in self.model.state_dict().values()))

    def test_manifest_explicitly_defers_quantization_and_does_not_dump_args(self):
        from argparse import Namespace
        paths = export_legacy_checkpoints(self.output, self.model, self.gate, self.centroid,
                                          self.points, self.config, args=Namespace(secret="do_not_export"))
        manifest_text = Path(paths["manifest"]).read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        self.assertEqual(manifest["quantization_status"], "pending_external_pipeline")
        self.assertEqual(manifest["inputs"]["mfcc"], ["N", 1, 20, "T"])
        self.assertIn("FUKE_FAH excluded", manifest["calibration_data_allowed"])
        self.assertNotIn("do_not_export", manifest_text)
        self.assertEqual(sorted(path.suffix for path in self.output.iterdir()), [".json", ".pth", ".pth", ".pth"])

    def test_absent_gate_has_no_fabricated_file_and_is_explicit(self):
        self.points["shared"]["gate_threshold"] = 0.0
        paths = self.export(gate=False)
        self.assertIsNone(paths["gate"])
        self.assertFalse((self.output / "best_tiny_gate.pth").exists())
        system = self.load(paths["system"])
        self.assertIsNone(system["gate_model"])
        self.assertFalse(system["gate_present"])
        manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
        self.assertIn("absent Gate", manifest["gate_absent_requirement"])

    def test_invalid_centroid_or_threshold_is_rejected_before_writing(self):
        with self.assertRaisesRegex(ValueError, "bypass"):
            self.export(gate=False)
        self.points["shared"]["kws_threshold"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.export()
        self.points["shared"]["kws_threshold"] = 0.8
        self.centroid = torch.tensor([1.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "normalized"):
            self.export()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_embedding_dimension_must_match_saved_reference(self):
        self.config["embedding_dim"] = 4
        with self.assertRaisesRegex(ValueError, "embedding_dim"):
            self.export()


if __name__ == "__main__":
    unittest.main()
