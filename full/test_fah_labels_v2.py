"""Regression checks for stale NPZ metadata and explicit folder authority."""
import contextlib
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fah_data_v2 import FeatureDataset
from fah_labels_v2 import inspect_label_metadata


class LabelAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fah_v2_labels_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def feature(self, cid, stored, name="sample", label=None):
        path = self.root / "features" / f"ID_{cid}_Hard" / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {"mfcc": np.zeros((20, 8), np.float32), "logmel": np.zeros((40, 8), np.float32),
                  "mfcc_gate": np.zeros((60, 8), np.float32), "class_id": np.asarray(stored)}
        if label is not None:
            values["label"] = np.asarray(label)
        np.savez(path, **values)
        return path

    def load(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return FeatureDataset(self.root / "features", **kwargs)

    def test_strict_default_reports_all_conflicts_before_iteration(self):
        self.feature(20, 19, "a", label=0)
        self.feature(20, 18, "b", label=0)
        report = self.root / "audit.csv"
        with self.assertRaisesRegex(ValueError, "2 conflicting NPZ files"):
            self.load(label_audit_path=report)
        with report.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["stored_value"] for row in rows}, {"19", "18"})
        self.assertTrue(all(row["folder_expected"] == "20" for row in rows))

    def test_explicit_folder_policy_preserves_data_and_uses_folder_labels(self):
        path = self.feature(20, 19, label=0)
        before = hashlib.sha256(path.read_bytes()).digest()
        dataset = self.load(label_policy="folder", label_audit_path=self.root / "audit.csv")
        self.assertEqual(dataset[0][4], 20)
        self.assertEqual(dataset[0][3].item(), 0)
        self.assertEqual(dataset.label_audit["conflicted_files"], 1)
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).digest())
        with np.load(path) as data:
            self.assertEqual(data["class_id"].item(), 19)

    def test_wake_nonwake_conflicts_are_not_hidden_in_summary(self):
        self.feature(20, 0, label=1)
        dataset = self.load(label_policy="folder")
        self.assertEqual(dataset.label_audit["issue_counts"]["wake_nonwake_conflict"], 2)
        self.assertEqual(dataset[0][4], 20)
        self.assertEqual(dataset[0][3].item(), 0)

    def test_noninteger_metadata_is_not_truncated_into_a_valid_id(self):
        self.feature(20, 20.9)
        with self.assertRaisesRegex(ValueError, "stored class_id=20.9"):
            self.load()

    def test_malformed_multivalue_class_id_is_audited(self):
        self.feature(20, [20, 20])
        with self.assertRaisesRegex(ValueError, "Label audit failed before training"):
            self.load()

    def test_binary_label_mismatch_is_checked_even_when_class_id_matches(self):
        self.feature(0, 0, label=0)
        with self.assertRaisesRegex(ValueError, "stored label=0"):
            self.load()

    def test_missing_optional_metadata_and_valid_float_id_are_supported(self):
        path = self.feature(20, 20.0, label=0.0)
        dataset = self.load()
        self.assertEqual(dataset.label_audit["issue_count"], 0)
        with np.load(path) as data:
            values = {key: data[key] for key in data.files if key not in ("class_id", "label")}
        np.savez(path, **values)
        dataset = self.load()
        self.assertEqual(dataset.label_audit["issue_count"], 0)

    def test_metadata_mutation_after_preflight_remains_an_error(self):
        path = self.feature(20, 20)
        dataset = self.load()
        with np.load(path) as data:
            values = {key: data[key] for key in data.files}
        values["class_id"] = np.asarray(19)
        np.savez(path, **values)
        with self.assertRaisesRegex(ValueError, "changed after the preflight"):
            dataset[0]

    def test_standalone_audit_reports_conflicts_without_modifying_data(self):
        path = self.feature(20, 19)
        before = path.read_bytes()
        report = self.root / "reports"
        command = [sys.executable, str(Path(__file__).with_name("fah_labels_v2.py")),
                   "--root", str(self.root / "features"), "--report-dir", str(report)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        summary = json.loads((report / "label_audit_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary[0]["policy"], "audit_only_no_training")
        self.assertEqual(summary[0]["conflicted_files"], 1)
        self.assertEqual(before, path.read_bytes())

    def test_standalone_audit_clean_exit_and_invalid_policy(self):
        self.feature(20, 20, label=0)
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("fah_labels_v2.py")),
                                 "--root", str(self.root / "features"), "--report-dir", str(self.root / "reports")],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with self.assertRaisesRegex(ValueError, "label_policy"):
            self.load(label_policy="ignore")


if __name__ == "__main__":
    unittest.main()
