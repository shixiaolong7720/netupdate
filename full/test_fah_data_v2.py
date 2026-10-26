"""Synthetic feature/DataLoader and hard-example sampling contract tests."""
import contextlib
import csv
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fah_data_v2 import EpisodeBatchSampler, FeatureDataset, TrainingPoolDataset


_ABSENT = object()


class FeatureAndSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def feature(self, root, cid, name="example", *, duration=_ABSENT, frames=8, folder=None):
        directory = Path(root) / (folder or f"ID_{cid}_samples")
        directory.mkdir(parents=True, exist_ok=True)
        values = {
            "mfcc": np.full((20, frames), float(cid), dtype=np.float32),
            "logmel": np.zeros((40, frames), dtype=np.float32),
            "mfcc_gate": np.zeros((60, frames), dtype=np.float32),
            "class_id": np.array(cid),
        }
        if duration is not _ABSENT:
            values["duration_sec"] = np.asarray(duration)
        path = directory / f"{name}.npz"
        np.savez(path, **values)
        return path

    def dataset(self, root, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return FeatureDataset(root, *args, **kwargs)

    def make_sampling_dataset(self, count=32):
        root = self.root / "train"
        for cid in (0, 1, 2, 21, 25):
            for index in range(count):
                self.feature(root, cid, f"{index:03d}", duration=1.5)
        return self.dataset(root)

    def test_duration_npz_default_and_manifest_priority(self):
        root = self.root / "durations"
        self.feature(root, 0, "a", duration=1.2)
        self.feature(root, 21, "b")
        dataset = self.dataset(root, default_duration_sec=1.7)
        self.assertEqual(dataset.duration_source_counts, {"manifest": 0, "npz": 1, "default": 1})
        np.testing.assert_allclose(dataset.durations_sec, [1.2, 1.7])
        manifest = self.root / "durations.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sid", "duration_sec"])
            writer.writerows(zip(dataset.sids, [2.1, 2.2]))
        overridden = self.dataset(root, default_duration_sec=1.7, duration_manifest=manifest)
        np.testing.assert_allclose(overridden.durations_sec, [2.1, 2.2])
        self.assertEqual(overridden.duration_source_counts["manifest"], 2)

    def test_manifest_missing_entry_is_an_error(self):
        root = self.root / "missing_manifest"
        self.feature(root, 0)
        manifest = self.root / "incomplete.csv"
        manifest.write_text("sid,duration_sec\nnot_a_sample.npz,1.5\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no entry"):
            self.dataset(root, duration_manifest=manifest)

    def test_noise_id25_is_excluded_but_cw_is_loaded(self):
        root = self.root / "noise"
        self.feature(root, 0)
        self.feature(root, 25, "cw", folder="ID_25_CW")
        self.feature(root, 25, "noise", folder="ID_25_noise")
        dataset = self.dataset(root)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(set(dataset.class_ids), {0, 25})
        self.assertIn("ID_25_CW/cw.npz", dataset.sids)
        self.assertFalse(any("noise" in sid for sid in dataset.sids))

    def test_real_tensor_dataloader_and_global_indices(self):
        roots = [self.root / "train", self.root / "adapt"]
        for root in roots:
            self.feature(root, 0, "same_name", duration=1.5)
            self.feature(root, 21, "same_name", duration=1.5)
        pool = TrainingPoolDataset([self.dataset(root) for root in roots])
        self.assertEqual(len(set(pool.sids)), 4)
        self.assertEqual(pool.frames, 8)
        visited = []
        loader = DataLoader(pool, batch_size=3, shuffle=False, num_workers=0)
        for mfcc, logmel, gate, labels, cids, sids, indices in loader:
            self.assertEqual(mfcc.shape[1:], (1, 20, 8))
            self.assertEqual(logmel.shape[1:], (1, 40, 8))
            self.assertEqual(gate.shape[1:], (1, 60, 8))
            self.assertEqual(mfcc.dtype, torch.float32)
            torch.testing.assert_close(labels, (cids == 0).float())
            for sid, index in zip(sids, indices.tolist()):
                self.assertEqual(sid, pool.sids[index])
                visited.append(index)
        self.assertEqual(visited, list(range(len(pool))))

    def test_bad_feature_shape_and_nan_fail_on_load(self):
        root = self.root / "invalid_shape"
        self.feature(root, 0, "a", frames=8)
        self.feature(root, 0, "b", frames=9)
        dataset = self.dataset(root)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            dataset[1]
        root = self.root / "invalid_finite"
        path = self.feature(root, 0)
        with np.load(path) as data:
            values = {key: data[key].copy() for key in data.files}
        values["logmel"][0, 0] = np.nan
        np.savez(path, **values)
        dataset = self.dataset(root)
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            dataset[0]

    def test_invalid_durations_are_rejected(self):
        root = self.root / "invalid_duration"
        self.feature(root, 0, duration=0)
        with self.assertRaisesRegex(ValueError, "duration_sec"):
            self.dataset(root)
        with self.assertRaisesRegex(ValueError, "duration_sec"):
            self.dataset(root, default_duration_sec=-1)

    def test_warmup_episodes_and_background_id25(self):
        dataset = self.make_sampling_dataset(count=12)
        sampler = EpisodeBatchSampler(dataset, n_classes=3, m_utterances=4,
                                      batches_per_epoch=6, warmup_epochs=1)
        for batch in sampler:
            ids, counts = np.unique(dataset.class_ids[batch], return_counts=True)
            np.testing.assert_array_equal(ids, [0, 1, 2])
            np.testing.assert_array_equal(counts, [4, 4, 4])
            self.assertEqual(len(set(batch)), 12)
        sampler.set_epoch(2)
        for step, batch in enumerate(sampler):
            cids = dataset.class_ids[batch]
            self.assertEqual(len(batch), 12)
            self.assertEqual(int(np.sum(cids == 0)), 4)
            self.assertEqual(len(set(batch)), 12)
            if step % 2:
                self.assertEqual(set(cids), {0, 21, 25})
            else:
                self.assertEqual(set(cids), {0, 1, 2})

    def test_partial_mining_uses_only_observed_scores(self):
        dataset = self.make_sampling_dataset(count=12)
        sampler = EpisodeBatchSampler(dataset, n_classes=3, m_utterances=4,
                                      batches_per_epoch=4, warmup_epochs=0)
        scores = np.full(len(dataset), np.nan)
        scores[::2] = np.linspace(0.05, 0.95, len(scores[::2]))
        gates = np.full(len(dataset), np.nan)
        gates[::2] = 0.9
        summary = sampler.update_mining(scores, [.6, .8, np.nextafter(1., np.inf)], gates)
        self.assertEqual(summary["observed_count"], len(scores[::2]))
        self.assertEqual(summary["operating_thresholds"], [.6, .8])
        self.assertTrue(np.all(np.isfinite(scores[sampler.hard_positive_indices])))
        for queue in sampler.hard_negative_indices.values():
            self.assertTrue(np.all(np.isfinite(scores[queue])))
        self.assertIn(25, sampler.hard_negative_indices)
        scores[0] = np.inf
        with self.assertRaisesRegex(ValueError, "infinity"):
            sampler.update_mining(scores, [.6])

    def test_hard_wake_quota_preserves_random_wake_coverage(self):
        dataset = self.make_sampling_dataset(count=32)
        sampler = EpisodeBatchSampler(dataset, n_classes=3, m_utterances=12,
                                      batches_per_epoch=64, warmup_epochs=0,
                                      hard_positive_fraction=.25, hard_negative_fraction=.25)
        scores = np.linspace(.1, .95, len(dataset))
        sampler.update_mining(scores, [.5, .8])
        hard_wake = set(sampler.hard_positive_indices.tolist())
        observed_wake = set()
        for batch in sampler:
            wake = [index for index in batch if dataset.class_ids[index] == 0]
            self.assertEqual(len(wake), 12)
            self.assertEqual(len(set(wake)), 12)
            self.assertGreaterEqual(len(hard_wake.intersection(wake)), 3)
            observed_wake.update(wake)
        self.assertEqual(observed_wake, set(np.flatnonzero(dataset.class_ids == 0)))

    def test_repeatability_and_unobserved_random_fallback(self):
        dataset = self.make_sampling_dataset(count=12)
        sampler = EpisodeBatchSampler(dataset, n_classes=3, m_utterances=4,
                                      batches_per_epoch=5, seed=123, warmup_epochs=0)
        sampler.update_mining(np.full(len(dataset), np.nan), [])
        self.assertEqual(len(sampler.hard_positive_indices), 0)
        self.assertEqual(len(sampler.hard_negative_indices), 0)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(2)
        self.assertNotEqual(first, list(sampler))

    def test_training_roots_cannot_be_repeated(self):
        root = self.root / "duplicate_root"
        self.feature(root, 0)
        dataset = self.dataset(root)
        with self.assertRaisesRegex(ValueError, "same feature root"):
            TrainingPoolDataset([dataset, self.dataset(root)])

    def test_split_separation_detects_path_aliases_and_copies(self):
        from train_mymodel_fah_v2 import assert_split_separation

        train_root, val_root = self.root / "train", self.root / "val"
        path = self.feature(train_root, 0)
        train = self.dataset(train_root)
        with self.assertRaisesRegex(ValueError, "Same NPZ"):
            assert_split_separation({"train": train, "val": self.dataset(train_root)})
        val_path = val_root / path.parent.name / path.name
        val_path.parent.mkdir(parents=True)
        shutil.copy2(path, val_path)
        val = self.dataset(val_root)
        with contextlib.redirect_stdout(io.StringIO()):
            assert_split_separation({"train": train, "val": val})
        with self.assertRaisesRegex(ValueError, "Identical NPZ"):
            assert_split_separation({"train": train, "val": val}, verify_content=True)


if __name__ == "__main__":
    unittest.main()
