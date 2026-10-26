"""Small CPU checks for the retained encoder and operating-point losses."""

import unittest

import torch
from torch.nn import functional as F

from fah_model_v2 import (
    DualStream_DSCNN_TCN_AttPool_GE2E,
    GE2ELoss,
    TinyMFCCGate,
    operating_point_loss,
    pack_episode_embeddings,
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_encoder_gate_and_ge2e_backward(self):
        torch.manual_seed(11)
        model = DualStream_DSCNN_TCN_AttPool_GE2E(
            base_ch=4, dropout=0.0, tcn_layers=2, tcn_dilations=(1, 2),
            att_hidden=8, embedding_dim=8,
        )
        gate = TinyMFCCGate(dropout=0.0)
        embeddings = model(torch.randn(6, 1, 20, 25), torch.randn(6, 1, 40, 25))
        self.assertEqual(tuple(embeddings.shape), (6, 8))
        torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(6))
        episode, ids = pack_episode_embeddings(embeddings, torch.tensor([0, 0, 0, 1, 1, 1]), 2, 3)
        self.assertEqual(ids.tolist(), [0, 1])
        loss, _, diagnostics = GE2ELoss()(episode)
        gate_logits = gate(torch.randn(6, 1, 60, 25)).flatten()
        loss = loss + F.binary_cross_entropy_with_logits(gate_logits, torch.tensor([1., 1., 1., 0., 0., 0.]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in (model, gate):
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertIn("episode_acc", diagnostics)

    def test_episode_shape_errors(self):
        with self.assertRaises(RuntimeError):
            pack_episode_embeddings(torch.randn(4, 8), torch.tensor([0, 0, 1, 1]), 2, 3)

    def test_operating_loss_gradient_reduces_loss_and_detaches_centroid(self):
        z = torch.nn.Parameter(torch.tensor([[0.0, 1.0], [0.8, 0.6], [0.7, 0.7], [0.6, 0.8]]))
        centroid = torch.tensor([1.0, 0.0], requires_grad=True)
        cids = torch.tensor([0, 0, 21, 1])
        thresholds = {21: 0.7, 1: 0.65}
        optimizer = torch.optim.SGD([z], lr=0.1)
        before = sum(operating_point_loss(z, cids, centroid, thresholds).values())
        before.backward()
        self.assertIsNone(centroid.grad)
        self.assertTrue(torch.isfinite(z.grad).all())
        optimizer.step()
        after = sum(operating_point_loss(z, cids, centroid, thresholds).values())
        self.assertLess(float(after.detach()), float(before.detach()))

    def test_absent_classes_and_unattainable_thresholds_stay_finite(self):
        z = torch.randn(3, 8, requires_grad=True)
        losses = operating_point_loss(z, torch.zeros(3, dtype=torch.int64), torch.randn(8), [1.00000001])
        self.assertEqual(set(losses), {"positive", "negative", "ranking"})
        self.assertTrue(all(float(v.detach()) == 0.0 for v in losses.values()))
        sum(losses.values()).backward()
        self.assertIsNotNone(z.grad)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_class_threshold_mapping_does_not_supervise_unmapped_negatives(self):
        z = torch.randn(4, 8, requires_grad=True)
        losses = operating_point_loss(z, torch.tensor([0, 0, 23, 23]), torch.randn(8), {21: 0.7})
        self.assertEqual(float(losses["negative"].detach()), 0.0)
        self.assertGreater(float(losses["positive"].detach()), 0.0)
        self.assertGreater(float(losses["ranking"].detach()), 0.0)

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            operating_point_loss(torch.randn(2, 8), torch.tensor([0, 21]), torch.randn(8), [0.7], temperature=0)


if __name__ == "__main__":
    unittest.main()
