# -*- coding: utf-8 -*-
"""Train the lite2 KWS encoder for real@1FA/h and real+CW@20FA/h.

This is an independent training entry point. It never imports/runs the legacy
training script, modifies source audio, or mines validation/test examples.
FA/h here follows an independent-clip protocol, not continuous-stream events.
See README_FAH_V2.md for data layout, commands and limitations.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from fah_data_v2 import FeatureDataset, TrainingPoolDataset, EpisodeBatchSampler
from fah_metrics_v2 import calibrate_operating_points, evaluate_operating_points
from fah_model_v2 import (
    DualStream_DSCNN_TCN_AttPool_GE2E, TinyMFCCGate, GE2ELoss,
    BackgroundMetricLoss, dual_spec_augment, gate_spec_augment,
    pack_episode_embeddings, cosine_to_unit_score, operating_point_loss, embedding_collapse_stats,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    root = r"D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake"
    p.add_argument("--train-dir", default=root + r"\train")
    p.add_argument("--val-dir", default=root + r"\val", help="Independent development/calibration set")
    p.add_argument("--adapt-dir", help="Optional independent TRAINING adaptation pool; never val/test")
    p.add_argument("--test-dir", help="Optional locked test root, e.g. FUKE_FAH with ID_* children")
    p.add_argument("--allow-test-cw-substitute", action="store_true",
                   help="Explicitly allow ID_1..20 instead of human ID_25 CW in a test report")
    for split in ("train", "val", "adapt", "test"):
        p.add_argument(f"--{split}-duration-manifest", help="CSV with sid,duration_sec; all examples required")
    p.add_argument("--sample-duration-sec", type=float, default=1.5,
                   help="Fallback duration of each independent clip if no manifest/NPZ metadata")
    p.add_argument("--label-policy", choices=("strict", "folder"), default="strict",
                   help="strict audits metadata before training; folder explicitly trusts verified ID folders and reports stale NPZ labels")
    p.add_argument("--hours-json", help='Optional JSON {"val":{"real":hours,"real_cw":hours},"test":{...}}')
    p.add_argument("--verify-content-disjoint", action="store_true", help="Additionally SHA256-check NPZ duplicates across splits")
    p.add_argument("--output", help="New/empty output directory; never overwrites an existing run")
    p.add_argument("--init-checkpoint", help="Warm-start old/new model weights; optimizer is restarted")
    p.add_argument("--evaluate-checkpoint", help="Evaluate v2 checkpoint on --test-dir at saved thresholds; no training")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--workers", type=int, default=0, help="0 is safe on Windows")
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=140)
    p.add_argument("--steps-per-epoch", type=int, default=800)
    p.add_argument("--n-classes", type=int, default=10)
    p.add_argument("--m-utterances", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--background-ramp-epochs", type=int, default=10)
    p.add_argument("--background-weight", type=float, default=0.1)
    p.add_argument("--positive-weight", type=float, default=0.2)
    p.add_argument("--negative-weight", type=float, default=0.05)
    p.add_argument("--ranking-weight", type=float, default=0.05, help="Set 0 for a sampling-only loss ablation")
    p.add_argument("--hard-positive-fraction", type=float, default=0.25)
    p.add_argument("--hard-negative-fraction", type=float, default=0.25)
    p.add_argument("--mining-every", type=int, default=5)
    p.add_argument("--mining-per-class", type=int, default=512, help="Max scored candidates per class per refresh; 0=all")
    p.add_argument("--enrollment-samples", type=int, default=2048)
    p.add_argument("--gate-candidates", type=int, default=64, help="Gate threshold search budget, includes bypass")
    p.add_argument("--disable-gate", action="store_true")
    p.add_argument("--gate-epochs", type=int, default=40)
    p.add_argument("--gate-steps", type=int, default=64)
    p.add_argument("--gate-batch-size", type=int, default=128)
    p.add_argument("--gate-lr", type=float, default=1e-3)
    p.add_argument("--gate-patience", type=int, default=8)
    p.add_argument("--cl-rounds", type=int, default=1)
    p.add_argument("--cl-epochs", type=int, default=8)
    p.add_argument("--cl-steps", type=int, default=300)
    p.add_argument("--cl-lr", type=float, default=8e-6)
    p.add_argument("--cl-patience", type=int, default=3)
    p.add_argument("--stability-weight", type=float, default=0.15)
    p.add_argument("--gate-cl-epochs", type=int, default=2,
                   help="Optional environment-gate adaptation; same FAH acceptance and rollback")
    p.add_argument("--cl-frr-tolerance", type=float, default=0.0,
                   help="Allowed absolute FRR regression per target vs incumbent; default none")
    p.add_argument("--smoke-test", action="store_true", help="Generate synthetic NPZs and exercise all training stages on CPU")
    args = p.parse_args(argv)
    positive = ("threads", "eval_batch_size", "steps_per_epoch", "n_classes", "m_utterances",
                "patience", "background_ramp_epochs", "mining_every", "enrollment_samples",
                "gate_candidates", "gate_steps", "gate_batch_size", "gate_patience", "cl_steps", "cl_patience")
    for key in positive:
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("workers", "epochs", "warmup_epochs", "gate_epochs", "cl_rounds", "cl_epochs",
                "gate_cl_epochs", "mining_per_class", "cl_frr_tolerance", "weight_decay",
                "background_weight", "positive_weight", "negative_weight", "ranking_weight", "stability_weight"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            p.error(f"--{key.replace('_', '-')} must be finite and nonnegative")
    for key in ("lr", "gate_lr", "cl_lr", "sample_duration_sec"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be finite and positive")
    if args.n_classes < 2 or args.m_utterances < 2:
        p.error("GE2E requires --n-classes >= 2 and --m-utterances >= 2")
    if args.gate_candidates < 2:
        p.error("--gate-candidates must be at least 2")
    if args.base_ch < 2 or args.embedding_dim < 2 or not 0 <= args.dropout < 1:
        p.error("Require --base-ch >= 2, --embedding-dim >= 2, and dropout in [0,1)")
    for key in ("hard_positive_fraction", "hard_negative_fraction"):
        if not np.isfinite(getattr(args, key)) or not 0 <= getattr(args, key) <= 1:
            p.error(f"--{key.replace('_', '-')} must be in [0,1]")
    if args.epochs == 0 and not (args.init_checkpoint or args.evaluate_checkpoint or args.smoke_test):
        p.error("--epochs 0 requires --init-checkpoint")
    if args.evaluate_checkpoint and not args.test_dir:
        p.error("--evaluate-checkpoint requires --test-dir")
    if args.init_checkpoint and args.evaluate_checkpoint:
        p.error("Choose training warm-start or checkpoint evaluation")
    return args


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def setup_runtime(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if name == "auto":
        name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check your PyTorch GPU installation")
    return torch.device(name)


def make_loader(dataset, args, *, indices=None, batch_sampler=None):
    kwargs = dict(num_workers=args.workers, pin_memory=args.device == "cuda")
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return DataLoader(dataset if indices is None else Subset(dataset, list(indices)),
                      batch_size=args.eval_batch_size, shuffle=False, **kwargs)


@torch.no_grad()
def build_centroid(model, enrollment_loader, device):
    model.eval()
    total = None
    count = 0
    for mfcc, mel, _, _, cids, _, _ in enrollment_loader:
        if not torch.all(cids == 0):
            raise ValueError("Enrollment must contain training-only ID_0 examples")
        z = model(mfcc.to(device), mel.to(device))
        total = z.sum(0) if total is None else total + z.sum(0)
        count += len(z)
    if count == 0 or not torch.isfinite(total).all() or total.norm() < 1e-8:
        raise RuntimeError("Invalid or empty wake enrollment embeddings")
    return F.normalize(total / count, dim=0).detach()


@torch.no_grad()
def collect_scores(model, gate, dataset, centroid, args, device, indices=None):
    model.eval()
    if gate is not None:
        gate.eval()
    scores = np.full(len(dataset), np.nan, dtype=np.float64)
    gate_probs = np.full(len(dataset), np.nan, dtype=np.float64)
    for mfcc, mel, gate_features, _, _, _, index in make_loader(dataset, args, indices=indices):
        z = model(mfcc.to(device), mel.to(device))
        unit = cosine_to_unit_score(z @ centroid)
        probability = (torch.sigmoid(gate(gate_features.to(device)).squeeze(1))
                       if gate is not None else torch.ones_like(unit))
        scores[index.numpy()] = unit.cpu().numpy()
        gate_probs[index.numpy()] = probability.cpu().numpy()
    return {"scores": scores, "gate_probs": gate_probs}


def calibrate(outputs, dataset, args, hours, gate):
    return calibrate_operating_points(outputs["scores"], outputs["gate_probs"], dataset.class_ids,
                                     dataset.durations_sec, gate_candidates=args.gate_candidates,
                                     negative_hours_override=hours, use_gate=gate is not None)


def objective(points):
    return 0.5 * (points["real"]["frr"] + points["real_cw"]["frr"])


def is_improved(candidate, incumbent, tolerance=None):
    if objective(candidate) >= objective(incumbent) - 1e-12:
        return False
    return tolerance is None or all(candidate[p]["frr"] <= incumbent[p]["frr"] + tolerance + 1e-12
                                    for p in ("real", "real_cw"))


def print_points(tag, points):
    for protocol in ("real", "real_cw"):
        m = points[protocol]
        print(f"[{tag}/{protocol}] FRR={100*m['frr']:.3f}% | FA/h={m['fa_per_hour']:.4f} "
              f"({m['fp']}/{m['negative_hours']:.4f}h) | gate={m['gate_threshold']:.6f} "
              f"kws={m['kws_threshold']:.6f} | feasible={m['feasible']}", flush=True)


def append_history(out, record):
    with (out / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def save_checkpoint(path, model, gate, centroid, points, args, stage, model_config):
    checkpoint = {
        "version": "fah_v2", "model_config": model_config,
        "kws_model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "gate_model": ({k: v.detach().cpu() for k, v in gate.state_dict().items()}
                       if gate is not None else None),
        "wake_centroid": centroid.detach().cpu(), "operating_points": points,
        "stage": stage, "args": vars(args),
        "decision": "(gate_probability >= gate_threshold) AND (unit_cosine >= kws_threshold)",
        "measurement": "independent clips, at most one activation per clip; not streaming FA/h",
        "deployment_operating_point": "shared",
    }
    temp = Path(str(path) + ".tmp")
    torch.save(checkpoint, temp)
    temp.replace(path)


def load_checkpoint(path, device):
    # New checkpoints only contain tensors and ordinary Python primitives.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a dictionary checkpoint")
    return checkpoint


def restore(path, model, device):
    checkpoint = load_checkpoint(path, device)
    model.load_state_dict(checkpoint["kws_model"], strict=True)
    gate = None
    if checkpoint.get("gate_model") is not None:
        gate = TinyMFCCGate().to(device)
        gate.load_state_dict(checkpoint["gate_model"], strict=True)
    # Preserve the enrolled reference bit-for-bit. Renormalizing a saved float32
    # centroid can move scores by an ULP across an exact nextafter threshold.
    centroid = checkpoint["wake_centroid"].to(device)
    if (centroid.ndim != 1 or not torch.isfinite(centroid).all()
            or not torch.isclose(centroid.norm(), centroid.new_tensor(1.0), atol=1e-4)):
        raise ValueError("Checkpoint wake centroid must be a finite normalized vector")
    return gate, centroid, checkpoint["operating_points"]


@torch.no_grad()
def predict_batch(model, gate, mfcc, logmel, mfcc_gate, centroid, operating_point):
    """Return bool decisions with EXACTLY the saved threshold semantics.

    Use points['shared'] for one deployable threshold pair. Inputs and models
    must already be on the same device. Convert predictions to float64 before
    comparison so a nextafter float64 threshold cannot round down to a tied
    float32 score. This API evaluates clips; streaming event handling is external.
    """
    model.eval()
    score = cosine_to_unit_score(model(mfcc, logmel) @ centroid).double()
    if gate is None:
        probability = torch.ones_like(score)
    else:
        gate.eval()
        probability = torch.sigmoid(gate(mfcc_gate).squeeze(1)).double()
    return ((probability >= float(operating_point["gate_threshold"]))
            & (score >= float(operating_point["kws_threshold"])))


def thresholds_by_cid(points):
    result = {cid: points["shared"]["kws_threshold"] for cid in range(26)}
    result[21] = points["real"]["kws_threshold"]
    for cid in (*range(1, 21), 25):
        result[cid] = points["real_cw"]["kws_threshold"]
    result.pop(0)
    return result


def refresh_mining(model, gate, pool, sampler, centroid, points, args, device, epoch):
    rng = np.random.default_rng(args.seed + 7919 * epoch)
    selected = []
    for cid in np.unique(pool.class_ids):
        candidates = np.flatnonzero(pool.class_ids == cid)
        if args.mining_per_class and len(candidates) > args.mining_per_class:
            candidates = rng.choice(candidates, args.mining_per_class, replace=False)
        selected.extend(candidates.tolist())
    scored = collect_scores(model, gate, pool, centroid, args, device, selected)
    sampler.update_mining(scored["scores"], [points[p]["kws_threshold"] for p in ("real", "real_cw")],
                          gate_probs=scored["gate_probs"])
    append_history(Path(args.output), {"stage": "mining", "epoch": epoch, "summary": sampler.last_mining_summary})
    print(f"[mining] scored {len(selected)}/{len(pool)} training/adaptation examples; no val/test mining", flush=True)


def train_kws(model, gate, pool, val, enrollment, points, args, device, out, hours,
              model_config, *, stage="stage1", continual=False):
    epochs = args.cl_epochs if continual else args.epochs
    steps = args.cl_steps if continual else args.steps_per_epoch
    patience = args.cl_patience if continual else args.patience
    warmup = 0 if continual else args.warmup_epochs
    sampler = EpisodeBatchSampler(pool, n_classes=args.n_classes, m_utterances=args.m_utterances,
                                  batches_per_epoch=steps, seed=args.seed + (10000 if continual else 0),
                                  warmup_epochs=warmup, hard_positive_fraction=args.hard_positive_fraction,
                                  hard_negative_fraction=args.hard_negative_fraction)
    loader = make_loader(pool, args, batch_sampler=sampler)
    metric_loss = GE2ELoss().to(device)
    background_loss = BackgroundMetricLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.cl_lr if continual else args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
    teacher = copy.deepcopy(model).eval() if continual else None
    if teacher is not None:
        teacher.requires_grad_(False)
    centroid = build_centroid(model, enrollment, device)
    checkpoint_path = out / f"{stage}_best.pt"
    best_points = points
    # A random cold-start model is not an eligible training result. Warm starts
    # and CL retain their input checkpoint so unsuccessful adaptation can roll back.
    has_best = continual or bool(args.init_checkpoint)
    if has_best:
        save_checkpoint(checkpoint_path, model, gate, centroid, best_points, args, stage + "/initial", model_config)
    stale = 0
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        if epoch > warmup and (epoch == warmup + 1 or (epoch - warmup - 1) % args.mining_every == 0):
            refresh_mining(model, gate, pool, sampler, centroid, best_points, args, device, epoch)
        reference = centroid.detach()
        model.train()
        sums = {k: 0.0 for k in ("loss", "positive", "negative", "ranking", "stability")}
        episode_acc, episode_margin, episode_count = 0.0, 0.0, 0
        collapse_cos, collapse_std = 0.0, 0.0
        ramp = 1.0 if continual else min(1.0, max(0.0, (epoch - warmup) / args.background_ramp_epochs))
        for mfcc, mel, _, _, cids, _, _ in loader:
            mfcc, mel = dual_spec_augment(mfcc.to(device), mel.to(device))
            cids = cids.to(device)
            optimizer.zero_grad(set_to_none=True)
            z = model(mfcc, mel)
            nonwake = cids[cids != 0]
            confusable = bool(((nonwake >= 1) & (nonwake <= 20)).all().item())
            if confusable:
                episode, _ = pack_episode_embeddings(z, cids, args.n_classes, args.m_utterances)
                base, _, diagnostic = metric_loss(episode)
                episode_acc += float(diagnostic["episode_acc"].detach())
                episode_margin += float(diagnostic["episode_margin"].detach())
                geometry = embedding_collapse_stats(z.detach())
                collapse_cos += geometry["pairwise_cos_mean"]
                collapse_std += geometry["feature_std"]
                episode_count += 1
            else:
                # Detach reference: negative rejection cannot move the positive centroid itself.
                base, _ = background_loss(z[cids != 0], reference)
                base = args.background_weight * ramp * base
            components = operating_point_loss(z, cids, reference, thresholds_by_cid(best_points))
            stability = z.sum() * 0.0
            if teacher is not None:
                with torch.no_grad():
                    teacher_z = teacher(mfcc, mel)
                stability = (1.0 - F.cosine_similarity(z, teacher_z, dim=1)).mean()
            loss = (base + ramp * (args.positive_weight * components["positive"]
                                  + args.negative_weight * components["negative"]
                                  + args.ranking_weight * components["ranking"])
                    + args.stability_weight * stability)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{stage}: non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
            optimizer.step()
            sums["loss"] += float(loss.detach())
            sums["stability"] += float(stability.detach())
            for key in ("positive", "negative", "ranking"):
                sums[key] += float(components[key].detach())
        scheduler.step()
        centroid = build_centroid(model, enrollment, device)
        outputs = collect_scores(model, gate, val, centroid, args, device)
        current = calibrate(outputs, val, args, hours, gate)
        record = {"stage": stage, "epoch": epoch, "losses": {k: v / steps for k, v in sums.items()},
                  "episode_accuracy": episode_acc / max(1, episode_count),
                  "episode_margin": episode_margin / max(1, episode_count),
                  "embedding_pairwise_cos": collapse_cos / max(1, episode_count),
                  "embedding_feature_std": collapse_std / max(1, episode_count), "metrics": current}
        append_history(out, record)
        print(f"[{stage}] epoch={epoch} loss={sums['loss']/steps:.5f} "
              f"episode_acc={record['episode_accuracy']:.3f}", flush=True)
        print_points(stage, current)
        if (epoch >= max(1, warmup) and episode_count and record["embedding_pairwise_cos"] > 0.9999
                and record["embedding_feature_std"] < 1e-4):
            raise RuntimeError("Embedding collapse detected; inspect class semantics and features before continuing")
        eligible = has_best or epoch >= min(epochs, max(1, warmup))
        improved = eligible and (not has_best or is_improved(
            current, best_points, args.cl_frr_tolerance if continual else None))
        if improved:
            best_points = current
            save_checkpoint(checkpoint_path, model, gate, centroid, best_points, args, f"{stage}/{epoch}", model_config)
            has_best = True
            stale = 0
        else:
            stale += 1
        # Give the GE2E-only warmup and background ramp a chance to finish.
        if stale >= patience and epoch > warmup + (0 if continual else args.background_ramp_epochs):
            break
    if not has_best:
        raise RuntimeError(f"{stage}: no trained checkpoint was eligible for selection")
    gate, centroid, best_points = restore(checkpoint_path, model, device)
    return model, gate, centroid, best_points


def train_gate(model, gate, pool, val, centroid, points, args, device, out, hours, model_config,
               *, stage="gate", continual=False):
    epochs = args.gate_cl_epochs if continual else args.gate_epochs
    if args.disable_gate or epochs == 0:
        return gate, points
    ids = pool.class_ids
    if not np.any(ids == 23) or not np.any(ids != 23):
        print(f"[{stage}] no ID_23 environment training pool; retain existing gate/bypass", flush=True)
        return gate, points
    path = out / f"{stage}_best.pt"
    save_checkpoint(path, model, gate, centroid, points, args, stage + "/initial", model_config)
    gate = TinyMFCCGate().to(device) if gate is None else gate
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.gate_lr * (0.2 if continual else 1.0),
                                  weight_decay=args.weight_decay)
    weights = np.zeros(len(pool), dtype=np.float64)
    other_ids = [cid for cid in np.unique(ids) if cid != 23]
    weights[ids == 23] = 0.5 / np.sum(ids == 23)
    for cid in other_ids:
        weights[ids == cid] = 0.5 / len(other_ids) / np.sum(ids == cid)
    best = points
    stale = 0
    for epoch in range(1, epochs + 1):
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        args.gate_steps * args.gate_batch_size, replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed + 30000 + epoch))
        loader = DataLoader(pool, batch_size=args.gate_batch_size, sampler=sampler,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
        gate.train()
        total = 0.0
        for _, _, features, _, cids, _, _ in loader:
            features = gate_spec_augment(features.to(device))
            target = (cids != 23).float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = gate(features).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite gate loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 3.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
        outputs = collect_scores(model, gate, val, centroid, args, device)
        current = calibrate(outputs, val, args, hours, gate)
        append_history(out, {"stage": stage, "epoch": epoch, "loss": total / args.gate_steps, "metrics": current})
        print_points(stage, current)
        # Initial KWS-only solution is also a candidate: the gate must earn its place.
        if is_improved(current, best, args.cl_frr_tolerance):
            best = current
            save_checkpoint(path, model, gate, centroid, best, args, f"{stage}/{epoch}", model_config)
            stale = 0
        else:
            stale += 1
            if stale >= args.gate_patience:
                break
    loaded_gate, _, best = restore(path, model, device)
    return loaded_gate, best


def save_outputs(out, split, outputs, dataset):
    np.savez_compressed(out / f"{split}_scores.npz", scores=outputs["scores"], gate_probs=outputs["gate_probs"],
                        cids=dataset.class_ids, durations_sec=dataset.durations_sec,
                        sids=np.asarray(dataset.sids, dtype=str))
    with (out / f"{split}_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(("sid", "cid", "duration_sec", "kws_score", "gate_probability"))
        writer.writerows(zip(dataset.sids, dataset.class_ids, dataset.durations_sec,
                             outputs["scores"], outputs["gate_probs"]))


def assert_split_separation(datasets, verify_content=False):
    active = [(name, ds) for name, ds in datasets.items() if ds is not None]
    owners = {}
    hashes = {}
    for name, ds in active:
        for path in ds.paths:
            resolved = str(Path(path).resolve())
            if resolved in owners and owners[resolved] != name:
                raise ValueError(f"Same NPZ appears in {owners[resolved]} and {name}: {path}")
            owners[resolved] = name
            if verify_content:
                digest = hashlib.sha256()
                with open(path, "rb") as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(block)
                key = digest.hexdigest()
                if key in hashes and hashes[key][0] != name:
                    raise ValueError(f"Identical NPZ contents across {hashes[key][0]} and {name}: {path}")
                hashes[key] = (name, path)
    print("Split paths checked. Speaker/source-recording separation must also hold in your dataset.", flush=True)


def create_smoke_data(args, out):
    rng = np.random.default_rng(args.seed)
    for split in ("train", "val", "adapt", "test"):
        root = out / "synthetic_features" / split
        for cid in (0, 1, 2, 21, 22, 23, 24, 25):
            folder = root / f"ID_{cid}_{'CW' if cid == 25 else 'synthetic'}"
            folder.mkdir(parents=True)
            for i in range(4):
                # Finite shape-correct features, not a performance benchmark.
                np.savez_compressed(folder / f"{split}_{cid}_{i}.npz",
                                    mfcc=rng.normal(size=(20, 32)).astype(np.float32),
                                    logmel=rng.normal(size=(40, 32)).astype(np.float32),
                                    mfcc_gate=rng.normal(size=(60, 32)).astype(np.float32), duration_sec=1.5)
        setattr(args, split + "_dir", str(root))
    args.device, args.workers, args.threads = "cpu", 0, 2
    args.base_ch, args.embedding_dim, args.dropout = 4, 16, 0.0
    args.epochs, args.steps_per_epoch, args.warmup_epochs = 2, 2, 0
    args.background_ramp_epochs = 1
    args.n_classes, args.m_utterances = 3, 2
    args.eval_batch_size, args.enrollment_samples = 16, 4
    args.gate_epochs, args.gate_steps, args.gate_batch_size = 1, 1, 8
    args.cl_rounds, args.cl_epochs, args.cl_steps, args.gate_cl_epochs = 1, 1, 2, 1
    args.gate_candidates, args.mining_per_class = 8, 4
    print("SYNTHETIC SMOKE TEST: correctness/execution only, no accuracy or FA/h generalization claim.", flush=True)


def load_hours(path):
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or any(k not in ("val", "test") for k in value):
        raise ValueError("hours JSON may contain only val and test dictionaries")
    for split, protocols in value.items():
        if not isinstance(protocols, dict):
            raise ValueError(f"{split} hours must be an object")
        for protocol, hours in protocols.items():
            if protocol not in ("real", "real_cw") or not np.isfinite(hours) or hours <= 0:
                raise ValueError(f"Invalid negative hours: {split}/{protocol}")
    return value


def evaluate_checkpoint(args, out, device, hours):
    checkpoint = load_checkpoint(args.evaluate_checkpoint, device)
    if checkpoint.get("version") != "fah_v2":
        raise ValueError("Frozen-threshold evaluation requires a v2 checkpoint")
    model = DualStream_DSCNN_TCN_AttPool_GE2E(**checkpoint["model_config"]).to(device)
    gate, centroid, points = restore(args.evaluate_checkpoint, model, device)
    if args.disable_gate and gate is not None:
        raise ValueError("Cannot change gate configuration during frozen-checkpoint evaluation")
    dataset = FeatureDataset(args.test_dir, args.sample_duration_sec, args.test_duration_manifest,
                             label_policy=args.label_policy, label_audit_path=out / "test_label_conflicts.csv")
    check_test_protocol(dataset, args)
    outputs = collect_scores(model, gate, dataset, centroid, args, device)
    metrics = evaluate_operating_points(outputs["scores"], outputs["gate_probs"], dataset.class_ids,
                                       dataset.durations_sec, points, negative_hours_override=hours.get("test"))
    save_outputs(out, "test", outputs, dataset)
    write_json(out / "test_frozen_metrics.json", metrics)
    print_points("test-frozen", metrics)


def check_test_protocol(dataset, args):
    if not np.any(dataset.class_ids == 25) and not args.allow_test_cw_substitute:
        raise ValueError("Human CW test requires ID_25_CW. For an explicitly labelled ID_1..20 "
                         "surrogate report, pass --allow-test-cw-substitute")


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output) if args.output else (Path(tempfile.mkdtemp(prefix="fah_v2_smoke_")) if args.smoke_test
                                               else Path("runs") / datetime.now().strftime("fah_v2_%Y%m%d_%H%M%S_%f"))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}; choose a new run directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        if args.init_checkpoint or args.evaluate_checkpoint:
            raise ValueError("Smoke test uses synthetic data and fresh model, not real checkpoints")
        create_smoke_data(args, out)
    # Keep args JSON/checkpoint values ordinary primitives and make the resolved
    # run path available for mining-history records, including auto-named runs.
    args.output = str(out.resolve())
    device = setup_runtime(args)
    args.device = device.type
    hours = load_hours(args.hours_json)
    write_json(out / "config.json", vars(args))
    write_json(out / "runtime.json", {"torch": str(torch.__version__), "numpy": np.__version__,
                                     "device": str(device), "synthetic": args.smoke_test})
    print(f"Output: {out.resolve()} | device={device}", flush=True)
    if args.evaluate_checkpoint:
        evaluate_checkpoint(args, out, device, hours)
        return
    datasets = {}
    for split in ("train", "val", "adapt", "test"):
        root = getattr(args, split + "_dir")
        datasets[split] = (FeatureDataset(root, args.sample_duration_sec,
                                         getattr(args, split + "_duration_manifest"), label_policy=args.label_policy,
                                         label_audit_path=out / f"{split}_label_conflicts.csv") if root else None)
    assert_split_separation(datasets, args.verify_content_disjoint)
    if datasets["test"] is not None:
        check_test_protocol(datasets["test"], args)
    train, val = datasets["train"], datasets["val"]
    write_json(out / "data_summary.json", {
        name: {"root": str(getattr(args, name + "_dir")), "samples": len(ds),
               "class_counts": {str(int(cid)): int(np.sum(ds.class_ids == cid)) for cid in np.unique(ds.class_ids)},
               "duration_sec_total": float(ds.durations_sec.sum()), "frames": ds.frames,
               "duration_source_counts": ds.duration_source_counts, "label_audit": ds.label_audit}
        for name, ds in datasets.items() if ds is not None})
    config = {"base_ch": args.base_ch, "embedding_dim": args.embedding_dim, "dropout": args.dropout}
    initial = load_checkpoint(args.init_checkpoint, device) if args.init_checkpoint else None
    if initial and "model_config" in initial:
        config = initial["model_config"]
    model = DualStream_DSCNN_TCN_AttPool_GE2E(**config).to(device)
    gate = None
    if initial:
        state = initial.get("kws_model", initial.get("model"))
        if state is None:
            raise ValueError("Warm-start checkpoint has neither kws_model nor model weights")
        model.load_state_dict(state, strict=True)
        gate_state = initial.get("gate_model", initial.get("gate"))
        if gate_state is not None and not args.disable_gate:
            gate = TinyMFCCGate().to(device)
            gate.load_state_dict(gate_state, strict=True)
    wake_indices = np.flatnonzero(train.class_ids == 0)
    rng = np.random.default_rng(args.seed + 2026)
    selected = rng.choice(wake_indices, min(args.enrollment_samples, len(wake_indices)), replace=False)
    enrollment = make_loader(train, args, indices=selected)
    centroid = build_centroid(model, enrollment, device)
    points = calibrate(collect_scores(model, gate, val, centroid, args, device), val, args, hours.get("val"), gate)
    print_points("initial-dev", points)
    model, gate, centroid, points = train_kws(model, gate, train, val, enrollment, points, args, device, out,
                                               hours.get("val"), config)
    gate, points = train_gate(model, gate, train, val, centroid, points, args, device, out, hours.get("val"), config)
    # Adaptation is explicit. Never turn validation or test into training data.
    pool = TrainingPoolDataset([train, datasets["adapt"]]) if datasets["adapt"] is not None else train
    for round_id in range(1, args.cl_rounds + 1):
        model, gate, centroid, points = train_kws(
            model, gate, pool, val, enrollment, points, args, device, out, hours.get("val"), config,
            stage=f"cl_{round_id:02d}", continual=True)
        gate, points = train_gate(model, gate, pool, val, centroid, points, args, device, out,
                                 hours.get("val"), config, stage=f"gate_cl_{round_id:02d}", continual=True)
    save_checkpoint(out / "best_system.pt", model, gate, centroid, points, args, "final", config)
    write_json(out / "operating_points.json", points)
    outputs = collect_scores(model, gate, val, centroid, args, device)
    save_outputs(out, "val", outputs, val)
    write_json(out / "val_calibrated_metrics.json", points)
    if datasets["test"] is not None:
        test = datasets["test"]
        outputs = collect_scores(model, gate, test, centroid, args, device)
        metrics = evaluate_operating_points(outputs["scores"], outputs["gate_probs"], test.class_ids,
                                           test.durations_sec, points, negative_hours_override=hours.get("test"))
        save_outputs(out, "test", outputs, test)
        write_json(out / "test_frozen_metrics.json", metrics)
        print_points("test-frozen", metrics)
    else:
        print("No --test-dir supplied: saved development metrics only; blind test not evaluated.", flush=True)
    print(f"Completed. Checkpoint: {out.resolve() / 'best_system.pt'}", flush=True)


if __name__ == "__main__":
    main()
