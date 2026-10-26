"""Low-FAR validation proxies for the ordinary float32 training workflow.

Only train/val/test feature splits are needed. Metrics are sample false-accept
rates, not false activations per hour and not quantized/on-board measurements.
The internal keys ``real`` and ``real_cw`` are retained for checkpoint/loss
compatibility: their ordinary/mixed protocols are defined explicitly below.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from fah_metrics_v2 import exact_threshold_for_budget, _gate_thresholds


MODE = "training_sample_far_proxy"
MEASUREMENT = "sample_false_accept_rate"


def _validated_inputs(scores, gate_probs, cids):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    gates = np.asarray(gate_probs, dtype=np.float64).reshape(-1)
    raw_ids = np.asarray(cids).reshape(-1)
    if not scores.size or scores.size != gates.size or scores.size != raw_ids.size:
        raise ValueError("scores, gate_probs and cids must be nonempty arrays of equal length")
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must contain finite unit scores in [0,1]")
    if np.any(~np.isfinite(gates)) or np.any((gates < 0) | (gates > 1)):
        raise ValueError("gate_probs must contain finite probabilities in [0,1]")
    try:
        valid_ids = np.isfinite(raw_ids) & (raw_ids >= 0) & (raw_ids == np.floor(raw_ids))
    except (TypeError, ValueError) as exc:
        raise ValueError("cids must contain nonnegative integer class IDs") from exc
    if not np.all(valid_ids):
        raise ValueError("cids must contain nonnegative integer class IDs")
    return scores, gates, raw_ids.astype(np.int64)


def _target_rate(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite rate in [0,1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be a finite rate in [0,1]")
    return result


def _protocol_specs(ids, ordinary_far, mixed_far):
    positive = ids == 0
    confusable = ((ids >= 1) & (ids <= 20)) | (ids == 25)
    ordinary = (ids != 0) & ~confusable
    if not np.any(positive):
        raise ValueError("Training proxy evaluation requires ID_0 positive examples")
    if not np.any(ordinary):
        raise ValueError("Training proxy evaluation requires ordinary negatives outside ID_1..20/ID_25")
    ordinary_ids = [int(cid) for cid in np.unique(ids[ordinary])]
    confusable_ids = [int(cid) for cid in np.unique(ids[confusable])]
    shared = {"positive": positive, "has_confusable_classes": bool(confusable_ids),
              "ordinary_negative_ids": ordinary_ids, "confusable_ids": confusable_ids}
    return {
        "real": {
            **shared, "negative": ordinary, "target_far": _target_rate(ordinary_far, "ordinary_far"),
            "definition": "ID_0_Wake vs all ordinary non-wake IDs excluding ID_1..20 and ID_25",
        },
        "real_cw": {
            **shared, "negative": ids != 0, "target_far": _target_rate(mixed_far, "mixed_far"),
            "definition": ("ID_0_Wake vs all non-wake IDs including available confusables"
                           if confusable_ids else "ID_0_Wake vs all non-wake IDs; no confusable classes present"),
        },
    }


def _point(scores, gates, spec, gate_threshold, kws_threshold):
    gate_threshold, kws_threshold = float(gate_threshold), float(kws_threshold)
    if not math.isfinite(gate_threshold) or not math.isfinite(kws_threshold):
        raise ValueError("Operating thresholds must be finite")
    gate_pass = gates >= gate_threshold
    prediction = gate_pass & (scores >= kws_threshold)
    positive, negative = spec["positive"], spec["negative"]
    n_pos, n_neg = int(np.sum(positive)), int(np.sum(negative))
    fp = int(np.sum(prediction & negative))
    fn = int(np.sum(~prediction & positive))
    tp, tn = n_pos - fn, n_neg - fp
    far = fp / n_neg
    return {
        "gate_threshold": gate_threshold, "kws_threshold": kws_threshold,
        "target_far": float(spec["target_far"]), "far": float(far), "frr": float(fn / n_pos),
        "fp": fp, "fn": fn, "tp": tp, "tn": tn,
        "n_pos": n_pos, "n_neg": n_neg, "npos": n_pos, "nneg": n_neg,
        "accuracy": float((tp + tn) / (n_pos + n_neg)),
        "gate_rejected_positive": int(np.sum(positive & ~gate_pass)),
        "kws_rejected_positive": int(np.sum(positive & gate_pass & (scores < kws_threshold))),
        "feasible": bool(far <= spec["target_far"]), "definition": spec["definition"],
        "has_confusable_classes": bool(spec["has_confusable_classes"]),
        "ordinary_negative_ids": list(spec["ordinary_negative_ids"]),
        "confusable_ids": list(spec["confusable_ids"]),
    }


def _group_metrics(scores, gates, ids, gate_threshold, kws_threshold):
    gate_pass = gates >= float(gate_threshold)
    prediction = gate_pass & (scores >= float(kws_threshold))
    groups = {}
    for cid in np.unique(ids):
        selected = ids == cid
        n = int(np.sum(selected))
        passed = int(np.sum(selected & prediction))
        rejected = n - passed
        positive = int(cid) == 0
        groups[str(int(cid))] = {
            "cid": int(cid), "n": n, "is_wake": positive,
            "accepted": passed, "rejected": rejected,
            "frr": float(rejected / n) if positive else None,
            "far": None if positive else float(passed / n),
            "gate_rejected": int(np.sum(selected & ~gate_pass)),
            "kws_rejected_after_gate": int(np.sum(selected & gate_pass & (scores < kws_threshold))),
        }
    return groups


def _point_key(point):
    return point["frr"], point["far"], point["gate_threshold"], point["kws_threshold"]


def _shared_point(scores, gates, ids, specs, gate_threshold, kws_threshold):
    return {
        "gate_threshold": float(gate_threshold), "kws_threshold": float(kws_threshold),
        **{p: _point(scores, gates, spec, gate_threshold, kws_threshold) for p, spec in specs.items()},
        "group_metrics": _group_metrics(scores, gates, ids, gate_threshold, kws_threshold),
    }


def calibrate_training_points(scores, gate_probs, cids, ordinary_far=0.0005, mixed_far=0.005,
                              gate_candidates=64, use_gate=True):
    """Calibrate validation-only sample-FAR proxies; no ID25 or duration required.

    For each searched Gate boundary the lowest feasible KWS threshold is exact,
    including tied scores. ``shared`` is a single pair satisfying both validation
    constraints; independent benchmark pairs are retained as ``real``/``real_cw``.
    Gate search is rank-subsampled when distinct scores exceed gate_candidates.
    """
    scores, gates, ids = _validated_inputs(scores, gate_probs, cids)
    specs = _protocol_specs(ids, ordinary_far, mixed_far)
    thresholds = _gate_thresholds(gates, gate_candidates, use_gate)
    best = {p: None for p in specs}
    best_shared = None
    for gate_threshold in thresholds:
        gate_pass = gates >= gate_threshold
        kws_thresholds = {}
        for protocol, spec in specs.items():
            budget = math.floor(spec["target_far"] * int(np.sum(spec["negative"])))
            kws = exact_threshold_for_budget(scores[spec["negative"] & gate_pass], budget)
            point = _point(scores, gates, spec, gate_threshold, kws)
            if not point["feasible"]:
                raise ArithmeticError("Exact threshold exceeded the sample-FAR budget")
            kws_thresholds[protocol] = kws
            if best[protocol] is None or _point_key(point) < _point_key(best[protocol]):
                best[protocol] = point
        shared = _shared_point(scores, gates, ids, specs, gate_threshold, max(kws_thresholds.values()))
        key = (max(shared[p]["frr"] for p in specs), sum(shared[p]["frr"] for p in specs),
               sum(shared[p]["far"] for p in specs), shared["gate_threshold"], shared["kws_threshold"])
        if best_shared is None or key < best_shared[0]:
            best_shared = key, shared
    return {
        **best, "shared": best_shared[1], "mode": MODE, "measurement": MEASUREMENT,
        "threshold_source": "validation_calibration", "gate_thresholds_searched": int(len(thresholds)),
        "use_gate": bool(use_gate),
        "proxy_notice": "Float32 validation sample-FAR proxy; quantized on-board FRR@FA/h is evaluated separately",
    }


def evaluate_training_points(scores, gate_probs, cids, operating_points):
    """Evaluate a normal test split once with frozen validation thresholds."""
    if not isinstance(operating_points, Mapping) or operating_points.get("mode") != MODE:
        raise ValueError("Expected training_sample_far_proxy operating points from validation calibration")
    scores, gates, ids = _validated_inputs(scores, gate_probs, cids)
    specs = _protocol_specs(ids, operating_points["real"]["target_far"], operating_points["real_cw"]["target_far"])
    result = {}
    for protocol, spec in specs.items():
        frozen = operating_points[protocol]
        point = _point(scores, gates, spec, frozen["gate_threshold"], frozen["kws_threshold"])
        point["calibration_definition"] = frozen.get("definition")
        point["protocol_shift"] = (frozen.get("ordinary_negative_ids") != point["ordinary_negative_ids"]
                                   or frozen.get("confusable_ids") != point["confusable_ids"])
        result[protocol] = point
    frozen = operating_points["shared"]
    result["shared"] = _shared_point(scores, gates, ids, specs, frozen["gate_threshold"], frozen["kws_threshold"])
    result.update({"mode": MODE, "measurement": MEASUREMENT,
                   "threshold_source": "frozen_validation_thresholds", "use_gate": bool(operating_points.get("use_gate", True)),
                   "proxy_notice": "Float32 normal-test sample-FAR proxy; quantized on-board FRR@FA/h is evaluated separately"})
    return result
