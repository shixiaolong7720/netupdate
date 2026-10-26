"""Exact, event-budget-aware FA/h metrics for independent audio clips.

This module deliberately has no dependency on PyTorch. Each negative example
represents one independent clip and can produce at most one activation. Its
duration contributes once to exposure. Overlapping windows from continuous
audio require a separate event-merging evaluator; summing their durations is
not a valid streaming FA/h measurement.

The two benchmark protocols are ``real`` (ID_0 vs ID_21, <=1 FA/h) and
``real_cw`` (ID_0 vs ID_21 + ID_25, <=20 FA/h). If ID_25 is absent, ID_1..20
are used as an explicitly reported development-set substitute for CW. Dataset
loading must exclude ID_25_noise: numeric IDs alone cannot distinguish noise
from human confusable words.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


PROTOCOL_TARGETS = {"real": 1.0, "real_cw": 20.0}


def exact_threshold_for_budget(negative_scores: np.ndarray, fp_budget: int) -> float:
    """Lowest float64 unit-score threshold satisfying an integer FP budget.

    Decisions use ``score >= threshold``. When ties straddle the budget, the
    complete tied group is rejected. Unlike taking the k-th largest score and
    moving above it, this uses the (k+1)-th largest score, allowing all k false
    activations whenever their scores are distinct. A threshold above 1 is a
    valid reject-all decision, including saturated score==1 examples.
    """
    scores = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("negative_scores must contain finite unit scores in [0, 1]")
    if isinstance(fp_budget, (bool, np.bool_)) or int(fp_budget) != fp_budget or fp_budget < 0:
        raise ValueError("fp_budget must be a nonnegative integer")
    budget = int(fp_budget)
    if scores.size <= budget:
        return 0.0
    # Element at ascending index n-k-1 is the first disallowed score.
    boundary = float(np.partition(scores, scores.size - budget - 1)[scores.size - budget - 1])
    return float(np.nextafter(np.float64(boundary), np.float64(np.inf)))


def protocol_mask(cids: np.ndarray, protocol: str) -> tuple[np.ndarray, str, bool]:
    """Return the selected examples, readable definition, and CW-fallback flag."""
    ids = np.asarray(cids).reshape(-1)
    if protocol not in PROTOCOL_TARGETS:
        raise ValueError(f"Unknown protocol: {protocol!r}")
    if not np.any(ids == 0):
        raise ValueError(f"{protocol}: missing ID_0 positive examples")
    if not np.any(ids == 21):
        raise ValueError(f"{protocol}: missing ID_21 real negative examples")
    if protocol == "real":
        return (ids == 0) | (ids == 21), "ID_0_Wake + ID_21_RealNonWake", False
    if np.any(ids == 25):
        return (
            (ids == 0) | (ids == 21) | (ids == 25),
            "ID_0_Wake + ID_21_RealNonWake + ID_25_HumanCW",
            False,
        )
    hard = (ids >= 1) & (ids <= 20)
    if not np.any(hard):
        raise ValueError("real_cw: missing both ID_25 CW and ID_1..20 development substitutes")
    return (
        (ids == 0) | (ids == 21) | hard,
        "ID_0_Wake + ID_21_RealNonWake + ID_1_to_20_HardCW (development substitute; ID_25 absent)",
        True,
    )


def _validate_inputs(scores, gate_probs, cids, durations_sec):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    gate = np.asarray(gate_probs, dtype=np.float64).reshape(-1)
    raw_ids = np.asarray(cids).reshape(-1)
    if scores.size == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    if scores.size != gate.size or scores.size != raw_ids.size:
        raise ValueError("scores, gate_probs and cids must have equal lengths")
    if np.any(~np.isfinite(raw_ids)) or np.any(raw_ids < 0) or np.any(raw_ids != np.floor(raw_ids)):
        raise ValueError("cids must contain nonnegative integer class IDs")
    ids = raw_ids.astype(np.int64)
    duration = np.asarray(durations_sec, dtype=np.float64)
    if duration.ndim == 0:
        duration = np.full(scores.size, float(duration), dtype=np.float64)
    duration = duration.reshape(-1)
    if duration.size != scores.size:
        raise ValueError("durations_sec must be a scalar or have one value per example")
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must contain finite unit scores in [0, 1]")
    if np.any(~np.isfinite(gate)) or np.any((gate < 0) | (gate > 1)):
        raise ValueError("gate_probs must contain finite probabilities in [0, 1]")
    if np.any(~np.isfinite(duration)) or np.any(duration < 0) or np.any(duration[ids != 0] <= 0):
        raise ValueError("durations_sec must be finite; all negative clips need positive durations")
    return scores, gate, ids, duration


def _prepare_protocols(ids, duration, negative_hours_override):
    if negative_hours_override is None:
        overrides = {}
    elif isinstance(negative_hours_override, Mapping):
        overrides = dict(negative_hours_override)
        unknown = set(overrides) - set(PROTOCOL_TARGETS)
        if unknown:
            raise ValueError(f"Unknown negative-hour override protocols: {sorted(unknown)}")
    else:
        raise ValueError("negative_hours_override must be a mapping from protocol to hours")
    prepared = {}
    for protocol, target in PROTOCOL_TARGETS.items():
        mask, definition, fallback = protocol_mask(ids, protocol)
        pos = mask & (ids == 0)
        neg = mask & (ids != 0)
        hours = float(overrides.get(protocol, float(np.sum(duration[neg])) / 3600.0))
        if not math.isfinite(hours) or hours <= 0:
            raise ValueError(f"{protocol}: negative exposure must be finite and greater than zero")
        prepared[protocol] = {
            "positive": pos,
            "negative": neg,
            "negative_hours": hours,
            "target_fah": target,
            "definition": definition,
            "uses_confusable_fallback": fallback,
            "hours_source": "override" if protocol in overrides else "sum_of_negative_clip_durations",
        }
    return prepared


def _point(scores, gate, spec, gate_threshold, kws_threshold):
    gate_threshold, kws_threshold = float(gate_threshold), float(kws_threshold)
    if not math.isfinite(gate_threshold) or not math.isfinite(kws_threshold):
        raise ValueError("Operating thresholds must be finite")
    gate_pass = gate >= gate_threshold
    predicted = gate_pass & (scores >= kws_threshold)
    pos, neg = spec["positive"], spec["negative"]
    n_pos, n_neg = int(np.sum(pos)), int(np.sum(neg))
    fp, fn = int(np.sum(predicted & neg)), int(np.sum(~predicted & pos))
    fah = fp / spec["negative_hours"]
    return {
        "gate_threshold": gate_threshold,
        "kws_threshold": kws_threshold,
        "target_fah": float(spec["target_fah"]),
        "frr": float(fn / n_pos),
        "fa_per_hour": float(fah),
        "fp": fp,
        "fn": fn,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "negative_hours": float(spec["negative_hours"]),
        "definition": spec["definition"],
        "uses_confusable_fallback": bool(spec["uses_confusable_fallback"]),
        "hours_source": spec["hours_source"],
        "feasible": bool(fah <= spec["target_fah"]),
        "gate_rejected_positive": int(np.sum(pos & ~gate_pass)),
        "kws_rejected_positive": int(np.sum(pos & gate_pass & (scores < kws_threshold))),
    }


def _gate_thresholds(gate, count, use_gate):
    if not use_gate:
        return np.asarray([0.0], dtype=np.float64)
    if isinstance(count, (bool, np.bool_)) or int(count) != count or count < 2:
        raise ValueError("gate_candidates must be an integer >=2")
    unique = np.unique(gate)
    if unique.size > count:
        positions = np.unique(np.rint(np.linspace(0, unique.size - 1, int(count))).astype(int))
        unique = unique[positions]
    return np.unique(np.concatenate([
        np.asarray([0.0], dtype=np.float64), unique,
        np.asarray([np.nextafter(np.float64(np.max(gate)), np.float64(np.inf))]),
    ]))


def _point_key(point):
    return (point["frr"], point["fa_per_hour"], point["gate_threshold"], point["kws_threshold"])


def calibrate_operating_points(
    scores,
    gate_probs,
    cids,
    durations_sec,
    gate_candidates: int = 64,
    negative_hours_override: Mapping[str, float] | None = None,
    use_gate: bool = True,
) -> dict[str, Any]:
    """Jointly calibrate Gate/KWS on development data for both FA/h budgets.

    The KWS threshold is exact for each searched Gate threshold. When the number
    of distinct Gate scores exceeds ``gate_candidates``, only evenly spaced
    ranks plus pass-all/reject-all boundaries are searched. Set a sufficiently
    large count to enumerate every distinct Gate boundary. ``shared`` is one
    deployable threshold pair that satisfies both development-set constraints;
    protocol-specific pairs are additionally retained for benchmark reporting.
    """
    scores, gate, ids, duration = _validate_inputs(scores, gate_probs, cids, durations_sec)
    specs = _prepare_protocols(ids, duration, negative_hours_override)
    relevant = specs["real_cw"]["positive"] | specs["real_cw"]["negative"]
    thresholds = _gate_thresholds(gate[relevant], gate_candidates, use_gate)
    best = {protocol: None for protocol in PROTOCOL_TARGETS}
    best_shared = None
    for gate_threshold in thresholds:
        gate_pass = gate >= gate_threshold
        kws_thresholds = {}
        for protocol, spec in specs.items():
            budget = math.floor(spec["target_fah"] * spec["negative_hours"])
            kws_threshold = exact_threshold_for_budget(scores[spec["negative"] & gate_pass], budget)
            point = _point(scores, gate, spec, gate_threshold, kws_threshold)
            kws_thresholds[protocol] = kws_threshold
            if not point["feasible"]:
                raise ArithmeticError("Exact threshold unexpectedly exceeded its FA/h budget")
            if best[protocol] is None or _point_key(point) < _point_key(best[protocol]):
                best[protocol] = point
        shared_kws = max(kws_thresholds.values())
        shared = {
            "gate_threshold": float(gate_threshold),
            "kws_threshold": float(shared_kws),
            **{p: _point(scores, gate, spec, gate_threshold, shared_kws) for p, spec in specs.items()},
        }
        shared_key = (
            max(shared[p]["frr"] for p in PROTOCOL_TARGETS),
            sum(shared[p]["frr"] for p in PROTOCOL_TARGETS),
            sum(shared[p]["fa_per_hour"] for p in PROTOCOL_TARGETS),
            shared["gate_threshold"], shared["kws_threshold"],
        )
        if best_shared is None or shared_key < best_shared[0]:
            best_shared = (shared_key, shared)
    return {
        **best,
        "shared": best_shared[1],
        "measurement": "independent_clip_activations_per_negative_audio_hour",
        "threshold_source": "development_calibration",
        "gate_thresholds_searched": int(len(thresholds)),
        "use_gate": bool(use_gate),
    }


def evaluate_operating_points(
    scores,
    gate_probs,
    cids,
    durations_sec,
    operating_points: Mapping[str, Any],
    negative_hours_override: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Measure frozen development thresholds on a new split; never recalibrate.

    Test FA/h can exceed its target and ``feasible`` then reports False. Human
    CW test data may replace development HardCW substitutes; this protocol shift
    is recorded separately from threshold provenance.
    """
    scores, gate, ids, duration = _validate_inputs(scores, gate_probs, cids, durations_sec)
    specs = _prepare_protocols(ids, duration, negative_hours_override)
    result = {}
    for protocol, spec in specs.items():
        frozen = operating_points[protocol]
        point = _point(scores, gate, spec, frozen["gate_threshold"], frozen["kws_threshold"])
        point["calibration_definition"] = frozen.get("definition")
        point["protocol_shift"] = frozen.get("definition") != point["definition"]
        result[protocol] = point
    shared = operating_points["shared"]
    result["shared"] = {
        "gate_threshold": float(shared["gate_threshold"]),
        "kws_threshold": float(shared["kws_threshold"]),
        **{p: _point(scores, gate, spec, shared["gate_threshold"], shared["kws_threshold"])
           for p, spec in specs.items()},
    }
    result["measurement"] = "independent_clip_activations_per_negative_audio_hour"
    result["threshold_source"] = "frozen_development_thresholds"
    result["use_gate"] = bool(operating_points.get("use_gate", True))
    return result
