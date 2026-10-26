"""Export FP32 training checkpoints with the original model-key conventions.

These files provide inputs to a separate quantization pipeline. This module
does not calibrate INT8 ranges, export ONNX, or run FUKE_FAH/board simulation.
The shared thresholds come from the floating-point validation split and must
not be presented as measured quantized-device operating points.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch


SCORE_DEFINITION = "unit_score=(cosine(embedding,wake_centroid)+1)/2"
THRESHOLD_SOURCE = "floating_point_validation_sample_far_proxy"
QUANTIZATION_STATUS = "pending_external_pipeline"


def _state_dict_fp32_cpu(module):
    result = {}
    for key, tensor in module.state_dict().items():
        # Preserve integer buffers where a model has them; all floating model
        # weights are exported as FP32. No trained module is modified in place.
        value = tensor.detach().cpu()
        if value.is_floating_point():
            value = value.to(dtype=torch.float32)
            if not torch.isfinite(value).all():
                raise ValueError(f"Cannot export nonfinite model tensor: {key}")
        result[key] = value.clone()
    return result


def _save_atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def export_legacy_checkpoints(output_dir, model, gate, centroid, operating_points, model_config, args=None):
    """Save old-style model keys and an explicit FP32 export manifest.

    Returns paths under keys ``kws``, ``gate`` (None if no gate), ``system`` and
    ``manifest``. ``operating_points['shared']`` supplies the top-level Gate and
    KWS thresholds. ``centroid`` is copied exactly, never renormalized; it must
    already be FP32 and normalized, as produced by training enrollment.

    With gate=None, no Gate file is emitted and the system checkpoint contains
    gate_model=None. Such a checkpoint requires inference/quantization tooling
    that explicitly supports an absent Gate; a mandatory dual-model loader must
    not call load_state_dict(None). No substitute random Gate is fabricated.

    args is accepted for the training caller's API but deliberately not dumped:
    complete namespaces may contain unrelated configuration or credentials.
    """
    if not isinstance(model_config, Mapping):
        raise ValueError("model_config must be a mapping of constructor arguments")
    config = dict(model_config)
    # Validate primitive, JSON-compatible metadata before creating output files.
    json.dumps(config, allow_nan=False)
    try:
        shared = operating_points["shared"]
        gate_threshold = float(shared["gate_threshold"])
        kws_threshold = float(shared["kws_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Export requires shared gate_threshold and kws_threshold") from exc
    if not all(math.isfinite(value) and value >= 0 for value in (gate_threshold, kws_threshold)):
        raise ValueError("Shared thresholds must be finite nonnegative values")
    if gate is None and gate_threshold > 0:
        raise ValueError("An absent Gate requires the shared bypass threshold 0")
    if not isinstance(centroid, torch.Tensor) or centroid.dtype != torch.float32:
        raise ValueError("Export centroid must be the training FP32 tensor; no implicit requantization")
    reference = centroid.detach().cpu().clone()
    if (reference.ndim != 1 or reference.numel() < 2 or not torch.isfinite(reference).all()
            or not torch.isclose(reference.norm(), reference.new_tensor(1.0), atol=1e-4)):
        raise ValueError("Export centroid must be a finite normalized vector")
    embedding_dim = int(config.get("embedding_dim", reference.numel()))
    if embedding_dim != reference.numel():
        raise ValueError("model_config embedding_dim differs from the enrollment centroid")

    kws_state = _state_dict_fp32_cpu(model)
    gate_state = _state_dict_fp32_cpu(gate) if gate is not None else None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    kws_path = output / "best_ge2e_kws.pth"
    gate_path = output / "best_tiny_gate.pth" if gate is not None else None
    system_path = output / "best_ge2e_kws_tiny_gate_system.pth"
    manifest_path = output / "export_manifest.json"
    common = {
        "model_config": config,
        "score_definition": SCORE_DEFINITION,
        "threshold_source": THRESHOLD_SOURCE,
        "quantization_status": QUANTIZATION_STATUS,
        "floating_point_dtype": "float32",
    }
    _save_atomic(kws_path, {
        **common, "model": kws_state, "wake_centroid": reference,
        "kws_threshold": kws_threshold, "embedding_dim": embedding_dim,
        "model_type": "DualStream_DSCNN_TCN_AttPool_GE2E",
    })
    if gate_path is not None:
        _save_atomic(gate_path, {
            **common, "gate": gate_state, "gate_threshold": gate_threshold,
            "gate_type": "TinyMFCCGate", "gate_definition": "ID23_reject_other_speech_pass",
        })
    decision = ("(p_gate>=gate_threshold) AND (unit_cosine_score>=kws_threshold)"
                if gate is not None else "unit_cosine_score>=kws_threshold; Gate absent")
    _save_atomic(system_path, {
        **common, "kws_model": kws_state, "gate_model": gate_state,
        "wake_centroid": reference, "gate_threshold": gate_threshold,
        "kws_threshold": kws_threshold, "embedding_dim": embedding_dim,
        "decision": decision, "gate_present": gate is not None,
        "deployment_operating_point": "shared",
    })
    paths = {"kws": str(kws_path), "gate": str(gate_path) if gate_path is not None else None,
             "system": str(system_path), "manifest": str(manifest_path)}
    manifest = {
        "export_type": "pytorch_fp32_state_dict_checkpoints",
        "checkpoints": {key: Path(value).name if value is not None else None
                        for key, value in paths.items() if key != "manifest"},
        "model_config": config,
        "inputs": {"mfcc": ["N", 1, 20, "T"], "logmel": ["N", 1, 40, "T"],
                   "mfcc_gate": ["N", 1, 60, "T"]},
        "shape_note": "N and T are symbolic documentation; no dynamic-shape graph export has been performed",
        "feature_contract": "Use the same feature extraction, normalization and clip/window definition as training",
        "floating_point_dtype": "float32",
        "score_definition": SCORE_DEFINITION,
        "wake_centroid_source": "training-only enrollment subset; copied without renormalization",
        "gate_present": gate is not None,
        "gate_absent_requirement": (None if gate is not None else
                                    "Inference and quantization loaders must support an absent Gate; gate_model is null"),
        "gate_threshold": gate_threshold, "kws_threshold": kws_threshold,
        "threshold_source": THRESHOLD_SOURCE,
        "quantization_status": QUANTIZATION_STATUS,
        "calibration_data_allowed": "train/val only; FUKE_FAH excluded",
        "artifacts_not_created": ["INT8 model", "ONNX graph", "board binary"],
        "followup": (
            "Quantize using the existing external pipeline and appropriate train-only representative data. "
            "Quantization may change embeddings, centroid and scores; reconstruct/calibrate with train/val "
            "as required by that pipeline. Use FUKE_FAH only for subsequent simulated-board acceptance."
        ),
    }
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(manifest_path)
    return paths
