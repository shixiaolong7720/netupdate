# -*- coding: utf-8 -*-
# Clean CL training version generated from the user's final lite2 configuration.
# This version keeps FRR-FAR / DET-oriented continual replay only.
# Hour-based false-activation evaluation and paper-style curve generation have been removed.
"""
DS-CNN + TCN + AttentiveStatsPooling + stable True GE2E metric learning
+ GroupNorm + pure-GE2E warm-up + ramped background metric rejection
+ staged Tiny Environmental Gate + FRR-FAR-oriented continual false-alarm replay.

Stage 1
-------
1) Confusable episode:
   ID_0 wake + N-1 IDs sampled from ID_1~ID_20, exactly M utterances per class.
   Optimize standard GE2E leave-one-out centroid softmax loss.
2) Background metric batch:
   M wake samples + stratified ID_21 / ID_22 / ID_24 samples.
   Push heterogeneous background embeddings away from the wake centroid without BCE.
3) Build a wake enrollment centroid from a fixed training-only enrollment subset.
   Validation/test KWS score = normalized cosine similarity to this centroid.

Stage 2
-------
Freeze KWS conceptually and independently train TinyMFCCGate:
MFCC+Delta+DeltaDelta [60,T] -> pointwise projection -> depthwise temporal conv
-> pointwise fusion -> dilated residual conv -> mean/max pooling -> binary gate.

Stage 3
-------
Calibrate Gate threshold on validation constraints, then calibrate KWS similarity
threshold at the target FAR. Final decision:
    (p_gate >= gate_threshold) AND (cosine_score >= kws_threshold)

Notes
-----
* ID_23 Env is handled primarily by Tiny Gate and is excluded from Stage-1 KWS metric batches.
* Validation/test samples never participate in wake centroid construction.
* Scores used by plotting are mapped from cosine [-1,1] to [0,1] by (cos+1)/2.
"""

import os
import glob
import csv
import time
import random
import shutil
import sys
import subprocess
import copy
import hashlib
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter

import librosa
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


# ============================================================================
# 0. Reproducibility and device
# ============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


SEED = 42
set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOWS_LOW_MEMORY_SAFE = (os.name == "nt")
# Windows + PyTorch DataLoader multiprocessing can exhaust the page file.
# Safe mode disables worker subprocesses and pinned-memory pressure.
PIN_MEMORY = False if WINDOWS_LOW_MEMORY_SAFE else (DEVICE.type == "cuda")

# RTX 5070 12 GB performance settings. Shapes are fixed, so cuDNN autotuning is beneficial.
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


# ============================================================================
# 1. Paths and hyperparameters
# ============================================================================
TRAIN_DATASET_DIR = r"D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake\train"
VAL_DATASET_DIR = r"D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake\val"
TEST_DATASET_DIR = r"D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake\test"

DIAG_DIR = r"D:\myproject\project\mode_pth\compare_test\消融\Mymodel\Mymodel_TTS_GE2E_CL\lite2_frrfar_0.0050"
KWS_BEST_MODEL_PATH = os.path.join(DIAG_DIR, "best_ge2e_kws.pth")
GATE_BEST_MODEL_PATH = os.path.join(DIAG_DIR, "best_tiny_gate.pth")
SYSTEM_BEST_MODEL_PATH = os.path.join(DIAG_DIR, "best_ge2e_kws_tiny_gate_system.pth")
os.makedirs(DIAG_DIR, exist_ok=True)

# Evaluation
TARGET_FAR = 0.0050
THR_MODE = "fp_budget"  # {"fp_budget", "roc"}
FREE_THR = 0.50
SWEEP_POINTS = 201

# FAR-FRR / DET-style curve output.
# The model is evaluated only once; these target FAR points are obtained by
# sweeping decision thresholds offline on the saved sample scores.
FAR_FRR_TARGET_FARS = np.asarray([
    0.001, 0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015,
    0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.070,
], dtype=np.float64)
FAR_FRR_XMAX = 0.070
FAR_FRR_YMAX = 0.020
FAR_FRR_YTICK_STEP = 0.002
FAR_FRR_MINOR_YTICK_N = 2
FAR_FRR_MODEL_LABEL = "MyModel + TTS + GE2E + CL + TinyGate"
FAR_FRR_DPI = 220



# DataLoader: tuned for RTX 5070 12 GB + a modern multi-core CPU
NUM_WORKERS = 0 if WINDOWS_LOW_MEMORY_SAFE else 6
EVAL_BATCH_SIZE = 256 if WINDOWS_LOW_MEMORY_SAFE else 512

# Stage-1 stable True GE2E
EMBEDDING_DIM = 128
N_CLASSES = 10
M_UTTERANCES = 12
BATCHES_PER_EPOCH = 800
ALTERNATE_BATCH_TYPES = True
P_CONFUSABLE = 0.50  # only used after warm-up when ALTERNATE_BATCH_TYPES=False
HARD_IDS = tuple(range(1, 21))
BACKGROUND_IDS = (21, 22, 24)
BACKGROUND_RATIOS = (0.50, 0.30, 0.20)  # Pep / Mask / TTS within background samples

# Conservative optimization is important for metric learning.
KWS_LR = 1e-4
KWS_WEIGHT_DECAY = 1e-4
KWS_EPOCHS = 140
KWS_EARLY_STOP_PATIENCE = 25
CLIP_NORM = 3.0

# First learn a valid ID_0~ID_20 metric space, then gradually add background rejection.
PURE_GE2E_WARMUP_EPOCHS = 10
BACKGROUND_RAMP_EPOCHS = 10
LAMBDA_BG_METRIC_MAX = 0.10
BG_NEG_MARGIN = 0.30
METRIC_TEMPERATURE = 0.10

# A fixed scale prevents the scalar temperature from hiding encoder collapse.
GE2E_FIXED_SCALE = 10.0
LEARN_GE2E_SCALE = False
GE2E_SCALE_MIN = 5.0
GE2E_SCALE_MAX = 30.0
GE2E_SCALE_LR = 1e-5

# Do not save a collapsed model merely because its fixed-FAR FRR is marginally lower.
MIN_EPISODE_ACC_FOR_SELECTION = 0.20
MIN_EPISODE_MARGIN_FOR_SELECTION = 0.00
COLLAPSE_CHECK_EPOCH = 10
COLLAPSE_PAIRWISE_COS = 0.985
COLLAPSE_FEATURE_STD = 0.010
ABORT_ON_COLLAPSE = True

# Fixed training-only wake enrollment subset
WAKE_ENROLL_SAMPLES = 2048
WAKE_ENROLL_BATCH_SIZE = 256 if WINDOWS_LOW_MEMORY_SAFE else 512
WAKE_ENROLL_SEED = 20260619

# Stage-1 augmentation
SPEC_P = 0.30
FREQ_MASK_MFCC = 2
FREQ_MASK_MEL = 4
TIME_MASK_FRAC = 0.03

# KWS encoder
BASE_C = 32
DROPOUT = 0.10
USE_TEMPORAL_1x5 = True
TCN_LAYERS = 4
TCN_DILATIONS = (1, 2, 2, 4)
ATT_HIDDEN = 96

# Tiny Gate
ENV_CID = 23
GATE_ENV_PER_EPOCH = 3800
GATE_WAKE_PER_EPOCH = 3200
GATE_PEP_PER_EPOCH = 4000
GATE_TTS_PER_EPOCH = 1600
GATE_HARD_PER_EPOCH = 800
GATE_MASK_PER_EPOCH = 480
GATE_ID24_PER_EPOCH = 320

GATE_BATCH_SIZE = 128 if WINDOWS_LOW_MEMORY_SAFE else 256
GATE_NUM_WORKERS = 0 if WINDOWS_LOW_MEMORY_SAFE else 6
GATE_EPOCHS = 40
GATE_EARLY_STOP_PATIENCE = 8
GATE_LR = 1e-3
GATE_WEIGHT_DECAY = 1e-4
GATE_DROPOUT = 0.10
GATE_ENV_LOSS_WEIGHT = 1.8
GATE_SPEC_P = 0.40
GATE_FREQ_MASK_MAX = 6
GATE_TIME_MASK_FRAC = 0.05

# Gate threshold constraints
GATE_THR_POINTS = 1001
MIN_WAKE_PASS = 0.999
MIN_HARD_PASS = 0.990
MIN_PEP_PASS = 0.980
MIN_MASK_PASS = 0.980
MIN_TTS_PASS = 0.980

GROUP_ORDER = [
    "ID_0_wake",
    "ID_1-20_hard",
    "ID_21_Pep",
    "ID_22_Mask",
    "ID_23_Env",
    "ID_24_TTS",
]


# ============================================================================
# Continual Learning / False Alarm Replay
# ============================================================================
ENABLE_CONTINUAL_REPLAY = True
CL_ROUNDS = 1  # conservative CL: avoid over-correcting wake boundary

# Prefer a dedicated adaptation split. If it does not exist, the script can
# deterministically split validation into adaptation-mining and calibration sets.
RAW_WAV_BASE_DIR = r"D:\myproject\project\Data\Dataset\hi_miya"
ADAPT_FEATURE_DIR = r"D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake\adapt"
ADAPT_WAV_DIR = os.path.join(RAW_WAV_BASE_DIR, "adapt")
ALLOW_SPLIT_VAL_FOR_ADAPT = True
VAL_ADAPT_FRACTION = 0.40
VAL_SPLIT_SEED = 20260620

REPLAY_WAV_ROOT = os.path.join(RAW_WAV_BASE_DIR, "replay_buffer_wav_true_ge2e")
REPLAY_FEATURE_ROOT = r"D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake\replay_buffer_true_ge2e"
POINTSOURCE_NOISE_DIR = r"D:\myproject\gittest\rirs_noises\RIRS_NOISES\pointsource_noises"
FEATURE_EXTRACT_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "extract_features_hi_miya_replay_true_ge2e.py",
)
AUTO_EXTRACT_REPLAY_FEATURES = True

# Mine only system-level false alarms: Gate passes AND KWS score crosses threshold.
REPLAY_TOPK_FP_TOTAL = 250  # less aggressive replay mining
REPLAY_TOPK_FP_PER_ID = 50
REPLAY_INCLUDE_ORIGINAL = True
REPLAY_AUG_PER_CASE = 1
REPLAY_TARGET_SR = 16000
REPLAY_TARGET_DUR = 1.5
REPLAY_GAIN_DB_RANGE = (-6.0, 3.0)
REPLAY_SHIFT_MS = 120
REPLAY_SNR_DB_RANGE = (20.0, 35.0)

# True-GE2E continual fine-tuning.
CL_KWS_EPOCHS = 8
CL_KWS_LR = 8e-6
CL_KWS_WEIGHT_DECAY = 1e-4
CL_KWS_PATIENCE = 3
CL_BATCHES_PER_EPOCH = 300
CL_CONFUSABLE_STEPS = 2
CL_BACKGROUND_STEPS = 1
CL_HARD_REPLAY_FRACTION = 0.25
CL_BG_REPLAY_FRACTION = 0.25
CL_HARD_ID_PRIORITY = 2.0
CL_LAMBDA_REPLAY_MARGIN = 0.12
CL_REPLAY_NEG_MARGIN = 0.25
CL_REPLAY_TEMPERATURE = 0.10
CL_LAMBDA_STABILITY = 0.15
CL_LAMBDA_CENTROID_ANCHOR = 0.15

# Tiny Gate continual fine-tuning.
CL_GATE_EPOCHS = 10
CL_GATE_LR = 2e-4
CL_GATE_PATIENCE = 4
CL_GATE_REPLAY_ENV_MAX = 2500
CL_GATE_REPLAY_NONENV_MAX = 2500
CL_GATE_LAMBDA_STABILITY = 0.05

# Disable Tiny Gate continual fine-tuning. The Gate trained in Stage 2 is kept fixed
# during CL so that replay does not make the system overly conservative.
ENABLE_GATE_CL = False

# CL model selection is aligned with FRR-FAR / DET metrics.
# Candidate CL checkpoints are accepted only if the low-FAR FRR curve, EER,
# DET-AUC, and Hard-DET-AUC improve on the validation/calibration split.
USE_CL_FARFRR_ACCEPTANCE = True

CL_FARFRR_TARGETS = np.asarray([
    0.001,
    0.0025,
    0.005,
    0.010,
    0.020,
], dtype=np.float64)
CL_FARFRR_WEIGHTS = np.asarray([
    0.25,
    0.25,
    0.25,
    0.15,
    0.10,
], dtype=np.float64)
CL_PRIMARY_FAR = 0.005

# Metric weights for checkpoint selection. Lower is better.
SELECT_WEIGHT_FRR_FAR = 0.20
SELECT_WEIGHT_EER = 0.25
SELECT_WEIGHT_DETAUC = 0.25
SELECT_WEIGHT_HARD_DETAUC = 0.30

# Acceptance tolerances. Values are absolute rates, e.g. 0.003 = 0.3 percentage points.
CL_ACCEPT_MAX_FRR_WORSE = 0.003
CL_ACCEPT_MAX_DETAUC_WORSE = 0.003
CL_ACCEPT_MIN_SCORE_GAIN = 1e-5

# Wake recall protection during CL. It prevents replay fine-tuning from pushing
# true wake samples below the low-FAR operating threshold.
CL_LAMBDA_WAKE_RECALL = 0.20
CL_WAKE_POS_MARGIN = 0.03
CL_WAKE_RECALL_TEMPERATURE = 0.10

# Per-epoch monitoring. Each KWS/CL epoch saves exactly three figures:
# confusion matrix, group error bar, and score distribution.
SAVE_EPOCH_MONITOR_FIGS = True
EPOCH_MONITOR_DIR = os.path.join(DIAG_DIR, "epoch_monitor")

# The blind test is evaluated once after all accepted replay rounds.
EVALUATE_TEST_EACH_ROUND = False


# ============================================================================
# 2. Dataset
# ============================================================================
class FolderDualDataset(Dataset):
    """
    Expected folder layout:
        root/ID_0.../*.npz
        root/ID_1.../*.npz
        ...

    Every NPZ must contain:
        mfcc:      [20,T]
        logmel:    [40,T]
        mfcc_gate: [60,T]

    Returns:
        mfcc [1,20,T], logmel [1,40,T], mfcc_gate [1,60,T],
        binary label, class_id, sample id.
    """

    def __init__(self, root_dir: str):
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"找不到数据集文件夹: {root_dir}")

        self.root_dir = root_dir
        self.file_paths: List[str] = []
        self.class_ids: List[int] = []
        self.labels: List[float] = []

        print(f"正在扫描特征目录: {root_dir}")
        folder_names = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("ID_")
        ]

        for folder_name in sorted(folder_names):
            parts = folder_name.split("_")
            if len(parts) < 2:
                continue
            try:
                class_id = int(parts[1])
            except ValueError:
                continue

            binary_label = 1.0 if class_id == 0 else 0.0
            folder_path = os.path.join(root_dir, folder_name)
            for npz_path in glob.glob(os.path.join(folder_path, "*.npz")):
                self.file_paths.append(npz_path)
                self.class_ids.append(class_id)
                self.labels.append(binary_label)

        self.class_ids = np.asarray(self.class_ids, dtype=np.int32)
        self.labels = np.asarray(self.labels, dtype=np.float32)

        if len(self.file_paths) == 0:
            raise RuntimeError(f"目录中没有找到任何 NPZ 特征: {root_dir}")

        unique, counts = np.unique(self.class_ids, return_counts=True)
        summary = ", ".join(f"ID_{int(c)}={int(n)}" for c, n in zip(unique, counts))
        print(f"扫描完成: total={len(self.file_paths)} | {summary}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        npz_path = self.file_paths[idx]
        try:
            with np.load(npz_path) as data:
                mfcc = torch.as_tensor(data["mfcc"], dtype=torch.float32).unsqueeze(0)
                logmel = torch.as_tensor(data["logmel"], dtype=torch.float32).unsqueeze(0)
                mfcc_gate = torch.as_tensor(data["mfcc_gate"], dtype=torch.float32).unsqueeze(0)
        except Exception as exc:
            raise RuntimeError(f"读取特征失败: {npz_path}") from exc

        if mfcc.ndim != 3 or mfcc.shape[1] != 20:
            raise ValueError(f"mfcc 应为 [1,20,T]，实际 {tuple(mfcc.shape)}: {npz_path}")
        if logmel.ndim != 3 or logmel.shape[1] != 40:
            raise ValueError(f"logmel 应为 [1,40,T]，实际 {tuple(logmel.shape)}: {npz_path}")
        if mfcc_gate.ndim != 3 or mfcc_gate.shape[1] != 60:
            raise ValueError(f"mfcc_gate 应为 [1,60,T]，实际 {tuple(mfcc_gate.shape)}: {npz_path}")

        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        cid = int(self.class_ids[idx])
        folder = os.path.basename(os.path.dirname(npz_path))
        sid = f"{folder}/{os.path.basename(npz_path)}"
        return mfcc, logmel, mfcc_gate, y, cid, sid


class MultiRootFolderDualDataset(Dataset):
    """
    Original train + cumulative replay roots.

    Returns the normal six fields plus is_replay. ID_0 is accepted only from the
    original root; replay roots contain negative samples assigned back to ID_1~ID_24.
    """

    def __init__(self, original_root: str, replay_roots: Sequence[str]):
        roots = [(original_root, False)] + [(r, True) for r in replay_roots]
        self.file_paths: List[str] = []
        self.class_ids: List[int] = []
        self.labels: List[float] = []
        self.is_replay: List[bool] = []

        for root, replay_flag in roots:
            if not os.path.isdir(root):
                print(f"警告: 跳过不存在的数据根目录: {root}")
                continue
            folders = [
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d)) and d.startswith("ID_")
            ]
            for folder in sorted(folders):
                try:
                    cid = int(folder.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if replay_flag and cid == 0:
                    continue
                for path in sorted(glob.glob(os.path.join(root, folder, "*.npz"))):
                    self.file_paths.append(path)
                    self.class_ids.append(cid)
                    self.labels.append(1.0 if cid == 0 else 0.0)
                    self.is_replay.append(bool(replay_flag))

        self.class_ids = np.asarray(self.class_ids, dtype=np.int32)
        self.labels = np.asarray(self.labels, dtype=np.float32)
        self.is_replay = np.asarray(self.is_replay, dtype=np.bool_)
        if len(self.file_paths) == 0:
            raise RuntimeError("MultiRootFolderDualDataset没有加载到任何NPZ")

        print(
            f"MultiRoot dataset: total={len(self.file_paths)} | "
            f"original={int((~self.is_replay).sum())} | replay={int(self.is_replay.sum())}"
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        with np.load(path) as data:
            mfcc = torch.as_tensor(data["mfcc"], dtype=torch.float32).unsqueeze(0)
            logmel = torch.as_tensor(data["logmel"], dtype=torch.float32).unsqueeze(0)
            mfcc_gate = torch.as_tensor(data["mfcc_gate"], dtype=torch.float32).unsqueeze(0)
        cid = int(self.class_ids[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        folder = os.path.basename(os.path.dirname(path))
        sid = f"{folder}/{os.path.basename(path)}"
        return mfcc, logmel, mfcc_gate, y, cid, sid, bool(self.is_replay[idx])


class ReplayAwareGE2EMetricBatchSampler(Sampler[List[int]]):
    """
    Keeps the exact N x M True-GE2E episode while injecting replay samples.

    * Confusable episode: ID_0 + N-1 Hard IDs; replay Hard samples remain in
      their original ID and contribute to the correct class centroid.
    * Background batch: ID_21/22/24 replay samples enter background metric loss.
    * ID_23 replay never enters KWS metric training and is reserved for Tiny Gate.
    """

    def __init__(self, dataset: MultiRootFolderDualDataset, seed: int):
        self.dataset = dataset
        self.base_seed = int(seed)
        self.current_epoch = 1
        self.rng = np.random.RandomState(self.base_seed)

        self.original_by_cid: Dict[int, np.ndarray] = {}
        self.replay_by_cid: Dict[int, np.ndarray] = {}
        for cid in np.unique(dataset.class_ids):
            mask_cid = dataset.class_ids == cid
            self.original_by_cid[int(cid)] = np.where(mask_cid & (~dataset.is_replay))[0]
            self.replay_by_cid[int(cid)] = np.where(mask_cid & dataset.is_replay)[0]

        if len(self.original_by_cid.get(0, np.array([]))) == 0:
            raise ValueError("CL数据中缺少原始ID_0")
        self.hard_ids = [
            cid for cid in HARD_IDS
            if len(self.original_by_cid.get(cid, np.array([]))) + len(self.replay_by_cid.get(cid, np.array([]))) > 0
        ]
        if len(self.hard_ids) < N_CLASSES - 1:
            raise ValueError("CL可用Hard ID不足N-1")
        for cid in BACKGROUND_IDS:
            if len(self.original_by_cid.get(cid, np.array([]))) + len(self.replay_by_cid.get(cid, np.array([]))) == 0:
                raise ValueError(f"CL背景池缺少ID_{cid}")

    def __len__(self) -> int:
        return CL_BATCHES_PER_EPOCH

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self.rng = np.random.RandomState(self.base_seed + 1013 * self.current_epoch)

    def _choice(self, pool: np.ndarray, n: int) -> List[int]:
        pool = np.asarray(pool, dtype=np.int64)
        if len(pool) == 0 or n <= 0:
            return []
        return self.rng.choice(pool, size=n, replace=len(pool) < n).astype(np.int64).tolist()

    def _sample_mixed(self, cid: int, n: int, replay_fraction: float) -> List[int]:
        orig = self.original_by_cid.get(int(cid), np.array([], dtype=np.int64))
        rep = self.replay_by_cid.get(int(cid), np.array([], dtype=np.int64))
        if len(rep) == 0:
            return self._choice(orig, n)
        if len(orig) == 0:
            return self._choice(rep, n)

        n_rep = int(round(n * float(replay_fraction)))
        n_rep = min(max(1, n_rep), n)
        n_orig = n - n_rep
        result = self._choice(rep, n_rep) + self._choice(orig, n_orig)
        self.rng.shuffle(result)
        return result

    @staticmethod
    def _background_counts(total: int) -> List[int]:
        ratios = np.asarray(BACKGROUND_RATIOS, dtype=np.float64)
        raw = ratios * total
        counts = np.floor(raw).astype(int)
        for idx in np.argsort(-(raw - counts))[: int(total - counts.sum())]:
            counts[idx] += 1
        return counts.tolist()

    def _select_hard_ids(self) -> List[int]:
        weights = np.ones(len(self.hard_ids), dtype=np.float64)
        for i, cid in enumerate(self.hard_ids):
            if len(self.replay_by_cid.get(cid, np.array([]))) > 0:
                weights[i] *= CL_HARD_ID_PRIORITY
        weights /= weights.sum()
        chosen = self.rng.choice(
            np.asarray(self.hard_ids),
            size=N_CLASSES - 1,
            replace=False,
            p=weights,
        )
        return [int(x) for x in chosen]

    def __iter__(self):
        period = max(1, CL_CONFUSABLE_STEPS + CL_BACKGROUND_STEPS)
        for step in range(CL_BATCHES_PER_EPOCH):
            use_confusable = (step % period) < CL_CONFUSABLE_STEPS
            batch: List[int] = []
            if use_confusable:
                batch.extend(self._choice(self.original_by_cid[0], M_UTTERANCES))
                for cid in self._select_hard_ids():
                    batch.extend(self._sample_mixed(cid, M_UTTERANCES, CL_HARD_REPLAY_FRACTION))
            else:
                batch.extend(self._choice(self.original_by_cid[0], M_UTTERANCES))
                total_bg = (N_CLASSES - 1) * M_UTTERANCES
                for cid, count in zip(BACKGROUND_IDS, self._background_counts(total_bg)):
                    batch.extend(self._sample_mixed(cid, count, CL_BG_REPLAY_FRACTION))
                wake_part = batch[:M_UTTERANCES]
                bg_part = batch[M_UTTERANCES:]
                self.rng.shuffle(bg_part)
                batch = wake_part + bg_part

            if len(batch) != N_CLASSES * M_UTTERANCES:
                raise RuntimeError("Replay-aware sampler batch size异常")
            yield batch


# ============================================================================
# 3. True GE2E / background metric batch sampler
# ============================================================================
class GE2EMetricBatchSampler(Sampler[List[int]]):
    """
    Alternates two batch types.

    Confusable episode:
        ID_0 + N-1 hard IDs from ID_1~ID_20, exactly M utterances/class.
        Total batch size = N*M.

    Background metric batch:
        M wake samples + (N-1)*M heterogeneous background samples,
        stratified over ID_21 / ID_22 / ID_24.
        Total batch size = N*M.

    ID_23 is intentionally excluded from KWS metric training and handled by Tiny Gate.
    """

    def __init__(
        self,
        dataset: FolderDualDataset,
        n_classes: int,
        m_utterances: int,
        batches_per_epoch: int,
        hard_ids: Sequence[int] = HARD_IDS,
        background_ids: Sequence[int] = BACKGROUND_IDS,
        background_ratios: Sequence[float] = BACKGROUND_RATIOS,
        alternate: bool = True,
        p_confusable: float = 0.5,
        seed: int = 1337,
    ):
        if n_classes < 2:
            raise ValueError("n_classes 必须至少为2")
        if m_utterances < 2:
            raise ValueError("m_utterances 必须至少为2，以计算 leave-one-out centroid")
        if len(background_ids) != len(background_ratios):
            raise ValueError("background_ids 与 background_ratios 长度不一致")
        if not np.isclose(sum(background_ratios), 1.0):
            raise ValueError("background_ratios 之和必须为1")

        self.dataset = dataset
        self.n_classes = int(n_classes)
        self.m_utterances = int(m_utterances)
        self.batches_per_epoch = int(batches_per_epoch)
        self.alternate = bool(alternate)
        self.p_confusable = float(p_confusable)
        self.base_seed = int(seed)
        self.current_epoch = 1
        self.rng = np.random.RandomState(self.base_seed)

        self.idx_by_cid: Dict[int, np.ndarray] = {
            int(cid): np.where(dataset.class_ids == cid)[0]
            for cid in np.unique(dataset.class_ids)
        }
        if 0 not in self.idx_by_cid:
            raise ValueError("训练集中缺少 ID_0 Wake")

        self.hard_ids = [
            int(cid) for cid in hard_ids
            if cid in self.idx_by_cid and len(self.idx_by_cid[cid]) > 0
        ]
        if len(self.hard_ids) < self.n_classes - 1:
            raise ValueError(
                f"可用 Hard ID 只有 {len(self.hard_ids)} 个，"
                f"不足以无放回选择 N-1={self.n_classes - 1} 个"
            )

        self.background_ids = [int(cid) for cid in background_ids]
        self.background_ratios = np.asarray(background_ratios, dtype=np.float64)
        for cid in self.background_ids:
            if cid not in self.idx_by_cid or len(self.idx_by_cid[cid]) == 0:
                raise ValueError(f"背景训练池缺少 ID_{cid}")

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Make episodic sampling reproducible and switch on background batches after warm-up."""
        self.current_epoch = int(epoch)
        self.rng = np.random.RandomState(self.base_seed + 1009 * self.current_epoch)

    def _sample_from_cid(self, cid: int, n: int) -> np.ndarray:
        pool = self.idx_by_cid[int(cid)]
        return self.rng.choice(pool, size=int(n), replace=(len(pool) < n)).astype(np.int64)

    def _background_counts(self, total: int) -> List[int]:
        raw = self.background_ratios * total
        counts = np.floor(raw).astype(int)
        remainder = int(total - counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            for i in order[:remainder]:
                counts[i] += 1
        return counts.tolist()

    def __iter__(self) -> Iterable[List[int]]:
        for step in range(self.batches_per_epoch):
            if self.current_epoch <= PURE_GE2E_WARMUP_EPOCHS:
                use_confusable = True
            elif self.alternate:
                use_confusable = (step % 2 == 0)
            else:
                use_confusable = bool(self.rng.rand() < self.p_confusable)

            batch: List[int] = []
            if use_confusable:
                selected_hard = self.rng.choice(
                    self.hard_ids,
                    size=self.n_classes - 1,
                    replace=False,
                ).tolist()
                selected_classes = [0] + [int(x) for x in selected_hard]
                for cid in selected_classes:
                    batch.extend(self._sample_from_cid(cid, self.m_utterances).tolist())
            else:
                batch.extend(self._sample_from_cid(0, self.m_utterances).tolist())
                total_bg = (self.n_classes - 1) * self.m_utterances
                counts = self._background_counts(total_bg)
                for cid, n in zip(self.background_ids, counts):
                    batch.extend(self._sample_from_cid(cid, n).tolist())
                # Randomize background order while keeping the first M wake samples contiguous.
                wake_part = batch[:self.m_utterances]
                bg_part = batch[self.m_utterances:]
                self.rng.shuffle(bg_part)
                batch = wake_part + bg_part

            expected = self.n_classes * self.m_utterances
            if len(batch) != expected:
                raise RuntimeError(f"Sampler batch size 异常: {len(batch)} != {expected}")
            yield batch


# ============================================================================
# 4. Augmentation
# ============================================================================
def dual_spec_augment(
    mfcc: torch.Tensor,
    logmel: torch.Tensor,
    p: float = SPEC_P,
    freq_mask_max_mfcc: int = FREQ_MASK_MFCC,
    freq_mask_max_mel: int = FREQ_MASK_MEL,
    time_mask_frac: float = TIME_MASK_FRAC,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mfcc.ndim != 4 or logmel.ndim != 4:
        raise ValueError("dual_spec_augment 输入必须为4D")
    if mfcc.shape[0] != logmel.shape[0] or mfcc.shape[-1] != logmel.shape[-1]:
        raise ValueError("MFCC与LogMel的batch/time维不一致")

    out_m = mfcc.clone()
    out_l = logmel.clone()
    b, _, fm, t = out_m.shape
    _, _, fl, _ = out_l.shape
    time_mask_max = max(2, int(t * time_mask_frac))

    for i in range(b):
        if torch.rand((), device=out_m.device).item() > p:
            continue

        t_len = int(torch.randint(0, time_mask_max + 1, (1,), device=out_m.device).item())
        if 0 < t_len < t:
            t0 = int(torch.randint(0, t - t_len + 1, (1,), device=out_m.device).item())
            out_m[i, :, :, t0:t0 + t_len] = 0.0
            out_l[i, :, :, t0:t0 + t_len] = 0.0

        fm_len = int(torch.randint(0, freq_mask_max_mfcc + 1, (1,), device=out_m.device).item())
        if 0 < fm_len < fm:
            f0 = int(torch.randint(0, fm - fm_len + 1, (1,), device=out_m.device).item())
            out_m[i, :, f0:f0 + fm_len, :] = 0.0

        fl_len = int(torch.randint(0, freq_mask_max_mel + 1, (1,), device=out_m.device).item())
        if 0 < fl_len < fl:
            f0 = int(torch.randint(0, fl - fl_len + 1, (1,), device=out_m.device).item())
            out_l[i, :, f0:f0 + fl_len, :] = 0.0

    return out_m, out_l


def gate_spec_augment(
    x: torch.Tensor,
    p: float = GATE_SPEC_P,
    freq_mask_max: int = GATE_FREQ_MASK_MAX,
    time_mask_frac: float = GATE_TIME_MASK_FRAC,
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"gate_spec_augment 需要4D输入，实际 {tuple(x.shape)}")
    out = x.clone()
    b, _, f, t = out.shape
    time_mask_max = max(1, int(t * time_mask_frac))

    for i in range(b):
        if torch.rand((), device=out.device).item() > p:
            continue
        t_len = int(torch.randint(0, time_mask_max + 1, (1,), device=out.device).item())
        if 0 < t_len < t:
            t0 = int(torch.randint(0, t - t_len + 1, (1,), device=out.device).item())
            out[i, :, :, t0:t0 + t_len] = 0.0
        f_len = int(torch.randint(0, freq_mask_max + 1, (1,), device=out.device).item())
        if 0 < f_len < f:
            f0 = int(torch.randint(0, f - f_len + 1, (1,), device=out.device).item())
            out[i, :, f0:f0 + f_len, :] = 0.0
    return out


# ============================================================================
# 5. KWS embedding encoder and Tiny Gate
# ============================================================================
def make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Return a GroupNorm whose group count always divides the channel count."""
    channels = int(channels)
    groups = min(int(max_groups), channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: Tuple[int, int] = (3, 3),
        s: Tuple[int, int] = (1, 1),
        p: Tuple[int, int] = (1, 1),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False),
            make_group_norm(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            make_group_norm(out_ch),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pw(self.dw(x)))


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = make_group_norm(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.dropout(x)
        return self.act(x + residual)


class AttentiveStatsPooling(nn.Module):
    def __init__(self, channels: int, att_hidden: int = 128):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(channels, att_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(att_hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.att(x), dim=-1)
        mean = torch.sum(weights * x, dim=-1)
        var = torch.sum(weights * (x - mean.unsqueeze(-1)).pow(2), dim=-1)
        std = torch.sqrt(var.clamp_min(1e-5))
        return torch.cat([mean, std], dim=1)


class DualStream_DSCNN_TCN_AttPool_GE2E(nn.Module):
    """Dual-stream encoder that outputs an L2-normalized embedding."""

    def __init__(
        self,
        base_ch: int = BASE_C,
        dropout: float = DROPOUT,
        use_temporal_1x5: bool = USE_TEMPORAL_1x5,
        tcn_layers: int = TCN_LAYERS,
        tcn_dilations: Sequence[int] = TCN_DILATIONS,
        att_hidden: int = ATT_HIDDEN,
        embedding_dim: int = EMBEDDING_DIM,
        mfcc_aux_scale: float = 1.0,
    ):
        super().__init__()
        if len(tcn_dilations) != tcn_layers:
            raise ValueError("TCN_DILATIONS长度必须等于TCN_LAYERS")

        self.base_ch = int(base_ch)
        self.dropout_p = float(dropout)
        self.use_temporal_1x5 = bool(use_temporal_1x5)
        self.mfcc_aux_scale = float(mfcc_aux_scale)

        self.logmel_branch = self._build_logmel_branch()
        self.mfcc_branch = self._build_mfcc_branch()

        c_last = self.base_ch * 4
        self.tcn_logmel = nn.Sequential(*[
            TemporalBlock(c_last, int(tcn_dilations[i]), self.dropout_p)
            for i in range(tcn_layers)
        ])
        self.tcn_mfcc = nn.Sequential(*[
            TemporalBlock(c_last, int(tcn_dilations[i]), self.dropout_p)
            for i in range(tcn_layers)
        ])

        self.pool_logmel = AttentiveStatsPooling(c_last, att_hidden)
        self.pool_mfcc = AttentiveStatsPooling(c_last, att_hidden)

        fused_dim = 4 * c_last
        hidden_dim = max(256, embedding_dim * 2)
        self.fuse_norm = nn.LayerNorm(fused_dim)
        self.embedding_head = nn.Sequential(
            nn.Dropout(self.dropout_p),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p * 0.5),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def _stem(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(1, self.base_ch, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            make_group_norm(self.base_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_p),
        )

    def _build_logmel_branch(self) -> nn.Sequential:
        layers: List[nn.Module] = [
            self._stem(),
            DepthwiseSeparableConv2d(self.base_ch, self.base_ch, dropout=self.dropout_p),
            DepthwiseSeparableConv2d(self.base_ch, self.base_ch, dropout=self.dropout_p),
        ]
        if self.use_temporal_1x5:
            layers.append(DepthwiseSeparableConv2d(
                self.base_ch, self.base_ch, k=(1, 5), p=(0, 2), dropout=self.dropout_p
            ))
        layers.extend([
            DepthwiseSeparableConv2d(self.base_ch, self.base_ch * 2, s=(2, 1), dropout=self.dropout_p),
            DepthwiseSeparableConv2d(self.base_ch * 2, self.base_ch * 2, dropout=self.dropout_p),
        ])
        if self.use_temporal_1x5:
            layers.append(DepthwiseSeparableConv2d(
                self.base_ch * 2, self.base_ch * 2, k=(1, 5), p=(0, 2), dropout=self.dropout_p
            ))
        layers.extend([
            DepthwiseSeparableConv2d(self.base_ch * 2, self.base_ch * 4, s=(2, 1), dropout=self.dropout_p),
            DepthwiseSeparableConv2d(self.base_ch * 4, self.base_ch * 4, dropout=self.dropout_p),
        ])
        return nn.Sequential(*layers)

    def _build_mfcc_branch(self) -> nn.Sequential:
        layers: List[nn.Module] = [
            self._stem(),
            DepthwiseSeparableConv2d(self.base_ch, self.base_ch, dropout=self.dropout_p),
            DepthwiseSeparableConv2d(self.base_ch, self.base_ch, dropout=self.dropout_p),
        ]
        if self.use_temporal_1x5:
            layers.append(DepthwiseSeparableConv2d(
                self.base_ch, self.base_ch, k=(1, 5), p=(0, 2), dropout=self.dropout_p
            ))
        layers.extend([
            DepthwiseSeparableConv2d(self.base_ch, self.base_ch * 2, dropout=self.dropout_p),
            DepthwiseSeparableConv2d(self.base_ch * 2, self.base_ch * 2, dropout=self.dropout_p),
        ])
        if self.use_temporal_1x5:
            layers.append(DepthwiseSeparableConv2d(
                self.base_ch * 2, self.base_ch * 2, k=(1, 5), p=(0, 2), dropout=self.dropout_p
            ))
        layers.extend([
            DepthwiseSeparableConv2d(self.base_ch * 2, self.base_ch * 4, dropout=self.dropout_p),
            DepthwiseSeparableConv2d(self.base_ch * 4, self.base_ch * 4, dropout=self.dropout_p),
        ])
        return nn.Sequential(*layers)

    @staticmethod
    def _encode(
        branch2d: nn.Module,
        tcn: nn.Module,
        pool: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = branch2d(x)
        x = x.mean(dim=2)
        x = tcn(x)
        return pool(x)

    def forward(self, mfcc: torch.Tensor, logmel: torch.Tensor) -> torch.Tensor:
        emb_l = self._encode(self.logmel_branch, self.tcn_logmel, self.pool_logmel, logmel)
        emb_m = self._encode(self.mfcc_branch, self.tcn_mfcc, self.pool_mfcc, mfcc)
        if self.mfcc_aux_scale != 1.0:
            emb_m = emb_m * self.mfcc_aux_scale
        fused = self.fuse_norm(torch.cat([emb_m, emb_l], dim=1))
        embedding = self.embedding_head(fused)
        return F.normalize(embedding, p=2, dim=1)


class TinyMFCCGate(nn.Module):
    def __init__(self, dropout: float = GATE_DROPOUT):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(60, 16, kernel_size=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )
        self.temporal_1 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=5, padding=2, groups=16, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv1d(16, 24, kernel_size=1, bias=False),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
        )
        self.temporal_2 = nn.Sequential(
            nn.Conv1d(24, 24, kernel_size=5, dilation=2, padding=4, groups=24, bias=False),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv1d(24, 24, kernel_size=1, bias=False),
            nn.GroupNorm(6, 24),
        )
        self.post_act = nn.SiLU()
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(48, 1))

    def forward(self, mfcc_gate: torch.Tensor) -> torch.Tensor:
        if mfcc_gate.ndim != 4 or mfcc_gate.shape[1:3] != (1, 60):
            raise ValueError(f"TinyMFCCGate需要[B,1,60,T]，实际{tuple(mfcc_gate.shape)}")
        x = mfcc_gate.squeeze(1)
        x = self.input_proj(x)
        x = self.temporal_1(x)
        residual = x
        x = self.temporal_2(x)
        x = self.post_act(x + residual)
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        return self.classifier(pooled)


# ============================================================================
# 6. Metric losses
# ============================================================================
class GE2ELoss(nn.Module):
    """Leave-one-out centroid GE2E softmax loss with a stable similarity scale."""

    def __init__(
        self,
        fixed_scale: float = GE2E_FIXED_SCALE,
        learn_scale: bool = LEARN_GE2E_SCALE,
        scale_min: float = GE2E_SCALE_MIN,
        scale_max: float = GE2E_SCALE_MAX,
    ):
        super().__init__()
        if fixed_scale <= 0 or scale_min <= 0 or scale_max <= scale_min:
            raise ValueError("GE2E scale configuration is invalid")
        self.learn_scale = bool(learn_scale)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        if self.learn_scale:
            # sigmoid(raw)=ratio gives a bounded learnable scale.
            ratio = (float(fixed_scale) - self.scale_min) / (self.scale_max - self.scale_min)
            ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
            raw = np.log(ratio / (1.0 - ratio))
            self.raw_scale = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        else:
            self.register_buffer("fixed_scale", torch.tensor(float(fixed_scale), dtype=torch.float32))

    def get_scale(self) -> torch.Tensor:
        if self.learn_scale:
            return self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(self.raw_scale)
        return self.fixed_scale

    def forward(self, embeddings: torch.Tensor):
        if embeddings.ndim != 3:
            raise ValueError(f"GE2ELoss需要[N,M,D]，实际{tuple(embeddings.shape)}")
        n_classes, n_utts, _ = embeddings.shape
        if n_classes < 2 or n_utts < 2:
            raise ValueError("GE2E要求N>=2且M>=2")

        embeddings = F.normalize(embeddings, p=2, dim=-1)
        full_centroids = F.normalize(embeddings.mean(dim=1), p=2, dim=-1)
        sums = embeddings.sum(dim=1, keepdim=True)
        loo_centroids = F.normalize((sums - embeddings) / float(n_utts - 1), p=2, dim=-1)

        sim_all = torch.einsum("nmd,kd->nmk", embeddings, full_centroids)
        sim_own = torch.sum(embeddings * loo_centroids, dim=-1)

        eye = torch.eye(n_classes, device=embeddings.device, dtype=torch.bool).view(n_classes, 1, n_classes)
        similarity = torch.where(eye, sim_own.unsqueeze(-1), sim_all)

        scale = self.get_scale()
        logits = scale * similarity
        targets = torch.arange(n_classes, device=embeddings.device).view(n_classes, 1).expand(n_classes, n_utts)
        loss = F.cross_entropy(logits.reshape(-1, n_classes), targets.reshape(-1))

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            episode_acc = (pred == targets).float().mean()
            own = similarity.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            masked = similarity.masked_fill(eye.expand(n_classes, n_utts, n_classes), -1e9)
            hardest_neg = masked.max(dim=-1).values
            margin = own - hardest_neg

        diagnostics = {
            "episode_acc": episode_acc,
            "own_sim": own.mean(),
            "hardest_neg_sim": hardest_neg.mean(),
            "episode_margin": margin.mean(),
            "scale": scale.detach(),
        }
        return loss, similarity, diagnostics


class BackgroundMetricLoss(nn.Module):
    """Push heterogeneous background embeddings below a cosine margin to wake centroid."""

    def __init__(self, neg_margin: float = BG_NEG_MARGIN, temperature: float = METRIC_TEMPERATURE):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature必须大于0")
        self.neg_margin = float(neg_margin)
        self.temperature = float(temperature)

    def forward(self, background_embeddings: torch.Tensor, wake_centroid: torch.Tensor):
        bg = F.normalize(background_embeddings, p=2, dim=-1)
        centroid = F.normalize(wake_centroid, p=2, dim=-1)
        similarity = torch.matmul(bg, centroid)
        loss = F.softplus((similarity - self.neg_margin) / self.temperature).mean()
        return loss, similarity



def pack_episode_embeddings(
    flat_embeddings: torch.Tensor,
    cids: torch.Tensor,
    expected_n: int,
    expected_m: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    unique_cids = torch.unique(cids, sorted=True)
    if unique_cids.numel() != expected_n:
        raise RuntimeError(f"GE2E episode期望{expected_n}类，实际{unique_cids.numel()}类: {unique_cids.tolist()}")

    groups: List[torch.Tensor] = []
    for cid in unique_cids:
        group = flat_embeddings[cids == cid]
        if group.shape[0] != expected_m:
            raise RuntimeError(f"ID_{int(cid.item())}期望{expected_m}条，实际{group.shape[0]}条")
        groups.append(group)
    return torch.stack(groups, dim=0), unique_cids


@torch.no_grad()
def embedding_collapse_stats(flat_embeddings: torch.Tensor) -> Dict[str, float]:
    """Measure angular collapse on one episodic batch."""
    z = F.normalize(flat_embeddings, p=2, dim=1)
    if z.shape[0] < 2:
        return {"pairwise_cos_mean": 1.0, "pairwise_cos_std": 0.0, "feature_std": 0.0}
    similarity = z @ z.transpose(0, 1)
    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    pairwise = similarity[mask]
    return {
        "pairwise_cos_mean": float(pairwise.mean().item()),
        "pairwise_cos_std": float(pairwise.std(unbiased=False).item()),
        "feature_std": float(z.std(dim=0, unbiased=False).mean().item()),
    }


def background_lambda_for_epoch(epoch: int) -> float:
    if epoch <= PURE_GE2E_WARMUP_EPOCHS:
        return 0.0
    progress = (epoch - PURE_GE2E_WARMUP_EPOCHS) / float(max(1, BACKGROUND_RAMP_EPOCHS))
    return float(LAMBDA_BG_METRIC_MAX * min(max(progress, 0.0), 1.0))


# ============================================================================
# 7. Metrics and threshold calibration
# ============================================================================
def cosine_to_unit_score(cosine: torch.Tensor) -> torch.Tensor:
    return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)


def compute_far_frr(scores: np.ndarray, labels: np.ndarray, threshold: float):
    preds = (scores >= threshold).astype(np.int32)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn + 1e-12)
    frr = fn / (fn + tp + 1e-12)
    return float(far), float(frr), cm


def choose_thr_fp_budget(scores: np.ndarray, labels: np.ndarray, target_far: float):
    negatives = scores[labels == 0]
    if len(negatives) == 0:
        return 1.0, 0.0
    fp_budget = int(np.floor(target_far * len(negatives)))
    if fp_budget <= 0:
        threshold = float(np.nextafter(np.max(negatives), np.inf))
        return threshold, float(np.mean(negatives >= threshold))
    sorted_desc = np.sort(negatives)[::-1]
    kth = sorted_desc[fp_budget - 1]
    threshold = float(np.nextafter(kth, np.inf))
    return threshold, float(np.mean(negatives >= threshold))


def choose_thr_roc(scores: np.ndarray, labels: np.ndarray, target_far: float):
    fpr, _, thresholds = roc_curve(labels, scores)
    valid = np.where(fpr <= target_far)[0]
    if len(valid) == 0:
        return 1.0, float(fpr[0])
    idx = valid[-1]
    return float(thresholds[idx]), float(fpr[idx])


def binary_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Optional[float]:
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))]).astype(np.int32)
    s = np.concatenate([pos_scores, neg_scores]).astype(np.float32)
    if np.allclose(s, s[0]):
        return 0.5
    fpr, tpr, _ = roc_curve(y, s)
    return float(auc(fpr, tpr))


def summarize_kws_metrics(scores: np.ndarray, labels: np.ndarray, cids: np.ndarray) -> Dict[str, Optional[float]]:
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    cids = np.asarray(cids, dtype=np.int32)
    wake = scores[labels == 1]
    hard = scores[(cids >= 1) & (cids <= 20)]

    hard_auc = binary_auc(wake, hard)
    wake_p05 = float(np.quantile(wake, 0.05)) if len(wake) else None
    hard_p95 = float(np.quantile(hard, 0.95)) if len(hard) else None
    hard_p99 = float(np.quantile(hard, 0.99)) if len(hard) else None
    margin = None if wake_p05 is None or hard_p95 is None else wake_p05 - hard_p95

    return {
        "hard_auc": hard_auc,
        "wake_p05": wake_p05,
        "hard_p95": hard_p95,
        "hard_p99": hard_p99,
        "wake_p05_minus_hard_p95": margin,
    }




def _score_array_from_outputs(outputs: Dict[str, np.ndarray]) -> np.ndarray:
    if "scores" in outputs:
        return np.asarray(outputs["scores"], dtype=np.float64)
    return np.asarray(outputs["probs"], dtype=np.float64)


def compute_eer_det_auc(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Return EER and DET-AUC as rates in [0,1]. Lower is better."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if len(np.unique(labels)) < 2:
        return float("inf"), float("inf")
    fpr, tpr, _ = roc_curve(labels, scores)
    frr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - frr)))
    eer = 0.5 * (float(fpr[idx]) + float(frr[idx]))
    det_auc = float(auc(fpr, frr))
    return eer, det_auc


def compute_hard_eer_det_auc(scores: np.ndarray, labels: np.ndarray, cids: np.ndarray) -> Tuple[float, float]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    cids = np.asarray(cids, dtype=np.int32).reshape(-1)
    hard_mask = (cids == 0) | ((cids >= 1) & (cids <= 20))
    return compute_eer_det_auc(scores[hard_mask], labels[hard_mask])


def frr_far_acceptance_metric(outputs: Dict[str, np.ndarray]) -> Dict[str, float]:
    """FRR--FAR-oriented validation metric for checkpoint/CL selection."""
    scores = _score_array_from_outputs(outputs)
    labels = np.asarray(outputs["labels"], dtype=np.int32).reshape(-1)
    cids = np.asarray(outputs["cids"], dtype=np.int32).reshape(-1)

    actual_fars, frrs, thresholds = far_frr_at_target_fars(scores, labels, CL_FARFRR_TARGETS)
    if len(frrs) == 0:
        weighted_frr = float("inf")
        primary_frr = float("inf")
        primary_actual_far = float("nan")
        primary_thr = float("nan")
    else:
        weights = np.asarray(CL_FARFRR_WEIGHTS, dtype=np.float64)
        weights = weights[:len(frrs)]
        weighted_frr = float(np.sum(frrs[:len(weights)] * weights) / np.sum(weights))
        primary_idx = int(np.argmin(np.abs(CL_FARFRR_TARGETS[:len(frrs)] - CL_PRIMARY_FAR)))
        primary_frr = float(frrs[primary_idx])
        primary_actual_far = float(actual_fars[primary_idx])
        primary_thr = float(thresholds[primary_idx])

    eer, det_auc = compute_eer_det_auc(scores, labels)
    hard_eer, hard_det_auc = compute_hard_eer_det_auc(scores, labels, cids)
    selection_score = (
        SELECT_WEIGHT_FRR_FAR * weighted_frr
        + SELECT_WEIGHT_EER * eer
        + SELECT_WEIGHT_DETAUC * det_auc
        + SELECT_WEIGHT_HARD_DETAUC * hard_det_auc
    )
    return {
        "selection_score": float(selection_score),
        "weighted_frr_far": float(weighted_frr),
        "primary_far": float(CL_PRIMARY_FAR),
        "primary_actual_far": float(primary_actual_far),
        "primary_frr": float(primary_frr),
        "primary_threshold": float(primary_thr),
        "eer": float(eer),
        "det_auc": float(det_auc),
        "hard_eer": float(hard_eer),
        "hard_det_auc": float(hard_det_auc),
    }


def update_metrics_with_farfr(metrics: Dict[str, Optional[float]], farfr: Dict[str, float]) -> Dict[str, Optional[float]]:
    metrics = dict(metrics)
    metrics.update({
        "selection_score": farfr["selection_score"],
        "weighted_frr_far": farfr["weighted_frr_far"],
        "primary_far": farfr["primary_far"],
        "primary_actual_far": farfr["primary_actual_far"],
        "primary_frr": farfr["primary_frr"],
        "eer": farfr["eer"],
        "det_auc": farfr["det_auc"],
        "hard_eer": farfr["hard_eer"],
        "hard_det_auc": farfr["hard_det_auc"],
        "eer_percent": 100.0 * farfr["eer"],
        "det_auc_percent": 100.0 * farfr["det_auc"],
        "hard_eer_percent": 100.0 * farfr["hard_eer"],
        "hard_det_auc_percent": 100.0 * farfr["hard_det_auc"],
    })
    return metrics


def save_epoch_monitor_figures(
    stage: str,
    epoch: int,
    outputs: Dict[str, np.ndarray],
    threshold: float,
    title_prefix: str,
) -> None:
    """Save the three compact monitoring figures requested by the user."""
    if not SAVE_EPOCH_MONITOR_FIGS:
        return
    scores = _score_array_from_outputs(outputs).astype(np.float32)
    labels = np.asarray(outputs["labels"], dtype=np.int32)
    cids = np.asarray(outputs["cids"], dtype=np.int32)
    preds = (scores >= float(threshold)).astype(np.int32)

    stage_dir = os.path.join(EPOCH_MONITOR_DIR, str(stage))
    os.makedirs(stage_dir, exist_ok=True)
    prefix = f"epoch{int(epoch):03d}"

    cm_path = os.path.join(stage_dir, f"{prefix}_confusion_matrix.png")
    group_path = os.path.join(stage_dir, f"{prefix}_group_error_bar.png")
    score_path = os.path.join(stage_dir, f"{prefix}_score_distribution.png")

    save_confusion_matrix_png(labels, preds, cm_path, f"{title_prefix} | epoch={epoch}, thr={threshold:.4f}")
    save_group_error_bar(labels, preds, cids, group_path, f"{title_prefix} group errors | epoch={epoch}")
    save_score_box(scores, cids, score_path, float(threshold), f"{title_prefix} score distribution | epoch={epoch}")

    # Latest copies make it easy to watch progress without opening a new file each epoch.
    for src_path, latest_name in [
        (cm_path, "latest_confusion_matrix.png"),
        (group_path, "latest_group_error_bar.png"),
        (score_path, "latest_score_distribution.png"),
    ]:
        try:
            shutil.copy2(src_path, os.path.join(stage_dir, latest_name))
        except OSError:
            pass


def print_farfr_monitor(prefix: str, farfr: Dict[str, float]) -> None:
    print(
        f"{prefix} Select={farfr['selection_score']:.6f} | "
        f"EER={100.0*farfr['eer']:.3f}% DET-AUC={100.0*farfr['det_auc']:.3f}% "
        f"Hard-DET-AUC={100.0*farfr['hard_det_auc']:.3f}% | "
        f"FRR@FAR{farfr['primary_far']:.4f}={100.0*farfr['primary_frr']:.3f}%"
    )

def cid_to_group(cid: int) -> str:
    if cid == 0:
        return "ID_0_wake"
    if 1 <= cid <= 20:
        return "ID_1-20_hard"
    if cid == 21:
        return "ID_21_Pep"
    if cid == 22:
        return "ID_22_Mask"
    if cid == 23:
        return "ID_23_Env"
    if cid == 24:
        return "ID_24_TTS"
    return "Other"


def fmt(value: Optional[float], ndigits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{ndigits}f}"


# ============================================================================
# 8. Enrollment centroid and output collection
# ============================================================================
def build_wake_enrollment_loader(dataset: FolderDualDataset) -> DataLoader:
    wake_indices = np.where(dataset.class_ids == 0)[0]
    if len(wake_indices) == 0:
        raise ValueError("训练集没有ID_0，无法构造Wake enrollment centroid")
    rng = np.random.RandomState(WAKE_ENROLL_SEED)
    n = min(WAKE_ENROLL_SAMPLES, len(wake_indices))
    selected = rng.choice(wake_indices, size=n, replace=False).astype(np.int64).tolist()
    subset = Subset(dataset, selected)
    return DataLoader(
        subset,
        batch_size=WAKE_ENROLL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )


@torch.no_grad()
def build_wake_centroid(model: nn.Module, enrollment_loader: DataLoader) -> torch.Tensor:
    model.eval()
    embeddings: List[torch.Tensor] = []
    for mfcc, logmel, _, _, cid, _ in enrollment_loader:
        if not torch.all(cid == 0):
            raise RuntimeError("Wake enrollment loader中出现非ID_0样本")
        mfcc = mfcc.to(DEVICE, non_blocking=True)
        logmel = logmel.to(DEVICE, non_blocking=True)
        embeddings.append(model(mfcc, logmel).detach())
    if not embeddings:
        raise RuntimeError("Wake enrollment embedding为空")
    centroid = F.normalize(torch.cat(embeddings, dim=0).mean(dim=0), p=2, dim=0)
    return centroid.detach()


@torch.no_grad()
def collect_kws_outputs(model: nn.Module, loader: DataLoader, wake_centroid: torch.Tensor):
    model.eval()
    centroid = F.normalize(wake_centroid.to(DEVICE), p=2, dim=0)
    scores, raw_cosines, labels, cids, sids = [], [], [], [], []

    for mfcc, logmel, _, y, cid, sid in loader:
        mfcc = mfcc.to(DEVICE, non_blocking=True)
        logmel = logmel.to(DEVICE, non_blocking=True)
        z = model(mfcc, logmel)
        cosine = torch.matmul(z, centroid)
        score = cosine_to_unit_score(cosine)

        scores.append(score.cpu().numpy().astype(np.float32))
        raw_cosines.append(cosine.cpu().numpy().astype(np.float32))
        labels.append(y.numpy().astype(np.int32).reshape(-1))
        cids.append(cid.numpy().astype(np.int32).reshape(-1))
        sids.extend(list(sid))

    return {
        "scores": np.concatenate(scores),
        "raw_cosines": np.concatenate(raw_cosines),
        "labels": np.concatenate(labels),
        "cids": np.concatenate(cids),
        "sids": np.asarray(sids, dtype=object),
    }


@torch.no_grad()
def collect_gate_outputs(gate_model: nn.Module, loader: DataLoader):
    gate_model.eval()
    probs, cids, sids = [], [], []
    for _, _, mfcc_gate, _, cid, sid in loader:
        mfcc_gate = mfcc_gate.to(DEVICE, non_blocking=True)
        probs.append(torch.sigmoid(gate_model(mfcc_gate).squeeze(1)).cpu().numpy().astype(np.float32))
        cids.append(cid.numpy().astype(np.int32).reshape(-1))
        sids.extend(list(sid))
    return np.concatenate(probs), np.concatenate(cids), np.asarray(sids, dtype=object)


@torch.no_grad()
def collect_system_outputs(
    kws_model: nn.Module,
    gate_model: nn.Module,
    loader: DataLoader,
    wake_centroid: torch.Tensor,
    gate_threshold: float,
):
    kws_model.eval()
    gate_model.eval()
    centroid = F.normalize(wake_centroid.to(DEVICE), p=2, dim=0)

    final_scores, kws_scores, raw_cosines, gate_probs = [], [], [], []
    labels_all, cids_all, sids_all = [], [], []

    for mfcc, logmel, mfcc_gate, y, cid, sid in loader:
        mfcc = mfcc.to(DEVICE, non_blocking=True)
        logmel = logmel.to(DEVICE, non_blocking=True)
        mfcc_gate = mfcc_gate.to(DEVICE, non_blocking=True)

        z = kws_model(mfcc, logmel)
        cosine = torch.matmul(z, centroid)
        score = cosine_to_unit_score(cosine)
        p_gate = torch.sigmoid(gate_model(mfcc_gate).squeeze(1))
        gate_pass = p_gate >= float(gate_threshold)
        final = torch.where(gate_pass, score, torch.zeros_like(score))

        final_scores.append(final.cpu().numpy().astype(np.float32))
        kws_scores.append(score.cpu().numpy().astype(np.float32))
        raw_cosines.append(cosine.cpu().numpy().astype(np.float32))
        gate_probs.append(p_gate.cpu().numpy().astype(np.float32))
        labels_all.append(y.numpy().astype(np.int32).reshape(-1))
        cids_all.append(cid.numpy().astype(np.int32).reshape(-1))
        sids_all.extend(list(sid))

    return {
        "scores": np.concatenate(final_scores),
        "kws_scores": np.concatenate(kws_scores),
        "raw_cosines": np.concatenate(raw_cosines),
        "gate_probs": np.concatenate(gate_probs),
        "labels": np.concatenate(labels_all),
        "cids": np.concatenate(cids_all),
        "sids": np.asarray(sids_all, dtype=object),
    }


def evaluate_kws_fixed_far(model: nn.Module, loader: DataLoader, wake_centroid: torch.Tensor):
    outputs = collect_kws_outputs(model, loader, wake_centroid)
    scores, labels = outputs["scores"], outputs["labels"]
    if THR_MODE == "fp_budget":
        threshold, _ = choose_thr_fp_budget(scores, labels, TARGET_FAR)
    else:
        threshold, _ = choose_thr_roc(scores, labels, TARGET_FAR)
    far, frr, cm = compute_far_frr(scores, labels, threshold)
    return {
        "thr": float(threshold),
        "target_far": TARGET_FAR,
        "far": far,
        "frr": frr,
        "cm": cm,
        "probs": scores,
        "raw_cosines": outputs["raw_cosines"],
        "labels": labels,
        "cids": outputs["cids"],
        "sids": outputs["sids"],
    }


# ============================================================================
# 9. Compact diagnostics
# ============================================================================
def ensure_best_dir(epoch: int, threshold: float) -> str:
    path = os.path.join(DIAG_DIR, f"best_epoch{epoch:03d}_thr{threshold:.4f}")
    os.makedirs(path, exist_ok=True)
    return path


def cleanup_root_visual_artifacts() -> None:
    os.makedirs(DIAG_DIR, exist_ok=True)
    for name in os.listdir(DIAG_DIR):
        path = os.path.join(DIAG_DIR, name)
        if os.path.isfile(path) and name.lower().endswith((".png", ".tsv", ".txt", ".csv")):
            try:
                os.remove(path)
            except OSError:
                pass
        elif os.path.isdir(path) and name in {"unfixed_epoch_diagnostics", "test_unfixed_diagnostics"}:
            shutil.rmtree(path, ignore_errors=True)


def save_confusion_matrix_png(labels: np.ndarray, preds: np.ndarray, path: str, title: str) -> None:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    row = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(6, 5))
    plt.imshow(row, aspect="auto")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True NoWake", "True Wake"])
    plt.title(title)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i,j]}\n{100*row[i,j]:.2f}%", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_group_error_bar(labels: np.ndarray, preds: np.ndarray, cids: np.ndarray, path: str, title: str) -> None:
    groups = np.asarray([cid_to_group(int(c)) for c in cids], dtype=object)
    rates, annotations = [], []
    for group in GROUP_ORDER:
        idx = groups == group
        n = int(idx.sum())
        if n == 0:
            rates.append(0.0)
            annotations.append("n=0")
            continue
        y, p = labels[idx], preds[idx]
        errors = int(np.sum(y != p))
        rate = errors / n
        rates.append(rate * 100.0)
        metric = "FRR" if group == "ID_0_wake" else "FA"
        annotations.append(f"{metric}={100*rate:.2f}%  ({errors}/{n})")

    y_pos = np.arange(len(GROUP_ORDER))
    plt.figure(figsize=(12, 5.5))
    plt.barh(y_pos, rates)
    plt.yticks(y_pos, GROUP_ORDER)
    plt.xlabel("Error rate (%)")
    plt.title(title)
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    max_rate = max(rates) if rates else 1.0
    plt.xlim(0, max(5.0, max_rate * 1.25 + 0.1))
    for i, (value, text) in enumerate(zip(rates, annotations)):
        plt.text(value + max(0.2, max_rate * 0.02), i, text, va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_score_box(scores: np.ndarray, cids: np.ndarray, path: str, threshold: float, title: str) -> None:
    groups = np.asarray([cid_to_group(int(c)) for c in cids], dtype=object)
    data, names = [], []
    for group in GROUP_ORDER:
        values = scores[groups == group]
        if len(values):
            data.append(values)
            names.append(group)
    plt.figure(figsize=(12, 5))
    plt.boxplot(data, tick_labels=names, showfliers=False)
    plt.axhline(threshold, linestyle="--", label=f"fixed-FAR threshold={threshold:.4f}")
    plt.axhline(FREE_THR, linestyle=":", label=f"free threshold={FREE_THR:.2f}")
    plt.ylabel("Wake-centroid similarity score")
    plt.title(title)
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_roc_pr(scores: np.ndarray, labels: np.ndarray, path: str, title: str) -> None:
    if len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.xlabel("FAR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()



def _quantile_compat(values: np.ndarray, q: float) -> float:
    """np.quantile wrapper compatible with both old and new NumPy versions."""
    values = np.asarray(values, dtype=np.float64)
    q = float(np.clip(q, 0.0, 1.0))
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        # NumPy < 1.22 uses interpolation instead of method.
        return float(np.quantile(values, q, interpolation="higher"))


def far_frr_at_target_fars(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fars: np.ndarray = FAR_FRR_TARGET_FARS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute DET-style FAR-FRR points by selecting one threshold per target FAR.

    Important:
        This does not run the model multiple times. It uses the scores already
        produced by one test/validation forward pass, then sweeps thresholds
        offline. For each target FAR, the threshold is chosen from the negative
        score distribution and the corresponding actual FAR and FRR are reported.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    target_fars = np.asarray(target_fars, dtype=np.float64).reshape(-1)

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    fars, frrs, thresholds = [], [], []
    for target_far in target_fars:
        target_far = float(np.clip(target_far, 0.0, 1.0))
        if target_far <= 0.0:
            # Strictly above the maximum negative score gives zero false alarms.
            threshold = float(np.nextafter(np.max(neg_scores), np.inf))
        elif target_far >= 1.0:
            threshold = float(np.nextafter(np.min(neg_scores), -np.inf))
        else:
            threshold = _quantile_compat(neg_scores, 1.0 - target_far)
            # Move the threshold slightly above the quantile to avoid including
            # too many tied negative scores when many samples have identical scores.
            threshold = float(np.nextafter(threshold, np.inf))

        far = float(np.mean(neg_scores >= threshold))
        frr = float(np.mean(pos_scores < threshold))
        fars.append(far)
        frrs.append(frr)
        thresholds.append(threshold)

    return np.asarray(fars), np.asarray(frrs), np.asarray(thresholds)


def save_far_frr_points_tsv(
    fars: np.ndarray,
    frrs: np.ndarray,
    thresholds: np.ndarray,
    target_fars: np.ndarray,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write("target_far\tachieved_far\tfrr\tthreshold\n")
        for target_far, far, frr, thr in zip(target_fars, fars, frrs, thresholds):
            file.write(f"{float(target_far):.8f}\t{float(far):.8f}\t{float(frr):.8f}\t{float(thr):.8f}\n")


def save_far_frr_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    path: str,
    title: str,
    model_label: str = FAR_FRR_MODEL_LABEL,
    target_fars: np.ndarray = FAR_FRR_TARGET_FARS,
    x_max: float = FAR_FRR_XMAX,
    y_max: float = FAR_FRR_YMAX,
    y_tick_step: float = FAR_FRR_YTICK_STEP,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Save the FAR-FRR curve like a DET-style figure."""
    fars, frrs, thresholds = far_frr_at_target_fars(scores, labels, target_fars)
    if len(fars) == 0:
        return fars, frrs, thresholds

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(7.2, 6.0))
    plt.plot(
        fars,
        frrs,
        marker="^",
        markersize=5,
        linewidth=1.8,
        label=model_label,
    )
    plt.xlabel("FAR", fontsize=14)
    plt.ylabel("FRR", fontsize=14)

    ax = plt.gca()
    ax.set_xlim(0.0, float(x_max))
    # Use a fixed fine-grained y-axis for low-FRR comparison.
    # If some points are above FAR_FRR_YMAX, they will be outside the displayed range
    # while still being saved in the TSV/NPZ artifacts.
    ax.set_ylim(0.0, float(y_max))
    ax.yaxis.set_major_locator(MultipleLocator(float(y_tick_step)))
    ax.yaxis.set_minor_locator(AutoMinorLocator(int(FAR_FRR_MINOR_YTICK_N)))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.18)

    plt.legend(fontsize=11, loc="best")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=FAR_FRR_DPI)
    plt.close()
    return fars, frrs, thresholds


def save_far_frr_artifacts(
    tag: str,
    scores: np.ndarray,
    labels: np.ndarray,
    cids: np.ndarray,
    out_dir: str,
    model_label: str = FAR_FRR_MODEL_LABEL,
) -> None:
    """
    Save all artifacts needed for the FAR-FRR figure:
        1) PNG curve
        2) TSV points with target FAR / achieved FAR / FRR / threshold
        3) NPZ raw scores, labels and cids for later multi-model comparison
    """
    os.makedirs(out_dir, exist_ok=True)
    curve_path = os.path.join(out_dir, f"{tag}_far_frr_curve.png")
    points_path = os.path.join(out_dir, f"{tag}_far_frr_points.tsv")
    scores_path = os.path.join(out_dir, f"{tag}_far_frr_scores.npz")

    fars, frrs, thresholds = save_far_frr_curve(
        scores=scores,
        labels=labels,
        path=curve_path,
        title=f"{tag} FAR-FRR Curve",
        model_label=model_label,
    )
    if len(fars):
        save_far_frr_points_tsv(fars, frrs, thresholds, FAR_FRR_TARGET_FARS, points_path)

    np.savez(
        scores_path,
        scores=np.asarray(scores, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        cids=np.asarray(cids, dtype=np.int32),
        target_fars=np.asarray(FAR_FRR_TARGET_FARS, dtype=np.float64),
        achieved_fars=np.asarray(fars, dtype=np.float64),
        frrs=np.asarray(frrs, dtype=np.float64),
        thresholds=np.asarray(thresholds, dtype=np.float64),
    )


def save_threshold_sweep(scores: np.ndarray, labels: np.ndarray, cids: np.ndarray, path: str, title: str) -> None:
    thresholds = np.linspace(0.0, 1.0, SWEEP_POINTS)
    masks = {
        "Global FAR": labels == 0,
        "Hard FA": (cids >= 1) & (cids <= 20),
        "Pep FA": cids == 21,
        "Mask FA": cids == 22,
        "Env FA": cids == 23,
        "TTS FA": cids == 24,
    }
    plt.figure(figsize=(10, 6))
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        values = [float(np.mean(scores[mask] >= thr)) for thr in thresholds]
        plt.plot(thresholds, values, label=name)
    wake = labels == 1
    if int(wake.sum()):
        frr = [float(np.mean(scores[wake] < thr)) for thr in thresholds]
        plt.plot(thresholds, frr, label="Global FRR")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_best_diagnostics(epoch: int, result: dict) -> None:
    """Save the same three figures for the best Stage-1 checkpoint."""
    best_dir = os.path.join(DIAG_DIR, "best_stage1_monitor")
    os.makedirs(best_dir, exist_ok=True)
    save_epoch_monitor_figures(
        stage="best_stage1",
        epoch=epoch,
        outputs=result,
        threshold=float(result["thr"]),
        title_prefix="Val GE2E KWS BEST",
    )


# ============================================================================
# 10. Checkpoints
# ============================================================================
def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def save_kws_checkpoint(
    path: str,
    model: nn.Module,
    ge2e_loss: GE2ELoss,
    wake_centroid: torch.Tensor,
    threshold: float,
    epoch: int,
    best_frr: float,
    metrics: Dict[str, Optional[float]],
) -> None:
    torch.save({
        "model": model.state_dict(),
        "ge2e_loss": ge2e_loss.state_dict(),
        "wake_centroid": wake_centroid.detach().cpu(),
        "kws_threshold": float(threshold),
        "target_far": float(TARGET_FAR),
        "epoch": int(epoch),
        "best_val_frr": float(best_frr),
        "metrics": {k: v for k, v in metrics.items()},
        "model_type": "DualStream_DSCNN_TCN_AttPool_GE2E",
        "embedding_dim": int(EMBEDDING_DIM),
        "score_definition": "unit_score=(cosine(embedding,wake_centroid)+1)/2",
        "training_loss": "stable_GE2E_leave_one_out + warmup_ramped_background_metric",
        "normalization": "GroupNorm",
        "ge2e_scale_mode": "learnable_bounded" if LEARN_GE2E_SCALE else "fixed",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, path)


def load_kws_checkpoint(path: str, model: nn.Module, ge2e_loss: Optional[GE2ELoss] = None):
    ckpt = safe_torch_load(path)
    model.load_state_dict(ckpt["model"], strict=True)
    if ge2e_loss is not None and "ge2e_loss" in ckpt:
        ge2e_loss.load_state_dict(ckpt["ge2e_loss"], strict=True)
    centroid = F.normalize(ckpt["wake_centroid"].float().to(DEVICE), p=2, dim=0)
    return ckpt, centroid


def save_gate_checkpoint(path: str, gate: nn.Module, threshold: float, epoch: int, metrics: dict, objective: float):
    torch.save({
        "gate": gate.state_dict(),
        "gate_threshold": float(threshold),
        "epoch": int(epoch),
        "metrics": {k: float(v) for k, v in metrics.items() if v is not None},
        "objective": float(objective),
        "gate_type": "TinyMFCCGate",
        "gate_definition": "ID23_reject_other_speech_pass",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, path)


def load_gate_checkpoint(path: str, gate: nn.Module):
    ckpt = safe_torch_load(path)
    gate.load_state_dict(ckpt["gate"], strict=True)
    return ckpt


def save_system_checkpoint(
    path: str,
    kws_model: nn.Module,
    gate_model: nn.Module,
    wake_centroid: torch.Tensor,
    gate_threshold: float,
    kws_threshold: float,
    kws_epoch: int,
    gate_epoch: int,
) -> None:
    torch.save({
        "kws_model": kws_model.state_dict(),
        "gate_model": gate_model.state_dict(),
        "wake_centroid": wake_centroid.detach().cpu(),
        "gate_threshold": float(gate_threshold),
        "kws_threshold": float(kws_threshold),
        "target_far": float(TARGET_FAR),
        "kws_epoch": int(kws_epoch),
        "gate_epoch": int(gate_epoch),
        "decision": "(p_gate>=gate_threshold) AND (unit_cosine_score>=kws_threshold)",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, path)


# ============================================================================
# 11. Tiny Gate sampling and calibration
# ============================================================================
class GateSubsetDataset(Dataset):
    def __init__(self, base_dataset: FolderDualDataset, indices: Sequence[int]):
        self.base_dataset = base_dataset
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        _, _, mfcc_gate, _, cid, sid = self.base_dataset[self.indices[item]]
        gate_y = torch.tensor(0.0 if cid == ENV_CID else 1.0, dtype=torch.float32)
        return mfcc_gate, gate_y, cid, sid


def sample_indices(rng: np.random.RandomState, pool: np.ndarray, n: int, name: str) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.int64)
    if len(pool) == 0:
        raise ValueError(f"Gate抽样池为空: {name}")
    replace = len(pool) < n
    if replace:
        print(f"警告: {name}数量={len(pool)}<目标={n}，采用有放回抽样")
    return rng.choice(pool, size=n, replace=replace).astype(np.int64)


def build_gate_epoch_indices(dataset: FolderDualDataset, epoch: int, seed: int = 42000) -> np.ndarray:
    rng = np.random.RandomState(seed + epoch)
    idx_by_cid = {
        int(cid): np.where(dataset.class_ids == cid)[0]
        for cid in np.unique(dataset.class_ids)
    }
    selected: List[int] = []
    selected.extend(sample_indices(rng, idx_by_cid.get(23, np.array([])), GATE_ENV_PER_EPOCH, "ID_23 Env"))
    selected.extend(sample_indices(rng, idx_by_cid.get(0, np.array([])), GATE_WAKE_PER_EPOCH, "ID_0 Wake"))
    selected.extend(sample_indices(rng, idx_by_cid.get(21, np.array([])), GATE_PEP_PER_EPOCH, "ID_21 Pep"))

    hard_each, remainder = divmod(GATE_HARD_PER_EPOCH, 20)
    for cid in range(1, 21):
        n = hard_each + (1 if cid <= remainder else 0)
        selected.extend(sample_indices(rng, idx_by_cid.get(cid, np.array([])), n, f"ID_{cid} Hard"))
    selected.extend(sample_indices(rng, idx_by_cid.get(22, np.array([])), GATE_MASK_PER_EPOCH, "ID_22 Mask"))
    selected.extend(sample_indices(rng, idx_by_cid.get(24, np.array([])), GATE_ID24_PER_EPOCH, "ID_24 TTS"))

    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    expected = (
        GATE_ENV_PER_EPOCH + GATE_WAKE_PER_EPOCH + GATE_PEP_PER_EPOCH
        + GATE_HARD_PER_EPOCH + GATE_MASK_PER_EPOCH + GATE_ID24_PER_EPOCH
    )
    if len(selected_array) != expected:
        raise RuntimeError(f"Gate抽样数量异常: {len(selected_array)} != {expected}")
    return selected_array


def gate_group_metrics(gate_probs: np.ndarray, cids: np.ndarray, threshold: float) -> Dict[str, Optional[float]]:
    gate_probs = np.asarray(gate_probs, dtype=np.float32)
    cids = np.asarray(cids, dtype=np.int32)

    def pass_rate(mask: np.ndarray) -> Optional[float]:
        return None if int(mask.sum()) == 0 else float(np.mean(gate_probs[mask] >= threshold))

    env_pass = pass_rate(cids == 23)
    return {
        "wake_pass": pass_rate(cids == 0),
        "hard_pass": pass_rate((cids >= 1) & (cids <= 20)),
        "pep_pass": pass_rate(cids == 21),
        "mask_pass": pass_rate(cids == 22),
        "env_pass": env_pass,
        "env_reject": None if env_pass is None else 1.0 - env_pass,
        "tts_pass": pass_rate(cids == 24),
    }


def choose_gate_threshold(gate_probs: np.ndarray, cids: np.ndarray):
    requirements = {
        "wake_pass": MIN_WAKE_PASS,
        "hard_pass": MIN_HARD_PASS,
        "pep_pass": MIN_PEP_PASS,
        "mask_pass": MIN_MASK_PASS,
        "tts_pass": MIN_TTS_PASS,
    }
    best_feasible = None
    best_fallback = None

    for threshold in np.linspace(0.0, 1.0, GATE_THR_POINTS):
        metrics = gate_group_metrics(gate_probs, cids, float(threshold))
        penalty = 0.0
        feasible = True
        for key, required in requirements.items():
            value = metrics[key]
            if value is None:
                continue
            shortfall = max(0.0, required - value)
            penalty += shortfall
            feasible = feasible and shortfall == 0.0

        env_reject = metrics["env_reject"] or 0.0
        fallback_obj = env_reject - 20.0 * penalty

        if feasible:
            candidate = (env_reject, float(threshold), metrics)
            if best_feasible is None or candidate[0] > best_feasible[0] + 1e-12 or (
                abs(candidate[0] - best_feasible[0]) <= 1e-12 and candidate[1] > best_feasible[1]
            ):
                best_feasible = candidate

        fallback = (fallback_obj, float(threshold), metrics)
        if best_fallback is None or fallback[0] > best_fallback[0]:
            best_fallback = fallback

    if best_feasible is not None:
        objective, threshold, metrics = best_feasible
        return threshold, metrics, float(objective), True

    objective, threshold, metrics = best_fallback
    print("警告: Gate严格通过率约束无解，使用惩罚式fallback阈值")
    return threshold, metrics, float(objective), False


def print_gate_metrics(prefix: str, threshold: float, metrics: dict, feasible: Optional[bool] = None) -> None:
    feasible_text = "" if feasible is None else f" | feasible={feasible}"
    print(
        f"{prefix} gate_thr={threshold:.4f}{feasible_text} | "
        f"WakePass={fmt(metrics.get('wake_pass'))} "
        f"HardPass={fmt(metrics.get('hard_pass'))} "
        f"PepPass={fmt(metrics.get('pep_pass'))} "
        f"MaskPass={fmt(metrics.get('mask_pass'))} "
        f"TTSPass={fmt(metrics.get('tts_pass'))} "
        f"EnvReject={fmt(metrics.get('env_reject'))}"
    )


# ============================================================================
# 12. Training stages
# ============================================================================
def build_kws_model() -> DualStream_DSCNN_TCN_AttPool_GE2E:
    return DualStream_DSCNN_TCN_AttPool_GE2E().to(DEVICE)


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def is_better_checkpoint(
    current_frr: float,
    current_metrics: Dict[str, Optional[float]],
    best_frr: float,
    best_hard_auc: float,
    best_margin: float,
) -> bool:
    eps = 1e-9
    hard_auc = current_metrics.get("hard_auc")
    margin = current_metrics.get("wake_p05_minus_hard_p95")
    hard_auc_value = -float("inf") if hard_auc is None else float(hard_auc)
    margin_value = -float("inf") if margin is None else float(margin)

    if current_frr < best_frr - eps:
        return True
    if abs(current_frr - best_frr) <= eps and hard_auc_value > best_hard_auc + eps:
        return True
    if (
        abs(current_frr - best_frr) <= eps
        and abs(hard_auc_value - best_hard_auc) <= eps
        and margin_value > best_margin + eps
    ):
        return True
    return False


def train_stage1_kws(train_dataset: FolderDualDataset, val_loader: DataLoader):
    print("\n========== Stage 1/3: Stable True GE2E embedding training ==========")
    sampler = GE2EMetricBatchSampler(
        train_dataset,
        n_classes=N_CLASSES,
        m_utterances=M_UTTERANCES,
        batches_per_epoch=BATCHES_PER_EPOCH,
        alternate=ALTERNATE_BATCH_TYPES,
        p_confusable=P_CONFUSABLE,
        seed=SEED + 100,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    enrollment_loader = build_wake_enrollment_loader(train_dataset)

    model = build_kws_model()
    ge2e_loss = GE2ELoss().to(DEVICE)
    bg_loss_fn = BackgroundMetricLoss().to(DEVICE)

    param_groups = [{
        "params": model.parameters(),
        "lr": KWS_LR,
        "weight_decay": KWS_WEIGHT_DECAY,
    }]
    if count_trainable_parameters(ge2e_loss) > 0:
        param_groups.append({
            "params": ge2e_loss.parameters(),
            "lr": GE2E_SCALE_LR,
            "weight_decay": 0.0,
        })
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, KWS_EPOCHS),
        eta_min=1e-6,
    )

    print(f"KWS encoder trainable params: {count_trainable_parameters(model):,}")
    print(f"GE2E learnable params: {count_trainable_parameters(ge2e_loss):,}")
    print(f"Episode shape: N={N_CLASSES}, M={M_UTTERANCES}, batch={N_CLASSES*M_UTTERANCES}")
    print(
        f"Stability: GroupNorm | dropout={DROPOUT:.2f} | SpecAugment p={SPEC_P:.2f} | "
        f"pure-GE2E warmup={PURE_GE2E_WARMUP_EPOCHS} epochs | scale={GE2E_FIXED_SCALE:.2f}"
    )

    cleanup_root_visual_artifacts()
    if os.path.exists(KWS_BEST_MODEL_PATH):
        os.remove(KWS_BEST_MODEL_PATH)

    best_frr = float("inf")
    best_hard_auc = -float("inf")
    best_margin = -float("inf")
    best_select = float("inf")
    best_epoch = -1
    best_threshold = 0.5
    no_improve = 0
    random_episode_acc = 1.0 / float(N_CLASSES)

    for epoch in range(1, KWS_EPOCHS + 1):
        sampler.set_epoch(epoch)
        model.train()
        ge2e_loss.train()
        lambda_bg = background_lambda_for_epoch(epoch)

        total_loss = 0.0
        ge2e_sum = 0.0
        bg_sum = 0.0
        ge2e_batches = 0
        bg_batches = 0
        episode_acc_sum = 0.0
        episode_margin_sum = 0.0
        own_sim_sum = 0.0
        neg_sim_sum = 0.0
        scale_sum = 0.0
        pairwise_mean_sum = 0.0
        pairwise_std_sum = 0.0
        feature_std_sum = 0.0

        for mfcc, logmel, _, _, cid, _ in train_loader:
            mfcc = mfcc.to(DEVICE, non_blocking=True)
            logmel = logmel.to(DEVICE, non_blocking=True)
            cid_t = cid.to(DEVICE).view(-1)
            mfcc, logmel = dual_spec_augment(mfcc, logmel)

            optimizer.zero_grad(set_to_none=True)
            embeddings = model(mfcc, logmel)
            non_wake_cids = cid_t[cid_t != 0]
            confusable_batch = bool(
                non_wake_cids.numel() > 0
                and torch.all((non_wake_cids >= 1) & (non_wake_cids <= 20)).item()
            )

            if confusable_batch:
                episode, _ = pack_episode_embeddings(
                    embeddings, cid_t, N_CLASSES, M_UTTERANCES
                )
                loss, _, diagnostics = ge2e_loss(episode)
                collapse = embedding_collapse_stats(embeddings)

                ge2e_sum += float(loss.item())
                ge2e_batches += 1
                episode_acc_sum += float(diagnostics["episode_acc"].item())
                episode_margin_sum += float(diagnostics["episode_margin"].item())
                own_sim_sum += float(diagnostics["own_sim"].item())
                neg_sim_sum += float(diagnostics["hardest_neg_sim"].item())
                scale_sum += float(diagnostics["scale"].item())
                pairwise_mean_sum += collapse["pairwise_cos_mean"]
                pairwise_std_sum += collapse["pairwise_cos_std"]
                feature_std_sum += collapse["feature_std"]
            else:
                if lambda_bg <= 0.0:
                    # This should not occur during pure-GE2E warm-up, but keep the guard explicit.
                    continue
                wake_z = embeddings[cid_t == 0]
                bg_z = embeddings[cid_t != 0]
                if wake_z.shape[0] < 2 or bg_z.shape[0] == 0:
                    raise RuntimeError("Background metric batch缺少足够的Wake或Background样本")
                wake_centroid_batch = F.normalize(wake_z.mean(dim=0), p=2, dim=0)
                loss_bg, _ = bg_loss_fn(bg_z, wake_centroid_batch)
                loss = lambda_bg * loss_bg
                bg_sum += float(loss_bg.item())
                bg_batches += 1

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Stage-1出现非有限loss: {loss.item()}")
            loss.backward()
            trainable = list(model.parameters()) + list(ge2e_loss.parameters())
            torch.nn.utils.clip_grad_norm_(trainable, CLIP_NORM)
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()
        wake_centroid = build_wake_centroid(model, enrollment_loader)
        val_result = evaluate_kws_fixed_far(model, val_loader, wake_centroid)
        metrics = summarize_kws_metrics(
            val_result["probs"], val_result["labels"], val_result["cids"]
        )
        farfr_metric = frr_far_acceptance_metric(val_result)
        metrics = update_metrics_with_farfr(metrics, farfr_metric)

        mean_loss = total_loss / max(1, ge2e_batches + bg_batches)
        mean_ge2e = ge2e_sum / max(1, ge2e_batches)
        mean_bg = bg_sum / max(1, bg_batches)
        mean_ep_acc = episode_acc_sum / max(1, ge2e_batches)
        mean_ep_margin = episode_margin_sum / max(1, ge2e_batches)
        mean_own = own_sim_sum / max(1, ge2e_batches)
        mean_neg = neg_sim_sum / max(1, ge2e_batches)
        mean_scale = scale_sum / max(1, ge2e_batches)
        mean_pairwise = pairwise_mean_sum / max(1, ge2e_batches)
        mean_pairwise_std = pairwise_std_sum / max(1, ge2e_batches)
        mean_feature_std = feature_std_sum / max(1, ge2e_batches)

        print(
            f"[GE2E-KWS] Epoch {epoch:03d} | loss={mean_loss:.5f} "
            f"ge2e={mean_ge2e:.5f} bg={mean_bg:.5f} lambda_bg={lambda_bg:.3f} | "
            f"EpAcc={mean_ep_acc:.4f} EpMargin={mean_ep_margin:.4f} "
            f"own={mean_own:.4f} neg={mean_neg:.4f} scale={mean_scale:.3f} | "
            f"pairCos={mean_pairwise:.4f}±{mean_pairwise_std:.4f} featStd={mean_feature_std:.5f} | "
            f"FRR@FAR{TARGET_FAR:g}={val_result['frr']:.4f} "
            f"(FAR={val_result['far']:.6f}, thr={val_result['thr']:.4f}) | "
            f"Hard-AUC={fmt(metrics['hard_auc'])} "
            f"Hard-p95={fmt(metrics['hard_p95'])} "
            f"tail-margin={fmt(metrics['wake_p05_minus_hard_p95'])}"
        )

        print_farfr_monitor("       [FRR-FAR monitor]", farfr_metric)
        save_epoch_monitor_figures(
            stage="stage1_kws",
            epoch=epoch,
            outputs=val_result,
            threshold=float(val_result["thr"]),
            title_prefix="Val GE2E KWS",
        )

        collapsed = (
            epoch >= COLLAPSE_CHECK_EPOCH
            and mean_ep_acc <= random_episode_acc + 0.02
            and mean_ep_margin <= 0.0
            and mean_pairwise >= COLLAPSE_PAIRWISE_COS
            and mean_feature_std <= COLLAPSE_FEATURE_STD
        )
        if collapsed:
            message = (
                "检测到embedding collapse：Episode准确率接近随机、margin非正、"
                "跨样本余弦接近1且特征维标准差过低。请先检查ID语义一致性和输入特征。"
            )
            if ABORT_ON_COLLAPSE:
                raise RuntimeError(message)
            print("警告:", message)

        eligible = (
            mean_ep_acc >= MIN_EPISODE_ACC_FOR_SELECTION
            and mean_ep_margin >= MIN_EPISODE_MARGIN_FOR_SELECTION
        )
        if not eligible:
            no_improve = 0
            print(
                f"  [checkpoint skipped] metric space not ready: "
                f"EpAcc={mean_ep_acc:.4f}<{MIN_EPISODE_ACC_FOR_SELECTION:.4f} or "
                f"EpMargin={mean_ep_margin:.4f}<{MIN_EPISODE_MARGIN_FOR_SELECTION:.4f}"
            )
            continue

        better = farfr_metric["selection_score"] < best_select - 1e-9
        if better:
            best_frr = val_result["frr"]
            best_hard_auc = -float("inf") if metrics["hard_auc"] is None else float(metrics["hard_auc"])
            best_margin = (
                -float("inf")
                if metrics["wake_p05_minus_hard_p95"] is None
                else float(metrics["wake_p05_minus_hard_p95"])
            )
            best_select = float(farfr_metric["selection_score"])
            best_epoch = epoch
            best_threshold = val_result["thr"]
            no_improve = 0
            save_kws_checkpoint(
                KWS_BEST_MODEL_PATH,
                model,
                ge2e_loss,
                wake_centroid,
                best_threshold,
                epoch,
                best_frr,
                metrics,
            )
            save_best_diagnostics(epoch, val_result)
            print(
                f"Stage-1 best updated: epoch={epoch}, Select={best_select:.6f}, "
                f"EER={100.0*farfr_metric['eer']:.3f}%, DET-AUC={100.0*farfr_metric['det_auc']:.3f}%, "
                f"Hard-DET-AUC={100.0*farfr_metric['hard_det_auc']:.3f}%, threshold={best_threshold:.4f}"
            )
        else:
            no_improve += 1
            print(f"  [KWS no improve {no_improve}/{KWS_EARLY_STOP_PATIENCE}]")
            if no_improve >= KWS_EARLY_STOP_PATIENCE:
                break

    if best_epoch < 0 or not os.path.exists(KWS_BEST_MODEL_PATH):
        raise RuntimeError(
            "Stage-1未形成可用GE2E度量空间，因此没有保存KWS checkpoint。"
            "请确认每个ID_1~ID_20内部是否对应单一固定词/短语。"
        )

    ckpt, wake_centroid = load_kws_checkpoint(KWS_BEST_MODEL_PATH, model, ge2e_loss)
    return model, ge2e_loss, wake_centroid, int(ckpt["epoch"]), float(ckpt["kws_threshold"])


def train_stage2_gate(train_dataset: FolderDualDataset, val_loader: DataLoader):
    print("\n========== Stage 2/3: Train independent Tiny Gate ==========")
    gate = TinyMFCCGate().to(DEVICE)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=GATE_LR, weight_decay=GATE_WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(reduction="none").to(DEVICE)
    print(f"Tiny Gate trainable params: {count_trainable_parameters(gate):,}")

    best_objective = -float("inf")
    best_epoch = -1
    best_threshold = 0.5
    no_improve = 0

    for epoch in range(1, GATE_EPOCHS + 1):
        indices = build_gate_epoch_indices(train_dataset, epoch)
        epoch_dataset = GateSubsetDataset(train_dataset, indices)
        generator = torch.Generator().manual_seed(90000 + epoch)
        loader = DataLoader(
            epoch_dataset,
            batch_size=GATE_BATCH_SIZE,
            shuffle=True,
            generator=generator,
            num_workers=GATE_NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            drop_last=False,
        )

        gate.train()
        loss_sum, n_seen = 0.0, 0
        for mfcc_gate, gate_y, cid, _ in loader:
            mfcc_gate = mfcc_gate.to(DEVICE, non_blocking=True)
            gate_y = gate_y.to(DEVICE, non_blocking=True)
            cid_t = cid.to(DEVICE).view(-1)
            mfcc_gate = gate_spec_augment(mfcc_gate)

            optimizer.zero_grad(set_to_none=True)
            logits = gate(mfcc_gate).squeeze(1)
            raw = criterion(logits, gate_y)
            weights = torch.ones_like(raw)
            weights[cid_t == ENV_CID] = GATE_ENV_LOSS_WEIGHT
            loss = (raw * weights).sum() / weights.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), CLIP_NORM)
            optimizer.step()

            batch_size = int(gate_y.numel())
            loss_sum += float(loss.item()) * batch_size
            n_seen += batch_size

        gate_probs, val_cids, _ = collect_gate_outputs(gate, val_loader)
        threshold, gate_metrics, objective, feasible = choose_gate_threshold(gate_probs, val_cids)
        print(f"[Gate] Epoch {epoch:03d} | loss={loss_sum/max(1,n_seen):.5f} | objective={objective:.5f}")
        print_gate_metrics("       ", threshold, gate_metrics, feasible)

        if objective > best_objective + 1e-9:
            best_objective = objective
            best_epoch = epoch
            best_threshold = threshold
            no_improve = 0
            save_gate_checkpoint(
                GATE_BEST_MODEL_PATH, gate, threshold, epoch, gate_metrics, objective
            )
            print(f"Stage-2 best updated: epoch={epoch}, gate_thr={threshold:.4f}")
        else:
            no_improve += 1
            print(f"  [Gate no improve {no_improve}/{GATE_EARLY_STOP_PATIENCE}]")
            if no_improve >= GATE_EARLY_STOP_PATIENCE:
                break

    ckpt = load_gate_checkpoint(GATE_BEST_MODEL_PATH, gate)
    return gate, int(ckpt["epoch"]), float(ckpt["gate_threshold"])



def write_farfr_metric_summary(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_final_farfr_outputs(
    kws_model: nn.Module,
    gate_model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    wake_centroid: torch.Tensor,
    gate_threshold: float,
    out_dir: str,
) -> List[Dict[str, object]]:
    """Final validation/test outputs for FRR--FAR, EER, DET-AUC and Hard-DET-AUC."""
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        outputs = collect_system_outputs(kws_model, gate_model, loader, wake_centroid, gate_threshold)
        scores = np.asarray(outputs["scores"], dtype=np.float32)
        labels = np.asarray(outputs["labels"], dtype=np.int32)
        cids = np.asarray(outputs["cids"], dtype=np.int32)
        farfr = frr_far_acceptance_metric(outputs)
        tag = f"{split_name}_system"
        split_dir = os.path.join(out_dir, "farfr_results", split_name)
        os.makedirs(split_dir, exist_ok=True)
        save_far_frr_artifacts(tag, scores, labels, cids, split_dir, model_label=FAR_FRR_MODEL_LABEL)
        save_epoch_monitor_figures(
            stage=f"final_{split_name}",
            epoch=0,
            outputs=outputs,
            threshold=float(farfr["primary_threshold"]) if np.isfinite(farfr["primary_threshold"]) else TARGET_FAR,
            title_prefix=f"Final {split_name} system",
        )
        row = {
            "split": split_name,
            "selection_score": farfr["selection_score"],
            "weighted_frr_far_percent": 100.0 * farfr["weighted_frr_far"],
            "primary_far": farfr["primary_far"],
            "primary_actual_far": farfr["primary_actual_far"],
            "primary_frr_percent": 100.0 * farfr["primary_frr"],
            "primary_threshold": farfr["primary_threshold"],
            "EER_percent": 100.0 * farfr["eer"],
            "DET_AUC_percent": 100.0 * farfr["det_auc"],
            "Hard_EER_percent": 100.0 * farfr["hard_eer"],
            "Hard_DET_AUC_percent": 100.0 * farfr["hard_det_auc"],
        }
        rows.append(row)
        print(
            f"[Final {split_name}] EER={row['EER_percent']:.4f}% | "
            f"DET-AUC={row['DET_AUC_percent']:.4f}% | "
            f"Hard-DET-AUC={row['Hard_DET_AUC_percent']:.4f}% | "
            f"FRR@FAR{farfr['primary_far']:.4f}={row['primary_frr_percent']:.4f}%"
        )
    write_farfr_metric_summary(os.path.join(out_dir, "farfr_metric_summary_val_test.csv"), rows)
    return rows

def save_system_diagnostics(
    tag: str,
    scores: np.ndarray,
    labels: np.ndarray,
    cids: np.ndarray,
    threshold: float,
    out_dir: str,
) -> None:
    preds = (scores >= threshold).astype(np.int32)
    save_confusion_matrix_png(
        labels, preds, os.path.join(out_dir, f"{tag}_confusion_matrix.png"),
        f"{tag} True-GE2E + Tiny hard gate",
    )
    save_group_error_bar(
        labels, preds, cids, os.path.join(out_dir, f"{tag}_group_error_bar.png"),
        f"{tag} group errors with Tiny hard gate",
    )
    save_score_box(
        scores, cids, os.path.join(out_dir, f"{tag}_score_distribution.png"),
        threshold, f"{tag} final score distribution",
    )
    save_roc_pr(scores, labels, os.path.join(out_dir, f"{tag}_roc_pr.png"), tag)
    save_far_frr_artifacts(
        tag, scores, labels, cids, out_dir,
        model_label="TrueGE2E + TinyGate",
    )
    save_threshold_sweep(
        scores, labels, cids, os.path.join(out_dir, f"{tag}_threshold_sweep.png"),
        f"{tag} threshold sweep",
    )


def calibrate_stage3_system(
    kws_model: nn.Module,
    gate_model: nn.Module,
    val_loader: DataLoader,
    wake_centroid: torch.Tensor,
    gate_threshold: float,
    kws_epoch: int,
    gate_epoch: int,
):
    print("\n========== Stage 3/3: Calibrate dual thresholds ==========")
    outputs = collect_system_outputs(
        kws_model, gate_model, val_loader, wake_centroid, gate_threshold
    )
    scores, labels = outputs["scores"], outputs["labels"]
    if THR_MODE == "fp_budget":
        kws_threshold, _ = choose_thr_fp_budget(scores, labels, TARGET_FAR)
    else:
        kws_threshold, _ = choose_thr_roc(scores, labels, TARGET_FAR)
    far, frr, cm = compute_far_frr(scores, labels, kws_threshold)
    gate_metrics = gate_group_metrics(outputs["gate_probs"], outputs["cids"], gate_threshold)
    print_gate_metrics("[System-Val]", gate_threshold, gate_metrics)
    print(f"[System-Val] kws_thr={kws_threshold:.4f} | FAR={far:.6f} | FRR={frr:.6f}")

    final_dir = os.path.join(DIAG_DIR, "final_farfr_results")
    os.makedirs(final_dir, exist_ok=True)
    with open(os.path.join(final_dir, "calibration_thresholds.txt"), "w", encoding="utf-8") as file:
        file.write(f"gate_threshold={gate_threshold:.8f}\n")
        file.write(f"kws_threshold={kws_threshold:.8f}\n")
        file.write(f"target_sample_far_for_threshold_calibration={TARGET_FAR:.8f}\n")
        file.write(f"sample_level_achieved_far={far:.8f}\n")
        file.write(f"sample_level_frr={frr:.8f}\n")
        file.write(f"confusion_matrix={cm.tolist()}\n")
        file.write("primary_metric=FRR--FAR / EER / DET-AUC / Hard-DET-AUC\n")
        for key, value in gate_metrics.items():
            file.write(f"{key}={'NA' if value is None else f'{value:.8f}'}\n")

    save_system_checkpoint(
        SYSTEM_BEST_MODEL_PATH,
        kws_model,
        gate_model,
        wake_centroid,
        gate_threshold,
        kws_threshold,
        kws_epoch,
        gate_epoch,
    )
    return float(kws_threshold), final_dir


def strict_test_system(
    kws_model: nn.Module,
    gate_model: nn.Module,
    test_loader: DataLoader,
    wake_centroid: torch.Tensor,
    gate_threshold: float,
    kws_threshold: float,
    final_dir: str,
) -> List[Dict[str, object]]:
    """Strict test with fixed validation thresholds. Only FRR-FAR outputs are generated."""
    print("\n========== Strict Test: fixed validation thresholds ==========")
    outputs = collect_system_outputs(
        kws_model, gate_model, test_loader, wake_centroid, gate_threshold
    )
    scores = np.asarray(outputs["scores"], dtype=np.float32)
    labels = np.asarray(outputs["labels"], dtype=np.int32)
    cids = np.asarray(outputs["cids"], dtype=np.int32)

    far, frr, cm = compute_far_frr(scores, labels, kws_threshold)
    gate_metrics = gate_group_metrics(outputs["gate_probs"], cids, gate_threshold)
    print_gate_metrics("[System-Test]", gate_threshold, gate_metrics)
    print(f"[System-Test] kws_thr={kws_threshold:.4f} | sample-level FAR={far:.6f} | FRR={frr:.6f}")
    print("[System-Test] CM [[TN,FP],[FN,TP]] =", cm.tolist())

    strict_dir = os.path.join(final_dir, "strict_test")
    os.makedirs(strict_dir, exist_ok=True)
    save_far_frr_artifacts("test_system", scores, labels, cids, strict_dir, model_label=FAR_FRR_MODEL_LABEL)

    row = {
        "split": "test",
        "score": "system",
        "threshold": float(kws_threshold),
        "sample_far": float(far),
        "sample_frr": float(frr),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    return [row]


# ============================================================================
# 13. Main
# ============================================================================

# ============================================================================
# 13. Continual false-alarm replay
# ============================================================================
def _stable_source_key(sid: str) -> str:
    """Group likely sibling augmentations together when validation is split."""
    sid = sid.replace("\\", "/")
    folder, name = sid.split("/", 1)
    stem = os.path.splitext(os.path.basename(name))[0]
    first_token = stem.split("_")[0]
    return f"{folder}/{first_token}"


def split_validation_for_adaptation(
    dataset: FolderDualDataset,
    adapt_fraction: float,
    seed: int,
) -> Tuple[Subset, Subset]:
    if not 0.0 < adapt_fraction < 1.0:
        raise ValueError("VAL_ADAPT_FRACTION必须在(0,1)内")

    groups: Dict[Tuple[int, str], List[int]] = {}
    for idx, (cid, path) in enumerate(zip(dataset.class_ids, dataset.file_paths)):
        folder = os.path.basename(os.path.dirname(path))
        sid = f"{folder}/{os.path.basename(path)}"
        groups.setdefault((int(cid), _stable_source_key(sid)), []).append(idx)

    adapt_indices: List[int] = []
    calib_indices: List[int] = []
    for (cid, _), indices in groups.items():
        token = f"{seed}|{cid}|{indices[0]}".encode("utf-8")
        value = int(hashlib.sha1(token).hexdigest()[:8], 16) / float(0xFFFFFFFF)
        (adapt_indices if value < adapt_fraction else calib_indices).extend(indices)

    # Fallback to stratified sample-level split if source grouping is too coarse.
    adapt_arr = np.asarray(adapt_indices, dtype=np.int64)
    calib_arr = np.asarray(calib_indices, dtype=np.int64)
    for cid in np.unique(dataset.class_ids):
        if int(np.sum(dataset.class_ids[adapt_arr] == cid)) == 0 or int(np.sum(dataset.class_ids[calib_arr] == cid)) == 0:
            rng = np.random.RandomState(seed)
            adapt_indices, calib_indices = [], []
            for c in np.unique(dataset.class_ids):
                idx = np.where(dataset.class_ids == c)[0]
                rng.shuffle(idx)
                n_adapt = min(max(1, int(round(len(idx) * adapt_fraction))), max(1, len(idx) - 1))
                adapt_indices.extend(idx[:n_adapt].tolist())
                calib_indices.extend(idx[n_adapt:].tolist())
            break

    print(
        f"Validation split: adapt={len(adapt_indices)} | calibration={len(calib_indices)} "
        f"(adapt_fraction={adapt_fraction:.2f})"
    )
    return Subset(dataset, adapt_indices), Subset(dataset, calib_indices)


def build_adaptation_and_calibration_sets(
    val_dataset: FolderDualDataset,
) -> Tuple[Dataset, Dataset, str]:
    if os.path.isdir(ADAPT_FEATURE_DIR):
        print(f"使用独立adaptation集: {ADAPT_FEATURE_DIR}")
        return FolderDualDataset(ADAPT_FEATURE_DIR), val_dataset, ADAPT_WAV_DIR

    if not ALLOW_SPLIT_VAL_FOR_ADAPT:
        raise FileNotFoundError(
            f"独立adaptation目录不存在: {ADAPT_FEATURE_DIR}，且未允许拆分validation"
        )

    print("警告: 未发现独立adaptation集，将validation按固定规则拆分为mining/calibration两部分。")
    adapt_set, calib_set = split_validation_for_adaptation(
        val_dataset, VAL_ADAPT_FRACTION, VAL_SPLIT_SEED
    )
    return adapt_set, calib_set, os.path.join(RAW_WAV_BASE_DIR, "val")


def system_metrics_from_outputs(outputs: dict, threshold: float) -> dict:
    far, frr, cm = compute_far_frr(outputs["scores"], outputs["labels"], threshold)
    return {"far": far, "frr": frr, "cm": cm}


def mine_system_false_alarms(
    outputs: dict,
    gate_threshold: float,
    kws_threshold: float,
    seen_sids: set,
) -> List[dict]:
    labels = np.asarray(outputs["labels"], dtype=np.int32)
    cids = np.asarray(outputs["cids"], dtype=np.int32)
    scores = np.asarray(outputs["scores"], dtype=np.float32)
    kws_scores = np.asarray(outputs["kws_scores"], dtype=np.float32)
    raw_cosines = np.asarray(outputs["raw_cosines"], dtype=np.float32)
    gate_probs = np.asarray(outputs["gate_probs"], dtype=np.float32)
    sids = np.asarray(outputs["sids"], dtype=object)

    mask = (
        (labels == 0)
        & (cids != 0)
        & (gate_probs >= float(gate_threshold))
        & (kws_scores >= float(kws_threshold))
    )
    indices = np.where(mask)[0]
    indices = indices[np.argsort(-scores[indices])]

    cases: List[dict] = []
    per_id = {}
    for idx in indices:
        sid = str(sids[idx])
        cid = int(cids[idx])
        if sid in seen_sids:
            continue
        if per_id.get(cid, 0) >= REPLAY_TOPK_FP_PER_ID:
            continue
        cases.append({
            "sid": sid,
            "cid": cid,
            "score": float(scores[idx]),
            "kws_score": float(kws_scores[idx]),
            "raw_cosine": float(raw_cosines[idx]),
            "gate_prob": float(gate_probs[idx]),
            "kws_threshold": float(kws_threshold),
            "gate_threshold": float(gate_threshold),
            "margin": float(kws_scores[idx] - kws_threshold),
        })
        per_id[cid] = per_id.get(cid, 0) + 1
        if len(cases) >= REPLAY_TOPK_FP_TOTAL:
            break
    return cases


def _list_noise_files(noise_dir: str) -> List[str]:
    result: List[str] = []
    if os.path.isdir(noise_dir):
        for pattern in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
            result.extend(glob.glob(os.path.join(noise_dir, "**", pattern), recursive=True))
    return sorted(result)


def _load_fixed_audio(path: str, sr: int, target_samples: int, rng: np.random.RandomState) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = y.astype(np.float32)
    if len(y) > target_samples:
        start = int(rng.randint(0, len(y) - target_samples + 1))
        y = y[start:start + target_samples]
    elif len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    return y.astype(np.float32)


def _safe_peak_normalize(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    y = np.nan_to_num(y.astype(np.float32))
    maximum = float(np.max(np.abs(y))) if len(y) else 0.0
    if maximum > peak and maximum > 1e-8:
        y = y / maximum * peak
    return y.astype(np.float32)


def _time_shift_zeropad(y: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return y.copy()
    output = np.zeros_like(y)
    if shift > 0:
        output[shift:] = y[:-shift]
    else:
        output[:shift] = y[-shift:]
    return output


def _mix_noise_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_rms = float(np.sqrt(np.mean(clean ** 2) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(noise ** 2) + 1e-12))
    if noise_rms < 1e-8:
        return clean
    noise = noise * (clean_rms / (10.0 ** (snr_db / 20.0)) / noise_rms)
    return (clean + noise).astype(np.float32)


def sid_to_wav_path(sid: str, wav_root: str) -> Optional[str]:
    sid = sid.replace("\\", "/")
    parts = sid.split("/", 1)
    if len(parts) != 2:
        return None
    folder, filename = parts
    stem = os.path.splitext(os.path.basename(filename))[0]
    direct = os.path.join(wav_root, folder, stem + ".wav")
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(wav_root, folder, "**", stem + ".wav"), recursive=True)
    if matches:
        return matches[0]
    matches = glob.glob(os.path.join(wav_root, "**", stem + ".wav"), recursive=True)
    return matches[0] if matches else None


def write_replay_audio(
    src_wav: str,
    dst_wav: str,
    noise_files: Sequence[str],
    rng: np.random.RandomState,
    original: bool,
) -> None:
    target_samples = int(REPLAY_TARGET_SR * REPLAY_TARGET_DUR)
    y = _load_fixed_audio(src_wav, REPLAY_TARGET_SR, target_samples, rng)
    if original:
        output = y
    else:
        gain_db = float(rng.uniform(*REPLAY_GAIN_DB_RANGE))
        output = y * (10.0 ** (gain_db / 20.0))
        max_shift = int(REPLAY_TARGET_SR * REPLAY_SHIFT_MS / 1000.0)
        output = _time_shift_zeropad(output, int(rng.randint(-max_shift, max_shift + 1)))
        if noise_files:
            noise_path = str(rng.choice(noise_files))
            noise = _load_fixed_audio(noise_path, REPLAY_TARGET_SR, target_samples, rng)
            output = _mix_noise_at_snr(output, noise, float(rng.uniform(*REPLAY_SNR_DB_RANGE)))
    os.makedirs(os.path.dirname(dst_wav), exist_ok=True)
    sf.write(dst_wav, _safe_peak_normalize(output), REPLAY_TARGET_SR)


def generate_replay_wavs(
    cases: Sequence[dict],
    source_wav_root: str,
    round_id: int,
) -> Tuple[str, str]:
    rng = np.random.RandomState(70000 + round_id)
    noises = _list_noise_files(POINTSOURCE_NOISE_DIR)
    round_dir = os.path.join(REPLAY_WAV_ROOT, f"round_{round_id:03d}")
    os.makedirs(round_dir, exist_ok=True)
    manifest = os.path.join(round_dir, "manifest.tsv")

    written = 0
    with open(manifest, "w", encoding="utf-8") as file:
        file.write(
            "round\tcid\tsid\tscore\tkws_score\traw_cosine\tgate_prob\tmargin\t"
            "src_wav\tdst_wav\tmode\n"
        )
        for case in cases:
            cid = int(case["cid"])
            if cid == 0:
                continue
            src = sid_to_wav_path(case["sid"], source_wav_root)
            if src is None:
                print(f"警告: 找不到原始wav: {case['sid']}")
                continue
            folder = os.path.join(round_dir, f"ID_{cid}")
            stem = os.path.splitext(os.path.basename(src))[0]
            modes = ["original"] if REPLAY_INCLUDE_ORIGINAL else []
            modes += ["aug"] * int(REPLAY_AUG_PER_CASE)
            for aug_idx, mode in enumerate(modes):
                dst = os.path.join(
                    folder,
                    f"r{round_id:03d}_ID{cid:02d}_p{case['score']:.4f}_{aug_idx:02d}_{stem}.wav",
                )
                write_replay_audio(src, dst, noises, rng, original=(mode == "original"))
                file.write(
                    f"{round_id}\t{cid}\t{case['sid']}\t{case['score']:.6f}\t"
                    f"{case['kws_score']:.6f}\t{case['raw_cosine']:.6f}\t"
                    f"{case['gate_prob']:.6f}\t{case['margin']:.6f}\t"
                    f"{src}\t{dst}\t{mode}\n"
                )
                written += 1
    print(f"Replay wav生成完成: {round_dir} | written={written}")
    return round_dir, manifest


def run_replay_feature_extraction(round_id: int) -> bool:
    if not AUTO_EXTRACT_REPLAY_FEATURES:
        print(
            f'请手动运行: python "{FEATURE_EXTRACT_SCRIPT}" '
            f"--replay-only --round-id {round_id}"
        )
        return False
    if not os.path.isfile(FEATURE_EXTRACT_SCRIPT):
        raise FileNotFoundError(f"Replay特征提取脚本不存在: {FEATURE_EXTRACT_SCRIPT}")
    command = [
        sys.executable,
        FEATURE_EXTRACT_SCRIPT,
        "--replay-only",
        "--round-id",
        str(round_id),
        "--replay-wav-root",
        REPLAY_WAV_ROOT,
        "--replay-feature-root",
        REPLAY_FEATURE_ROOT,
    ]
    print("运行Replay特征提取:", " ".join(command))
    subprocess.run(command, check=True)
    return True


def replay_feature_round_dir(round_id: int) -> str:
    return os.path.join(REPLAY_FEATURE_ROOT, f"round_{round_id:03d}")


def count_replay_wavs(round_id: int) -> int:
    """Count generated replay wavs for one replay round."""
    round_dir = os.path.join(REPLAY_WAV_ROOT, f"round_{round_id:03d}")
    if not os.path.isdir(round_dir):
        return 0
    return len(glob.glob(os.path.join(round_dir, "ID_*", "*.wav")))


def cosine_stability_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()


def wake_recall_margin_loss(
    wake_embeddings: torch.Tensor,
    reference_centroid: torch.Tensor,
    threshold: float,
    margin: float = CL_WAKE_POS_MARGIN,
    temperature: float = CL_WAKE_RECALL_TEMPERATURE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    wake_embeddings = F.normalize(wake_embeddings, p=2, dim=-1)
    centroid = F.normalize(reference_centroid.to(wake_embeddings.device), p=2, dim=0)
    cosine = torch.matmul(wake_embeddings, centroid)
    score = cosine_to_unit_score(cosine)
    target_score = float(threshold) + float(margin)
    loss = F.softplus((target_score - score) / float(temperature)).mean()
    return loss, score


def finetune_kws_with_replay(
    model: nn.Module,
    ge2e_loss: GE2ELoss,
    original_dataset: FolderDualDataset,
    replay_roots: Sequence[str],
    val_loader: DataLoader,
    previous_wake_centroid: torch.Tensor,
    round_id: int,
    gate_model: Optional[nn.Module] = None,
    gate_threshold: Optional[float] = None,
) -> Tuple[nn.Module, GE2ELoss, torch.Tensor, bool]:
    print(f"\n========== CL Round {round_id}: True-GE2E replay fine-tuning ==========")
    dataset = MultiRootFolderDualDataset(TRAIN_DATASET_DIR, replay_roots)
    sampler = ReplayAwareGE2EMetricBatchSampler(dataset, seed=SEED + 5000 + round_id)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    enrollment_loader = build_wake_enrollment_loader(original_dataset)

    teacher = copy.deepcopy(model).to(DEVICE).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    baseline_centroid = build_wake_centroid(model, enrollment_loader)
    baseline_result = evaluate_kws_fixed_far(model, val_loader, baseline_centroid)
    baseline_metrics = summarize_kws_metrics(
        baseline_result["probs"], baseline_result["labels"], baseline_result["cids"]
    )
    best_frr = float(baseline_result["frr"])
    best_hard_auc = -float("inf") if baseline_metrics["hard_auc"] is None else float(baseline_metrics["hard_auc"])
    best_margin = (
        -float("inf")
        if baseline_metrics["wake_p05_minus_hard_p95"] is None
        else float(baseline_metrics["wake_p05_minus_hard_p95"])
    )

    baseline_farfr_metric =    baseline_farfr_metric = frr_far_acceptance_metric(baseline_result)
    if USE_CL_FARFRR_ACCEPTANCE and gate_model is not None and gate_threshold is not None:
        baseline_system_outputs = collect_system_outputs(
            model, gate_model, val_loader, baseline_centroid, float(gate_threshold)
        )
        baseline_farfr_metric = frr_far_acceptance_metric(baseline_system_outputs)
    print_farfr_monitor("[CL-KWS baseline FAR-FRR]", baseline_farfr_metric)
    best_select = float(baseline_farfr_metric["selection_score"])

    round_dir = os.path.join(DIAG_DIR, f"cl_round_{round_id:03d}")
    os.makedirs(round_dir, exist_ok=True)
    candidate_path = os.path.join(round_dir, "best_cl_kws.pth")
    save_kws_checkpoint(
        candidate_path, model, ge2e_loss, baseline_centroid,
        baseline_result["thr"], 0, best_frr, baseline_metrics
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CL_KWS_LR, weight_decay=CL_KWS_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, CL_KWS_EPOCHS), eta_min=1e-6
    )
    background_loss = BackgroundMetricLoss()
    replay_margin_loss = BackgroundMetricLoss(
        neg_margin=CL_REPLAY_NEG_MARGIN,
        temperature=CL_REPLAY_TEMPERATURE,
    )
    no_improve = 0
    accepted = False

    for epoch in range(1, CL_KWS_EPOCHS + 1):
        sampler.set_epoch(epoch)
        model.train()
        loss_sum = ge2e_sum = bg_sum = replay_sum = stability_sum = anchor_sum = wake_recall_sum = 0.0
        ge2e_batches = bg_batches = n_batches = 0
        episode_acc_sum = episode_margin_sum = 0.0

        for mfcc, logmel, _, _, cid, _, is_replay in loader:
            mfcc = mfcc.to(DEVICE, non_blocking=True)
            logmel = logmel.to(DEVICE, non_blocking=True)
            cid_t = cid.to(DEVICE).view(-1)
            replay_t = is_replay.to(DEVICE).bool().view(-1)
            mfcc, logmel = dual_spec_augment(mfcc, logmel)

            optimizer.zero_grad(set_to_none=True)
            z = model(mfcc, logmel)
            with torch.no_grad():
                teacher_z = teacher(mfcc, logmel)

            nonwake = cid_t[cid_t != 0]
            is_confusable = bool(
                nonwake.numel() > 0
                and torch.all((nonwake >= 1) & (nonwake <= 20)).item()
            )
            wake_z = z[cid_t == 0]
            if wake_z.shape[0] < 2:
                raise RuntimeError("CL batch缺少Wake")
            wake_centroid_batch = F.normalize(wake_z.mean(dim=0), p=2, dim=0)

            if is_confusable:
                episode, _ = pack_episode_embeddings(z, cid_t, N_CLASSES, M_UTTERANCES)
                base_loss, _, diagnostics = ge2e_loss(episode)
                ge2e_sum += float(base_loss.item())
                ge2e_batches += 1
                episode_acc_sum += float(diagnostics["episode_acc"].item())
                episode_margin_sum += float(diagnostics["episode_margin"].item())
            else:
                bg_z = z[cid_t != 0]
                base_loss, _ = background_loss(bg_z, wake_centroid_batch)
                base_loss = LAMBDA_BG_METRIC_MAX * base_loss
                bg_sum += float(base_loss.item())
                bg_batches += 1

            replay_mask = replay_t & (cid_t != 0)
            if replay_mask.any():
                replay_loss, _ = replay_margin_loss(z[replay_mask], wake_centroid_batch)
            else:
                replay_loss = z.sum() * 0.0

            original_mask = ~replay_t
            if original_mask.any():
                stability = cosine_stability_loss(z[original_mask], teacher_z[original_mask])
            else:
                stability = z.sum() * 0.0

            centroid_anchor = 1.0 - torch.sum(
                F.normalize(wake_centroid_batch, p=2, dim=0)
                * F.normalize(previous_wake_centroid.to(DEVICE), p=2, dim=0)
            )

            wake_recall_loss, _ = wake_recall_margin_loss(
                wake_z,
                previous_wake_centroid,
                threshold=float(baseline_result["thr"]),
            )

            loss = (
                base_loss
                + CL_LAMBDA_REPLAY_MARGIN * replay_loss
                + CL_LAMBDA_WAKE_RECALL * wake_recall_loss
                + CL_LAMBDA_STABILITY * stability
                + CL_LAMBDA_CENTROID_ANCHOR * centroid_anchor
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("CL KWS出现非有限loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()

            n_batches += 1
            loss_sum += float(loss.item())
            replay_sum += float(replay_loss.item())
            stability_sum += float(stability.item())
            anchor_sum += float(centroid_anchor.item())
            wake_recall_sum += float(wake_recall_loss.item())

        scheduler.step()
        centroid = build_wake_centroid(model, enrollment_loader)
        result = evaluate_kws_fixed_far(model, val_loader, centroid)
        metrics = summarize_kws_metrics(result["probs"], result["labels"], result["cids"])
        current_farfr_metric = frr_far_acceptance_metric(result)
        current_system_outputs = None
        if USE_CL_FARFRR_ACCEPTANCE and gate_model is not None and gate_threshold is not None:
            current_system_outputs = collect_system_outputs(
                model, gate_model, val_loader, centroid, float(gate_threshold)
            )
            current_farfr_metric = frr_far_acceptance_metric(current_system_outputs)
        metrics = update_metrics_with_farfr(metrics, current_farfr_metric)
        mean_acc = episode_acc_sum / max(1, ge2e_batches)
        mean_margin = episode_margin_sum / max(1, ge2e_batches)

        print(
            f"[CL-KWS r{round_id:03d}] Epoch {epoch:03d} | "
            f"loss={loss_sum/max(1,n_batches):.5f} ge2e={ge2e_sum/max(1,ge2e_batches):.5f} "
            f"bg={bg_sum/max(1,bg_batches):.5f} replay={replay_sum/max(1,n_batches):.5f} "
            f"wake={wake_recall_sum/max(1,n_batches):.5f} "
            f"stability={stability_sum/max(1,n_batches):.5f} anchor={anchor_sum/max(1,n_batches):.5f} | "
            f"EpAcc={mean_acc:.4f} EpMargin={mean_margin:.4f} | "
            f"FRR={result['frr']:.4f} Hard-AUC={fmt(metrics['hard_auc'])} "
            f"tail-margin={fmt(metrics['wake_p05_minus_hard_p95'])}"
        )

        print_farfr_monitor(f"       [CL-FRR-FAR r{round_id:03d}]", current_farfr_metric)
        monitor_outputs = current_system_outputs if current_system_outputs is not None else result
        save_epoch_monitor_figures(
            stage=f"cl_round_{round_id:03d}",
            epoch=epoch,
            outputs=monitor_outputs,
            threshold=float(current_farfr_metric["primary_threshold"]) if np.isfinite(current_farfr_metric["primary_threshold"]) else float(result["thr"]),
            title_prefix=f"Val CL r{round_id:03d}",
        )

        eligible = mean_acc >= MIN_EPISODE_ACC_FOR_SELECTION and mean_margin >= 0.0
        if USE_CL_FARFRR_ACCEPTANCE:
            score_gain = best_select - float(current_farfr_metric["selection_score"])
            primary_frr_worse = float(current_farfr_metric["primary_frr"]) - float(baseline_farfr_metric["primary_frr"])
            det_auc_worse = float(current_farfr_metric["det_auc"]) - float(baseline_farfr_metric["det_auc"])
            hard_det_auc_worse = float(current_farfr_metric["hard_det_auc"]) - float(baseline_farfr_metric["hard_det_auc"])
            better = (
                eligible
                and score_gain >= CL_ACCEPT_MIN_SCORE_GAIN
                and primary_frr_worse <= CL_ACCEPT_MAX_FRR_WORSE
                and det_auc_worse <= CL_ACCEPT_MAX_DETAUC_WORSE
                and hard_det_auc_worse <= CL_ACCEPT_MAX_DETAUC_WORSE
            )
        else:
            better = eligible and is_better_checkpoint(
                result["frr"], metrics, best_frr, best_hard_auc, best_margin
            )
        if better:
            accepted = True
            best_select = float(current_farfr_metric["selection_score"])
            best_frr = float(result["frr"])
            best_hard_auc = -float("inf") if metrics["hard_auc"] is None else float(metrics["hard_auc"])
            best_margin = (
                -float("inf")
                if metrics["wake_p05_minus_hard_p95"] is None
                else float(metrics["wake_p05_minus_hard_p95"])
            )
            save_kws_checkpoint(
                candidate_path, model, ge2e_loss, centroid,
                result["thr"], epoch, best_frr, metrics
            )
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CL_KWS_PATIENCE:
                break

    checkpoint, centroid = load_kws_checkpoint(candidate_path, model, ge2e_loss)
    if accepted:
        shutil.copy2(candidate_path, KWS_BEST_MODEL_PATH)
        print(f"CL KWS round {round_id} accepted.")
    else:
        print(f"CL KWS round {round_id}未优于基线，恢复进入该轮前的模型。")
    return model, ge2e_loss, centroid, accepted


class GateReplaySubsetDataset(Dataset):
    def __init__(self, base: MultiRootFolderDualDataset, indices: Sequence[int]):
        self.base = base
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        _, _, gate, _, cid, sid, replay = self.base[self.indices[item]]
        target = torch.tensor(0.0 if cid == ENV_CID else 1.0, dtype=torch.float32)
        return gate, target, cid, sid, replay


def build_gate_cl_indices(dataset: MultiRootFolderDualDataset, epoch: int) -> np.ndarray:
    rng = np.random.RandomState(88000 + epoch)
    original = np.where(~dataset.is_replay)[0]
    replay = np.where(dataset.is_replay)[0]

    selected: List[int] = []
    original_cids = dataset.class_ids[original]
    original_map = {int(cid): original[original_cids == cid] for cid in np.unique(original_cids)}
    selected.extend(sample_indices(rng, original_map.get(23, np.array([])), GATE_ENV_PER_EPOCH, "orig Env"))
    selected.extend(sample_indices(rng, original_map.get(0, np.array([])), GATE_WAKE_PER_EPOCH, "orig Wake"))
    selected.extend(sample_indices(rng, original_map.get(21, np.array([])), GATE_PEP_PER_EPOCH, "orig Pep"))
    hard_each, remainder = divmod(GATE_HARD_PER_EPOCH, 20)
    for cid in range(1, 21):
        n = hard_each + (1 if cid <= remainder else 0)
        selected.extend(sample_indices(rng, original_map.get(cid, np.array([])), n, f"orig Hard {cid}"))
    selected.extend(sample_indices(rng, original_map.get(22, np.array([])), GATE_MASK_PER_EPOCH, "orig Mask"))
    selected.extend(sample_indices(rng, original_map.get(24, np.array([])), GATE_ID24_PER_EPOCH, "orig TTS"))

    if len(replay) > 0:
        replay_env = replay[dataset.class_ids[replay] == 23]
        replay_nonenv = replay[dataset.class_ids[replay] != 23]
        if len(replay_env):
            n = min(CL_GATE_REPLAY_ENV_MAX, len(replay_env))
            selected.extend(rng.choice(replay_env, size=n, replace=False).tolist())
        if len(replay_nonenv):
            n = min(CL_GATE_REPLAY_NONENV_MAX, len(replay_nonenv))
            selected.extend(rng.choice(replay_nonenv, size=n, replace=False).tolist())

    result = np.asarray(selected, dtype=np.int64)
    rng.shuffle(result)
    return result


def finetune_gate_with_replay(
    gate: nn.Module,
    replay_roots: Sequence[str],
    val_loader: DataLoader,
    round_id: int,
) -> Tuple[nn.Module, float, bool]:
    print(f"\n========== CL Round {round_id}: Tiny Gate replay fine-tuning ==========")
    dataset = MultiRootFolderDualDataset(TRAIN_DATASET_DIR, replay_roots)
    teacher = copy.deepcopy(gate).to(DEVICE).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    probs, cids, _ = collect_gate_outputs(gate, val_loader)
    baseline_thr, baseline_metrics, baseline_obj, _ = choose_gate_threshold(probs, cids)
    best_obj = baseline_obj
    best_thr = baseline_thr
    round_dir = os.path.join(DIAG_DIR, f"cl_round_{round_id:03d}")
    os.makedirs(round_dir, exist_ok=True)
    candidate_path = os.path.join(round_dir, "best_cl_gate.pth")
    save_gate_checkpoint(candidate_path, gate, best_thr, 0, baseline_metrics, best_obj)

    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=CL_GATE_LR, weight_decay=GATE_WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    accepted = False
    no_improve = 0

    for epoch in range(1, CL_GATE_EPOCHS + 1):
        indices = build_gate_cl_indices(dataset, epoch)
        loader = DataLoader(
            GateReplaySubsetDataset(dataset, indices),
            batch_size=GATE_BATCH_SIZE,
            shuffle=True,
            num_workers=GATE_NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        gate.train()
        total = 0.0
        seen = 0
        for features, target, cid, _, is_replay in loader:
            features = gate_spec_augment(features.to(DEVICE, non_blocking=True))
            target = target.to(DEVICE, non_blocking=True)
            cid_t = cid.to(DEVICE).view(-1)
            replay_t = is_replay.to(DEVICE).bool().view(-1)

            optimizer.zero_grad(set_to_none=True)
            logits = gate(features).squeeze(1)
            raw = criterion(logits, target)
            weights = torch.ones_like(raw)
            weights[cid_t == ENV_CID] = GATE_ENV_LOSS_WEIGHT
            classification = (raw * weights).sum() / weights.sum().clamp_min(1.0)

            with torch.no_grad():
                teacher_prob = torch.sigmoid(teacher(features).squeeze(1))
            student_prob = torch.sigmoid(logits)
            original_mask = ~replay_t
            stability = (
                F.mse_loss(student_prob[original_mask], teacher_prob[original_mask])
                if original_mask.any()
                else logits.sum() * 0.0
            )
            loss = classification + CL_GATE_LAMBDA_STABILITY * stability
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), CLIP_NORM)
            optimizer.step()

            batch_size = int(target.numel())
            total += float(loss.item()) * batch_size
            seen += batch_size

        probs, cids, _ = collect_gate_outputs(gate, val_loader)
        threshold, metrics, objective, feasible = choose_gate_threshold(probs, cids)
        print(
            f"[CL-Gate r{round_id:03d}] Epoch {epoch:03d} | "
            f"loss={total/max(1,seen):.5f} objective={objective:.5f} feasible={feasible}"
        )
        if objective > best_obj + 1e-9:
            accepted = True
            best_obj = objective
            best_thr = threshold
            save_gate_checkpoint(candidate_path, gate, threshold, epoch, metrics, objective)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CL_GATE_PATIENCE:
                break

    checkpoint = load_gate_checkpoint(candidate_path, gate)
    best_thr = float(checkpoint["gate_threshold"])
    if accepted:
        shutil.copy2(candidate_path, GATE_BEST_MODEL_PATH)
        print(f"CL Gate round {round_id} accepted.")
    else:
        print(f"CL Gate round {round_id}未优于基线，恢复进入该轮前的Gate。")
    return gate, best_thr, accepted

def main() -> None:
    print(
        f"device={DEVICE} | profile=RTX5070_12GB | "
        f"Stage1=Stable-True-GE2E | Stage2=TinyGate | "
        f"Stage4=FRR-FAR-oriented Replay | TARGET_FAR={TARGET_FAR}"
    )

    train_dataset = FolderDualDataset(TRAIN_DATASET_DIR)
    val_dataset_full = FolderDualDataset(VAL_DATASET_DIR)
    test_dataset = FolderDualDataset(TEST_DATASET_DIR)
    adapt_dataset, calibration_dataset, adapt_wav_root = build_adaptation_and_calibration_sets(
        val_dataset_full
    )

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    adapt_loader = DataLoader(
        adapt_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader_full = DataLoader(
        val_dataset_full,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Round 0: stable True GE2E + independent Tiny Gate.
    kws_model, ge2e_loss, wake_centroid, kws_epoch, _ = train_stage1_kws(
        train_dataset, calibration_loader
    )
    gate_model, gate_epoch, gate_threshold = train_stage2_gate(
        train_dataset, calibration_loader
    )
    kws_threshold, final_dir = calibrate_stage3_system(
        kws_model, gate_model, calibration_loader, wake_centroid,
        gate_threshold, kws_epoch, gate_epoch,
    )

    if not ENABLE_CONTINUAL_REPLAY or CL_ROUNDS <= 0:
        evaluate_final_farfr_outputs(
            kws_model, gate_model, val_loader_full, test_loader,
            wake_centroid, gate_threshold, final_dir,
        )
        print(f"FRR-FAR results saved to: {final_dir}")
        return

    replay_roots: List[str] = []
    seen_sids: set = set()

    for round_id in range(1, CL_ROUNDS + 1):
        print(f"\n================ Continual Replay Round {round_id:03d} ================")
        outputs = collect_system_outputs(
            kws_model, gate_model, adapt_loader, wake_centroid, gate_threshold
        )
        before = system_metrics_from_outputs(outputs, kws_threshold)
        print(
            f"[Adapt before r{round_id:03d}] FAR={before['far']:.6f} "
            f"FRR={before['frr']:.6f}"
        )

        cases = mine_system_false_alarms(
            outputs, gate_threshold, kws_threshold, seen_sids
        )
        print(f"挖掘到新的系统级false alarms: {len(cases)}")
        if not cases:
            print("没有新的false alarm，停止持续学习。")
            break
        seen_sids.update(case["sid"] for case in cases)

        generate_replay_wavs(cases, adapt_wav_root, round_id)
        n_replay_wavs = count_replay_wavs(round_id)
        if n_replay_wavs <= 0:
            print("本轮没有成功生成任何Replay wav，停止持续学习。")
            break

        try:
            run_replay_feature_extraction(round_id)
        except subprocess.CalledProcessError as exc:
            print(f"Replay特征提取失败，停止持续学习: returncode={exc.returncode}")
            break

        replay_root = replay_feature_round_dir(round_id)
        if not os.path.isdir(replay_root):
            print(f"Replay特征目录不存在，停止: {replay_root}")
            break
        n_replay_npz = len(glob.glob(os.path.join(replay_root, "ID_*", "*.npz")))
        if n_replay_npz <= 0:
            print(f"Replay特征目录中没有NPZ，停止持续学习: {replay_root}")
            break
        replay_roots.append(replay_root)

        previous_centroid = wake_centroid.detach().clone()
        kws_model, ge2e_loss, wake_centroid, kws_accepted = finetune_kws_with_replay(
            kws_model, ge2e_loss, train_dataset, replay_roots,
            calibration_loader, previous_centroid, round_id,
            gate_model=gate_model, gate_threshold=gate_threshold,
        )
        if ENABLE_GATE_CL:
            gate_model, gate_threshold, gate_accepted = finetune_gate_with_replay(
                gate_model, replay_roots, calibration_loader, round_id,
            )
        else:
            gate_accepted = False
            print("Tiny Gate CL is disabled; keep Stage-2 gate model and threshold unchanged.")

        kws_threshold, final_dir = calibrate_stage3_system(
            kws_model, gate_model, calibration_loader, wake_centroid,
            gate_threshold, kws_epoch + round_id, gate_epoch + round_id,
        )
        print(
            f"Round {round_id:03d} complete | KWS accepted={kws_accepted} | "
            f"Gate accepted={gate_accepted} | gate_thr={gate_threshold:.6f} | "
            f"kws_thr={kws_threshold:.6f}"
        )

        if EVALUATE_TEST_EACH_ROUND:
            strict_test_system(
                kws_model, gate_model, test_loader, wake_centroid,
                gate_threshold, kws_threshold, final_dir,
            )

    evaluate_final_farfr_outputs(
        kws_model, gate_model, val_loader_full, test_loader,
        wake_centroid, gate_threshold, final_dir,
    )
    print("\nTrue-GE2E continual replay training completed.")
    print(f"FRR-FAR results saved to: {final_dir}")
    print(f"KWS checkpoint: {KWS_BEST_MODEL_PATH}")
    print(f"Gate checkpoint: {GATE_BEST_MODEL_PATH}")
    print(f"System checkpoint: {SYSTEM_BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
