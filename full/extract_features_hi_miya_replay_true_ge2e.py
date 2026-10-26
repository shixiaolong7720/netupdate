import os
import glob
import shutil
import argparse
from datetime import datetime

import numpy as np
import librosa
import pywt
import torch
from tqdm import tqdm

# ==========================================================
# 0) 全局配置：路径
# ==========================================================
TARGET_SR = 16000
TARGET_DUR = 1.5

# 你的 wav 数据集根目录：含 train/val/test，每个 split 下是 ID_* 文件夹
BASE_DATASET_DIR = r"D:\myproject\project\Data\Dataset\hi_miya"
# 输出特征根目录
BASE_FEATURE_DIR = r"D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake"

SPLITS = ["train", "val", "test"]

# replay wav / feature paths：由训练脚本生成 replay_buffer_wav/round_xxx/ID_*
REPLAY_WAV_ROOT = os.path.join(BASE_DATASET_DIR, "replay_buffer_wav")
REPLAY_FEATURE_ROOT = os.path.join(BASE_FEATURE_DIR, "replay_buffer")

# 默认不再对已有 hi_miya 数据集做破坏性 VAD 路由；replay 特征提取也不做 VAD 路由，
# 因为 ID_23 环境误触发本身可能无语音，不能被移动到 ID_25。
RUN_VAD_ROUTING = False
PROCESS_STANDARD_SPLITS = False
PROCESS_REPLAY = True

# Silero-VAD 本地仓库（必须包含 hubconf.py）
LOCAL_VAD_REPO = r"D:\myproject\gittest\silero-vad-master\silero-vad"

# ==========================================================
# 1) Stage A：VAD 路由配置
# ==========================================================
SCAN_CIDS = list(range(1, 22)) + [23, 24]   # ID_1~ID_21 + ID_23 + ID_24
ENV_CID = 23
NOISE_CID_FOLDER = "ID_25_noise"  # VAD-fail 统一移入这里（每个 split 内创建）

VAD_THRESHOLD = 0.30
MIN_SPEECH_RATIO = 0.03
MIN_MAX_SEG_MS = 250.0
MIN_DUR_SEC = 0.20

# 尽量用更稳健的 silero 参数（若版本不支持会自动回退）
VAD_MIN_SPEECH_MS = 200
VAD_MIN_SILENCE_MS = 100
VAD_SPEECH_PAD_MS = 0

# 是否“破坏性移动”（在原 split 下移动 wav 到 ID_25_noise）
# 你这次需求是“必须确保 moved 后原位置不存在”，所以这里 True
VAD_MOVE_IN_PLACE = False

# ==========================================================
# 2) Stage B：特征提取配置
# ==========================================================
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40
N_MFCC = 20
F_MAX = 6000

STRICT_FIXED_LENGTH = True
EPS = 1e-6

TARGET_SAMPLES = int(TARGET_SR * TARGET_DUR)
TARGET_FRAMES = 1 + max(0, (TARGET_SAMPLES - N_FFT) // HOP_LENGTH)

# MFCC gate (Δ/ΔΔ)
DELTA_WIDTH = 9
USE_DELTA = True

# ✅ 特征提取时是否跳过 ID_25_noise（你要求它不参与 KWS）
SKIP_ID25_IN_FEATURE = True

print(
    f"[Config] sr={TARGET_SR}, dur={TARGET_DUR}s, target_samples={TARGET_SAMPLES}, "
    f"n_fft={N_FFT}, hop={HOP_LENGTH}, fmax={F_MAX}, target_frames={TARGET_FRAMES}"
)


# ==========================================================
# 3) 加载 Silero-VAD（惰性加载）
# ==========================================================
vad_model = None
get_speech_timestamps = None


def ensure_vad_loaded():
    global vad_model, get_speech_timestamps
    if vad_model is not None and get_speech_timestamps is not None:
        return
    print("🔧 Loading silero-vad ...")
    model, utils = torch.hub.load(
        repo_or_dir=LOCAL_VAD_REPO,
        model="silero_vad",
        source="local"
    )
    (gst, _, _, _, _) = utils
    model.eval()
    vad_model = model
    get_speech_timestamps = gst
    print("✅ silero-vad loaded.\n")



def safe_get_speech_timestamps(wav_tensor: torch.Tensor, sr: int):
    ensure_vad_loaded()
    try:
        return get_speech_timestamps(
            wav_tensor, vad_model,
            sampling_rate=sr,
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )
    except TypeError:
        return get_speech_timestamps(
            wav_tensor, vad_model,
            sampling_rate=sr,
            threshold=VAD_THRESHOLD,
        )


def compute_vad_stats(wav: np.ndarray, sr: int):
    dur_sec = len(wav) / float(sr)
    if dur_sec <= 0:
        return 0.0, 0.0, 0.0, 0, 0.0

    wav_tensor = torch.from_numpy(wav).float()
    segs = safe_get_speech_timestamps(wav_tensor, sr)
    if not segs:
        return 0.0, 0.0, 0.0, 0, dur_sec

    speech_samples = 0
    max_seg_samples = 0
    for s in segs:
        st = int(s["start"])
        ed = int(s["end"])
        if ed > st:
            L = ed - st
            speech_samples += L
            if L > max_seg_samples:
                max_seg_samples = L

    speech_sec = speech_samples / float(sr)
    speech_ratio = speech_sec / (dur_sec + 1e-12)
    max_seg_ms = (max_seg_samples / float(sr)) * 1000.0
    return float(speech_ratio), float(speech_sec), float(max_seg_ms), int(len(segs)), float(dur_sec)


def vad_pass(speech_ratio: float, max_seg_ms: float) -> bool:
    return (speech_ratio >= MIN_SPEECH_RATIO) or (max_seg_ms >= MIN_MAX_SEG_MS)


def parse_cid(folder_name: str) -> int:
    try:
        return int(folder_name.split("_")[1])
    except Exception:
        return -1


def safe_move(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)

    # 强校验：dst 必须存在，src 必须不存在
    if not os.path.exists(dst):
        raise RuntimeError(f"[MOVE FAIL] dst not found after move: {dst}")
    if os.path.exists(src):
        raise RuntimeError(f"[MOVE FAIL] src still exists after move: {src}")
    return dst


def route_split_with_vad(split_dir: str):
    """
    对 split 下的 ID_1~24 跑 VAD：
      - FAIL -> move to ID_25_noise
      - PASS -> keep
    生成 report.tsv
    """
    noise_dir = os.path.join(split_dir, NOISE_CID_FOLDER)
    os.makedirs(noise_dir, exist_ok=True)

    report_path = os.path.join(split_dir, f"vad_route_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("cid\tsrc_folder\twav\tdecision\tdst\t"
                "speech_ratio\tspeech_sec\tmax_seg_ms\tnum_segs\tdur_sec\t"
                "vad_thr\tmin_ratio\tmin_maxseg_ms\tnote\n")

        id_folders = [d for d in os.listdir(split_dir)
                      if os.path.isdir(os.path.join(split_dir, d)) and d.startswith("ID_")]
        id_folders = sorted(id_folders)

        total = moved = kept = err = 0

        for folder in id_folders:
            cid = parse_cid(folder)
            if cid not in SCAN_CIDS:
                continue
            if folder == NOISE_CID_FOLDER:
                continue

            folder_path = os.path.join(split_dir, folder)
            wavs = sorted(glob.glob(os.path.join(folder_path, "*.wav")))
            if not wavs:
                continue

            for wav_path in tqdm(wavs, desc=f"[VAD]{os.path.basename(split_dir)}:{folder}", leave=False):
                total += 1
                note = ""
                try:
                    wav, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
                except Exception as e:
                    err += 1
                    note = f"read_error:{e}"
                    f.write(f"{cid}\t{folder}\t{wav_path}\tERR\t\t0\t0\t0\t0\t0\t"
                            f"{VAD_THRESHOLD}\t{MIN_SPEECH_RATIO}\t{MIN_MAX_SEG_MS}\t{note}\n")
                    continue

                dur_sec = len(wav) / float(TARGET_SR)
                if dur_sec < MIN_DUR_SEC:
                    # 太短：视为 FAIL
                    decision = "FAIL"
                    try:
                        dst_name = f"{folder}__{os.path.basename(wav_path)}"
                        dst_path = os.path.join(noise_dir, dst_name)
                        if VAD_MOVE_IN_PLACE:
                            dst_final = safe_move(wav_path, dst_path)
                        else:
                            os.makedirs(noise_dir, exist_ok=True)
                            shutil.copy2(wav_path, dst_path)
                            dst_final = dst_path
                        moved += 1
                        f.write(f"{cid}\t{folder}\t{wav_path}\t{decision}\t{dst_final}\t"
                                f"0\t0\t0\t0\t{dur_sec:.4f}\t{VAD_THRESHOLD}\t{MIN_SPEECH_RATIO}\t{MIN_MAX_SEG_MS}\tshort<min_dur\n")
                    except Exception as e:
                        err += 1
                        f.write(f"{cid}\t{folder}\t{wav_path}\tERR\t\t0\t0\t0\t0\t{dur_sec:.4f}\t"
                                f"{VAD_THRESHOLD}\t{MIN_SPEECH_RATIO}\t{MIN_MAX_SEG_MS}\tmove_error:{e}\n")
                    continue

                speech_ratio, speech_sec, max_seg_ms, num_segs, dur_sec = compute_vad_stats(wav, TARGET_SR)
                if vad_pass(speech_ratio, max_seg_ms):
                    decision = "PASS"
                    kept += 1
                    f.write(f"{cid}\t{folder}\t{wav_path}\t{decision}\t\t"
                            f"{speech_ratio:.6f}\t{speech_sec:.4f}\t{max_seg_ms:.2f}\t{num_segs}\t{dur_sec:.4f}\t"
                            f"{VAD_THRESHOLD}\t{MIN_SPEECH_RATIO}\t{MIN_MAX_SEG_MS}\t\n")
                else:
                    decision = "FAIL"
                    try:
                        dst_name = f"{folder}__{os.path.basename(wav_path)}"
                        dst_path = os.path.join(noise_dir, dst_name)
                        if VAD_MOVE_IN_PLACE:
                            dst_final = safe_move(wav_path, dst_path)
                        else:
                            shutil.copy2(wav_path, dst_path)
                            dst_final = dst_path
                        moved += 1
                        f.write(f"{cid}\t{folder}\t{wav_path}\t{decision}\t{dst_final}\t"
                                f"{speech_ratio:.6f}\t{speech_sec:.4f}\t{max_seg_ms:.2f}\t{num_segs}\t{dur_sec:.4f}\t"
                                f"{VAD_THRESHOLD}\t{MIN_SPEECH_RATIO}\t{MIN_MAX_SEG_MS}\t\n")
                    except Exception as e:
                        err += 1
                        f.write(f"{cid}\t{folder}\t{wav_path}\tERR\t\t"
                                f"{speech_ratio:.6f}\t{speech_sec:.4f}\t{max_seg_ms:.2f}\t{num_segs}\t{dur_sec:.4f}\t"
                                f"{VAD_THRESHOLD}\t{MIN_SPEECH_RATIO}\t{MIN_MAX_SEG_MS}\tmove_error:{e}\n")

    print(f"✅ VAD routing done: {split_dir}")
    print(f"📄 report: {report_path}")
    print(f"📁 noise : {noise_dir}\n")


# ==========================================================
# 4) 特征提取：你原有逻辑 + MFCC轻量CMVN + mfcc_gate
# ==========================================================
def haar_wavelet_denoise_optimized(y: np.ndarray, frame_length=1024, fade_len=16) -> np.ndarray:
    denoised_y = np.zeros_like(y)
    num_frames = int(np.ceil(len(y) / frame_length))
    buffer = np.zeros(len(y) + frame_length, dtype=np.float32)
    weight_buffer = np.zeros_like(buffer)
    window = np.ones(frame_length, dtype=np.float32)

    if fade_len > 0 and fade_len * 2 < frame_length:
        fade_curve = np.hanning(fade_len * 2)
        window[:fade_len] = fade_curve[:fade_len]
        window[-fade_len:] = fade_curve[-fade_len:]

    for i in range(num_frames):
        start = i * frame_length
        end = min((i + 1) * frame_length, len(y))
        frame = y[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode="constant")

        cA, cD = pywt.wavedec(frame, "haar", level=1)
        median_cD = np.median(cD)
        mad = np.median(np.abs(cD - median_cD))

        if mad > 1e-6 and not np.all(cD == 0):
            sigma = mad / 0.6745
            tau = sigma * np.sqrt(2 * np.log(frame_length))
            tau = min(tau, np.max(np.abs(cD)) * 0.9)
            cD_thresh = pywt.threshold(cD, value=tau, mode="soft")
        else:
            cD_thresh = cD

        reconstructed_frame = pywt.waverec([cA, cD_thresh], "haar")
        reconstructed_frame = reconstructed_frame[:frame_length] * window

        buffer[start:start + frame_length] += reconstructed_frame
        weight_buffer[start:start + frame_length] += window

    valid = weight_buffer[:len(y)] > 0
    denoised_y[valid] = buffer[:len(y)][valid] / weight_buffer[:len(y)][valid]
    denoised_y[~valid] = y[~valid]
    denoised_y = np.nan_to_num(denoised_y, nan=0.0, posinf=0.0, neginf=0.0)
    return denoised_y.astype(np.float32)


def spectral_denoising_tunable(
    x: np.ndarray, alpha=0.60, w_s=0.4,
    use_quantile_minmax=True, q_lo=0.01, q_hi=0.99,
    soft_mask=True, beta=0.20, delta=0.04, eps=1e-6
) -> np.ndarray:
    if use_quantile_minmax:
        x_min = np.quantile(x, q_lo)
        x_max = np.quantile(x, q_hi)
    else:
        x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max - x_min < eps:
        return x.astype(np.float32)

    xn = (x - x_min) / (x_max - x_min + eps)
    mu_t = np.mean(xn, axis=1, keepdims=True)
    xt = xn - mu_t
    mu_f = np.mean(xn, axis=0, keepdims=True)
    xs = xn - mu_f

    if soft_mask:
        Mt = 1.0 / (1.0 + np.exp(-(xt + delta) / (beta + eps)))
        Ms = 1.0 / (1.0 + np.exp(-(xs + delta) / (beta + eps)))
    else:
        Mt = (xt > -delta).astype(np.float32)
        Ms = (xs > -delta).astype(np.float32)

    xd = (1 - alpha) * (w_s * alpha * xs * Ms + (1 - w_s) * (1 - alpha) * xt * Mt) + alpha * xn
    return xd.astype(np.float32)


def cmvn_per_utt(x, eps=1e-6, std_floor=0.3, clip=5.0):
    mu = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    std = np.maximum(std, std_floor)
    x_hat = (x - mu) / (std + eps)
    if clip is not None:
        x_hat = np.clip(x_hat, -clip, clip)
    return x_hat.astype(np.float32)


def maybe_fix_length(y: np.ndarray):
    if len(y) == TARGET_SAMPLES:
        return y.astype(np.float32)
    if STRICT_FIXED_LENGTH:
        return None
    if len(y) > TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]
    else:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode="constant")
    return y.astype(np.float32)


def mfcc_with_deltas(mfcc20: np.ndarray, width: int = 9) -> np.ndarray:
    d1 = librosa.feature.delta(mfcc20, width=width, order=1, axis=1)
    d2 = librosa.feature.delta(mfcc20, width=width, order=2, axis=1)
    return np.concatenate([mfcc20, d1, d2], axis=0).astype(np.float32)


def extract_features_all(wav_path: str):
    y, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    y = maybe_fix_length(y)
    if y is None:
        return None, None, None

    y_denoised = haar_wavelet_denoise_optimized(y, frame_length=1024)
    y_preemph = librosa.effects.preemphasis(y_denoised, coef=0.8)

    mel_spec = librosa.feature.melspectrogram(
        y=y_preemph, sr=TARGET_SR,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=F_MAX,
        power=2.0, center=False
    )
    log_mel = np.log(mel_spec + EPS).astype(np.float32)

    mfcc_kws = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC).astype(np.float32)  # [20,T]

    if log_mel.shape[1] != TARGET_FRAMES or mfcc_kws.shape[1] != TARGET_FRAMES:
        return None, None, None

    logmel_denoised = spectral_denoising_tunable(log_mel, alpha=0.7, w_s=0.1)

    # ✅ MFCC：轻量 CMVN（温和，不破坏倒谱空间拓扑）
    mfcc_kws_proc = cmvn_per_utt(mfcc_kws, std_floor=0.3, clip=5.0)

    # ✅ Gate：在 mfcc_kws_proc 上做 Δ/ΔΔ
    mfcc_gate = mfcc_with_deltas(mfcc_kws_proc, width=DELTA_WIDTH) if USE_DELTA else None

    return mfcc_kws_proc, mfcc_gate, logmel_denoised


def build_features_for_split(split: str):
    src_split_dir = os.path.join(BASE_DATASET_DIR, split)
    dest_split_dir = os.path.join(BASE_FEATURE_DIR, split)
    if not os.path.exists(src_split_dir):
        print(f"⏭️ missing split: {src_split_dir}")
        return

    id_folders = [d for d in os.listdir(src_split_dir)
                  if os.path.isdir(os.path.join(src_split_dir, d)) and d.startswith("ID_")]

    if SKIP_ID25_IN_FEATURE:
        id_folders = [d for d in id_folders if d != NOISE_CID_FOLDER]

    if not id_folders:
        return

    print(f"\n📦 Feature extracting [{split}] | folders={len(id_folders)} (skip_ID25={SKIP_ID25_IN_FEATURE})")

    for folder_name in sorted(id_folders):
        src_folder = os.path.join(src_split_dir, folder_name)
        dest_folder = os.path.join(dest_split_dir, folder_name)
        os.makedirs(dest_folder, exist_ok=True)

        class_id = int(folder_name.split("_")[1])
        binary_label = 1 if class_id == 0 else 0

        wav_files = glob.glob(os.path.join(src_folder, "*.wav"))
        if not wav_files:
            continue

        skipped = success = 0
        for wav_path in tqdm(wav_files, desc=f"[FEAT]{split}:{folder_name}", leave=False):
            base = os.path.splitext(os.path.basename(wav_path))[0]
            npz_path = os.path.join(dest_folder, f"{base}.npz")
            if os.path.exists(npz_path):
                success += 1
                continue

            mfcc_kws, mfcc_gate, logmel = extract_features_all(wav_path)
            if mfcc_kws is None:
                skipped += 1
                continue

            save_dict = dict(
                mfcc=mfcc_kws,          # KWS 主线继续用这个
                mfcc_kws=mfcc_kws,
                logmel=logmel,
                label=np.array(binary_label, dtype=np.int8),
                class_id=np.array(class_id, dtype=np.int16),
            )
            if mfcc_gate is not None:
                save_dict["mfcc_gate"] = mfcc_gate

            np.savez_compressed(npz_path, **save_dict)
            success += 1

        if skipped > 0:
            print(f"  ⚠️ {folder_name}: success={success} skipped={skipped}")




def build_features_for_root(src_root: str, dest_root: str, tag: str = "replay"):
    """
    对任意 root/ID_* 结构的 wav 目录提取特征。
    用于 replay_buffer_wav/round_xxx -> replay_buffer/round_xxx。
    不做 VAD 路由，不移动文件。
    """
    if not os.path.exists(src_root):
        print(f"⏭️ missing root: {src_root}")
        return 0, 0

    id_folders = [
        d for d in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, d)) and d.startswith("ID_")
    ]

    if SKIP_ID25_IN_FEATURE:
        id_folders = [d for d in id_folders if d != NOISE_CID_FOLDER]

    if not id_folders:
        print(f"⏭️ no ID folders under: {src_root}")
        return 0, 0

    total_success, total_skipped = 0, 0
    print(f"\n📦 Feature extracting [{tag}] | src={src_root} | folders={len(id_folders)}")

    for folder_name in sorted(id_folders):
        src_folder = os.path.join(src_root, folder_name)
        dest_folder = os.path.join(dest_root, folder_name)
        os.makedirs(dest_folder, exist_ok=True)

        try:
            class_id = int(folder_name.split("_")[1])
        except Exception:
            print(f"  ⚠️ skip invalid ID folder: {folder_name}")
            continue
        binary_label = 1 if class_id == 0 else 0

        wav_files = sorted(glob.glob(os.path.join(src_folder, "*.wav")))
        if not wav_files:
            continue

        skipped = success = 0
        for wav_path in tqdm(wav_files, desc=f"[FEAT]{tag}:{folder_name}", leave=False):
            base = os.path.splitext(os.path.basename(wav_path))[0]
            npz_path = os.path.join(dest_folder, f"{base}.npz")
            if os.path.exists(npz_path):
                success += 1
                continue

            mfcc_kws, mfcc_gate, logmel = extract_features_all(wav_path)
            if mfcc_kws is None:
                skipped += 1
                continue

            save_dict = dict(
                mfcc=mfcc_kws,
                mfcc_kws=mfcc_kws,
                logmel=logmel,
                label=np.array(binary_label, dtype=np.int8),
                class_id=np.array(class_id, dtype=np.int16),
            )
            if mfcc_gate is not None:
                save_dict["mfcc_gate"] = mfcc_gate

            np.savez_compressed(npz_path, **save_dict)
            success += 1

        total_success += success
        total_skipped += skipped
        if skipped > 0:
            print(f"  ⚠️ {folder_name}: success={success} skipped={skipped}")

    print(f"✅ Feature extraction done [{tag}] | success={total_success} skipped={total_skipped}")
    print(f"📁 dest: {dest_root}\n")
    return total_success, total_skipped


def build_features_for_replay_round(round_id: int):
    src_round = os.path.join(REPLAY_WAV_ROOT, f"round_{round_id:03d}")
    dst_round = os.path.join(REPLAY_FEATURE_ROOT, f"round_{round_id:03d}")
    return build_features_for_root(src_round, dst_round, tag=f"replay_round{round_id:03d}")


def parse_args():
    parser = argparse.ArgumentParser(description="hi_miya feature extraction with True-GE2E FA replay support")
    parser.add_argument("--standard", action="store_true",
                        help="extract standard train/val/test splits under BASE_DATASET_DIR")
    parser.add_argument("--replay-only", action="store_true",
                        help="extract replay_buffer_wav only; no VAD routing")
    parser.add_argument("--round-id", type=int, default=None,
                        help="specific replay round id, e.g. 1 for round_001")
    parser.add_argument("--all-replay", action="store_true",
                        help="extract all existing replay rounds")
    parser.add_argument("--run-vad-routing", action="store_true",
                        help="run VAD routing for standard splits")
    parser.add_argument("--replay-wav-root", type=str, default=None,
                        help="override REPLAY_WAV_ROOT")
    parser.add_argument("--replay-feature-root", type=str, default=None,
                        help="override REPLAY_FEATURE_ROOT")
    return parser.parse_args()


def main():
    global REPLAY_WAV_ROOT, REPLAY_FEATURE_ROOT
    args = parse_args()

    if args.replay_wav_root:
        REPLAY_WAV_ROOT = os.path.abspath(args.replay_wav_root)
    if args.replay_feature_root:
        REPLAY_FEATURE_ROOT = os.path.abspath(args.replay_feature_root)

    do_standard = bool(args.standard)
    do_replay = bool(args.replay_only or args.all_replay or args.round_id is not None)

    # Without explicit arguments, process replay only to avoid touching source splits.
    if not do_standard and not do_replay:
        do_replay = PROCESS_REPLAY

    if do_standard:
        if args.run_vad_routing or RUN_VAD_ROUTING:
            for split in SPLITS:
                split_dir = os.path.join(BASE_DATASET_DIR, split)
                if os.path.exists(split_dir):
                    route_split_with_vad(split_dir)
        for split in SPLITS:
            build_features_for_split(split)

    if do_replay:
        print(f"[Replay roots] wav={REPLAY_WAV_ROOT}")
        print(f"[Replay roots] feature={REPLAY_FEATURE_ROOT}")
        if args.round_id is not None:
            build_features_for_replay_round(args.round_id)
        else:
            if not os.path.exists(REPLAY_WAV_ROOT):
                print(f"⏭️ replay wav root missing: {REPLAY_WAV_ROOT}")
            else:
                round_dirs = [
                    d for d in os.listdir(REPLAY_WAV_ROOT)
                    if os.path.isdir(os.path.join(REPLAY_WAV_ROOT, d)) and d.startswith("round_")
                ]
                if not round_dirs:
                    print(f"⏭️ no replay rounds under: {REPLAY_WAV_ROOT}")
                for rd in sorted(round_dirs):
                    try:
                        rid = int(rd.split("_")[1])
                    except Exception:
                        continue
                    build_features_for_replay_round(rid)

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
