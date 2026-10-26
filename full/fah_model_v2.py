# -*- coding: utf-8 -*-
"""FAH v2 model and losses, without training or filesystem side effects.

The dual-stream encoder, environment gate, and GE2E implementation retain the
original lite2 architecture so old state_dicts remain compatible. Operating-point
losses use unit scores (cosine + 1) / 2; BackgroundMetricLoss retains raw cosine.
"""

from collections.abc import Mapping
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

EMBEDDING_DIM = 128
BG_NEG_MARGIN = 0.30
METRIC_TEMPERATURE = 0.10
GE2E_FIXED_SCALE = 10.0
LEARN_GE2E_SCALE = False
GE2E_SCALE_MIN = 5.0
GE2E_SCALE_MAX = 30.0
SPEC_P = 0.30
FREQ_MASK_MFCC = 2
FREQ_MASK_MEL = 4
TIME_MASK_FRAC = 0.03
BASE_C = 32
DROPOUT = 0.10
USE_TEMPORAL_1x5 = True
TCN_LAYERS = 4
TCN_DILATIONS = (1, 2, 2, 4)
ATT_HIDDEN = 96
GATE_DROPOUT = 0.10
GATE_SPEC_P = 0.40
GATE_FREQ_MASK_MAX = 6
GATE_TIME_MASK_FRAC = 0.05


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


def cosine_to_unit_score(cosine: torch.Tensor) -> torch.Tensor:
    return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)


def operating_point_loss(
    embeddings: torch.Tensor,
    cids: torch.Tensor,
    reference_centroid: torch.Tensor,
    thresholds,
    temperature: float = 0.10,
    pos_margin: float = 0.03,
    neg_margin: float = 0.03,
) -> Dict[str, torch.Tensor]:
    """Return independently weighted wake-vs-rest loss components.

    ``thresholds`` can be a sequence of unit-score thresholds, a scalar, or a
    mapping from class ID to its relevant unit-score threshold. With a sequence,
    positives use the maximum valid threshold and negatives use the minimum.
    A class mapping avoids forcing ordinary and confusable negatives to use the
    same operating point. Missing mapping keys receive no threshold loss.

    Thresholds outside [0, 1] represent no attainable training target and are
    ignored. Positive and negative targets are bounded to [0, 1]. Ranking remains
    available even when no usable threshold exists. It compares the lowest-score
    positive quartile with the highest-score negative quartile. The centroid is
    detached so rejecting a negative cannot move the positive reference itself.

    Return keys are ``positive``, ``negative`` and ``ranking``; callers choose
    their coefficients. These are unit-score losses, unlike the raw-cosine
    ``BackgroundMetricLoss`` retained above for backward compatibility.
    """
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if min(pos_margin, neg_margin) < 0 or not np.isfinite([pos_margin, neg_margin]).all():
        raise ValueError("margins must be finite and nonnegative")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [batch, embedding_dim]")
    cids = cids.to(device=embeddings.device).reshape(-1)
    if cids.numel() != embeddings.shape[0]:
        raise ValueError("cids and embeddings must have equal batch size")
    centroid = reference_centroid.detach().to(device=embeddings.device, dtype=embeddings.dtype).reshape(-1)
    if centroid.numel() != embeddings.shape[1]:
        raise ValueError("reference centroid has an incompatible dimension")
    score = cosine_to_unit_score(F.normalize(embeddings, p=2, dim=-1) @ F.normalize(centroid, p=2, dim=0))
    zero = embeddings.sum() * 0.0
    losses = {"positive": zero, "negative": zero, "ranking": zero}

    def valid_threshold(value):
        value = float(value)
        return value if np.isfinite(value) and 0.0 <= value <= 1.0 else None

    class_thresholds = None
    if isinstance(thresholds, Mapping):
        class_thresholds = {int(k): valid_threshold(v) for k, v in thresholds.items()}
        valid = [v for v in class_thresholds.values() if v is not None]
    else:
        raw = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        valid = [v for v in (valid_threshold(x) for x in raw) if v is not None]

    positive = score[cids == 0]
    negative = score[cids != 0]
    if valid:
        if positive.numel():
            target = min(1.0, max(valid) + pos_margin)
            losses["positive"] = F.softplus((target - positive) / temperature).mean()
        if negative.numel():
            if class_thresholds is None:
                target = max(0.0, min(valid) - neg_margin)
                losses["negative"] = F.softplus((negative - target) / temperature).mean()
            else:
                terms = []
                for cid, threshold in class_thresholds.items():
                    if cid == 0 or threshold is None:
                        continue
                    selected = score[cids == cid]
                    if selected.numel():
                        terms.append(F.softplus((selected - max(0.0, threshold - neg_margin)) / temperature))
                if terms:
                    losses["negative"] = torch.cat(terms).mean()

    if positive.numel() and negative.numel():
        n_pos = max(1, (positive.numel() + 3) // 4)
        n_neg = max(1, (negative.numel() + 3) // 4)
        low_positive = torch.topk(positive, n_pos, largest=False).values
        high_negative = torch.topk(negative, n_neg, largest=True).values
        pairwise_gap = high_negative[:, None] - low_positive[None, :]
        losses["ranking"] = F.softplus((pairwise_gap + pos_margin + neg_margin) / temperature).mean()
    return losses
