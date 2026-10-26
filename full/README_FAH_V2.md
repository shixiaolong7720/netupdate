# FAH v2：面向低误报工作点的训练版本

本版保留原 lite2 的双分支 DS-CNN + TCN + AttentiveStatsPooling、128 维归一化 embedding、GE2E 和 Tiny Environmental Gate，修改训练、采样、校准及验收流程。原来的三个脚本无需覆盖。

主要优化：

- 开发集按普通负样本 **FRR@≤1 FA/h**、加入混淆词 **FRR@≤20 FA/h** 选模，CL 默认要求两项均不退化且平均 FRR 改善。
- 精确利用整数误报预算，处理并列分数、零误报预算、得分等于 1 等边界。
- 联合校准 Gate/KWS，保存两个 benchmark 工作点以及可部署的共享阈值对。
- 训练池中定期挖掘低分正样本、高分负样本、两个工作点附近的边界负样本；默认各 25% 困难采样。每批保持原 GE2E 的 N×M 结构。
- 保留 GE2E，新增工作点相关的正负样本损失和轻量排序损失。旧的单位分数与原始余弦 margin 分开处理。
- 直接重用训练/独立 adapt 的 NPZ，不依赖生成 replay WAV，不移动原音频，不重新提取特征。
- Gate 可单独微调处理环境样本；无指标收益时恢复已有 Gate 或旁路方案。
- 测试只使用开发集冻结阈值，实际 FA/h 超预算时标记 `feasible=false`。
- 冷启动不会把随机 epoch 0 当成最终训练结果；warm-start 和 CL 保留输入模型以便无收益时回退。

## 文件与依赖

将以下文件放在同一目录：

```text
train_mymodel_fah_v2.py     # 主训练与冻结评测入口
fah_model_v2.py            # 原网络、GE2E、新损失
fah_data_v2.py             # NPZ、时长、困难采样
fah_metrics_v2.py          # 精确预算、联合校准、冻结阈值评测
fah_labels_v2.py           # 训练前标签审计，也可单独用 NumPy 运行
requirements_fah_v2.txt
```

Python 3.10+。在已有能训练原模型的 PyTorch 环境中，新增流程仅依赖 NumPy 和 PyTorch，无需为本版训练安装 librosa、PyWavelets、sklearn 或 matplotlib。

```text
python -m pip install -r requirements_fah_v2.txt
```

GPU 训练请根据 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装与你的 GPU/驱动相容的 CUDA 版本。RTX 5070 要使用支持其架构的 PyTorch/CUDA 构建。CPU 合成测试不等于该 GPU 已验证。

## 数据要求

每个传入目录直接包含 `ID_*` 子目录，例如：

```text
train/
  ID_0_wake/*.npz
  ID_1_.../*.npz
  ...
  ID_20_.../*.npz
  ID_21_Pep/*.npz
  ID_22_Mask/*.npz         # 可选背景类
  ID_23_Env/*.npz          # 训练 Gate 时需要
  ID_24_TTS/*.npz          # 可选背景类
val/                      # 独立开发集，至少 ID0、ID21、ID25 或 ID1..20
adapt/                    # 可选，独立训练适配池；不可指向 val/test
FUKE_FAH/                 # 盲测集：ID0、ID21、ID25_CW
```

每个 NPZ 需有 `mfcc[20,T]`、`logmel[40,T]`、`mfcc_gate[60,T]`。同一数据集的 T 必须相同，训练/adapt 合并池也需相同。原特征提取器的 16kHz、1.5s、512 FFT、160 hop、`center=False` 对应 T=147。训练、开发和测试应使用同一特征流水线及裁窗策略。

ID0 是唤醒词。ID1～20 每个 ID 应对应一个固定词/短语，才能合理使用 GE2E 类质心。默认 N=10，需要至少 9 个非空 hard ID；小数据调试可显式减小 `--n-classes`，每个类最好有至少 M 个不同样本。独立训练 ID25 作为异质负样本参加背景训练，不被强行视为一个 GE2E 词类。

`ID_25_noise` 被排除。ID25 的其他目录必须确实是混淆词。最终真人混淆词盲测默认要求 ID25；需要使用 ID1～20 替代评测时，显式传 `--allow-test-cw-substitute`，报告会标注替代协议。

按说话人、原始录音来源划分 train/adapt/val/test，增强兄弟样本放在同一划分。脚本不自动拆分验证集或将其用于训练；会检查同一路径重复，可用 `--verify-content-disjoint` 增加 NPZ 字节级 SHA256 重复检查。内容检查不能识别经过重新压缩、裁切或增强的同源音频，仍需正确划分原始数据。

### 文件夹 ID 与 NPZ class_id 不一致

旧训练脚本以 `ID_*` 文件夹作为类别。部分旧特征在重分类、移动或复用后，NPZ 内的 `class_id`/`label` 可能仍是旧值。也可能是文件确实被放错目录；仅凭报错无法判断哪一方正确。

新版默认 `--label-policy strict`：加载时完整检查标签元数据，训练前报出实际存储值、文件夹期望值以及冲突总数。每个 split 的详细记录写入输出目录的 `train_label_conflicts.csv`、`val_label_conflicts.csv` 等。原始 NPZ 不会被修改。

可先用不依赖 PyTorch 的只读审计命令查看整个训练/验证集：

```text
python fah_labels_v2.py --root "D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake\train" --root "D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake\val" --report-dir "D:\runs\label_audit01"
```

审计会输出 CSV 和 `label_audit_summary.json`；发现冲突时退出码为 1，表示发现数据问题，报告仍完整保存。`wake_nonwake_conflict` 代表正负样本语义冲突，应重点核查原始标注/音频。没有冲突时退出码为 0。

**只有确认文件夹分类正确、NPZ 元数据过期后**，才在训练命令后加 `--label-policy folder`。该选项明确按文件夹生成类别及二值标签，保留警告、完整 CSV 和 `data_summary.json` 审计摘要；不跳过样本、不重写 NPZ。如果文件夹放错，应先根据原始标注修正数据归类。

```text
python train_mymodel_fah_v2.py --label-policy folder --test-dir "D:\myproject\project\Data\Datafeature\FUKE_FAH" --output "D:\runs\fah_v2_run02"
```

### FA/h 的时长口径

当前实现是**独立音频片段协议**：每个负片段最多一次触发，FA/h＝错误触发片段数÷负音频小时数。它不执行连续音频滑窗、事件合并或冷却时间处理。重叠切片不能简单相加时长；即使覆盖总时长，也不能替代流式事件计数。

时长优先级为：显式 CSV > NPZ `duration_sec` > `--sample-duration-sec`（默认 1.5）。加载时输出来源统计，并保存在 `data_summary.json`。

CSV 示例，sid 为相对该数据根目录的路径，使用 `/`：

```csv
sid,duration_sec
ID_0_wake/example0.npz,1.5
ID_21_Pep/example1.npz,1.5
ID_25_CW/example2.npz,1.5
```

如果提供 CSV，必须覆盖该数据集所有加载样本。分别使用 `--train-duration-manifest`、`--val-duration-manifest`、`--adapt-duration-manifest`、`--test-duration-manifest`。

确知完整协议负音频小时数且计数方式仍匹配独立片段时，可用 `--hours-json hours.json` 明确覆盖：

```json
{
  "val": {"real": 10.0, "real_cw": 12.0},
  "test": {"real": 20.0, "real_cw": 24.0}
}
```

这些数字仅示意，必须替换为真实负音频时长，不能包括正样本时长。

## Windows 运行

进入本代码所在目录。默认训练/开发路径与原脚本一致：

```text
D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake\train
D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake\val
```

如果 `FUKE_FAH` 下面直接有 ID 子目录，可执行：

```text
python train_mymodel_fah_v2.py --test-dir "D:\myproject\project\Data\Datafeature\FUKE_FAH" --output "D:\myproject\project\mode_pth\fah_v2_run01"
```

如果 `FUKE_FAH` 下面还有 `test` 这一层，`--test-dir` 应指向该 `test` 目录。输出目录必须为新目录或空目录，防止覆盖已有训练。

显式指定训练/开发/适配路径：

```text
python train_mymodel_fah_v2.py --train-dir "D:\data\train" --val-dir "D:\data\val" --adapt-dir "D:\data\adapt" --test-dir "D:\data\FUKE_FAH" --output "D:\runs\fah_v2_run02"
```

适配集是可选项。不提供时，困难样本仍从训练池挖掘。提供时，独立 adapt 仅在 CL 阶段合并参与训练。训练用的混淆词、说话人和源录音必须与盲测集隔离。

从可信旧模型开始微调，跳过从零训练但继续 Gate/CL：

```text
python train_mymodel_fah_v2.py --init-checkpoint "D:\models\best_ge2e_kws_tiny_gate_system.pth" --epochs 0 --test-dir "D:\data\FUKE_FAH" --output "D:\runs\fah_v2_finetune01"
```

支持旧 `model` 或 `kws_model` 权重键；旧模型没有结构元数据时，`--base-ch`、`--embedding-dim`、`--dropout` 应与旧训练一致。新 checkpoint 自动读取结构配置。此参数是权重 warm-start，不恢复旧优化器或学习率进度。

只重新评测 v2 模型，完全沿用已保存阈值：

```text
python train_mymodel_fah_v2.py --evaluate-checkpoint "D:\runs\fah_v2_run01\best_system.pt" --test-dir "D:\data\FUKE_FAH" --output "D:\runs\fah_v2_locked_test01"
```

完整参数：`python train_mymodel_fah_v2.py --help`。不提供 `--test-dir` 时只输出开发集结果，明确标记盲测未执行。

## 默认训练策略与消融

1. Stage 1：GE2E warmup 10 epochs；随后渐增背景和工作点损失，定期刷新训练池困难样本。两项开发 FRR 的均值选择 checkpoint，记录每项结果。Stage 1 可以存在指标权衡，需查看 history。
2. Gate：仅 ID23 为负，其余为语音通过类；按两个 FAH 工作点联合校准，KWS-only 也是候选。只有两项 FRR 均不退化、均值改善才接受 Gate。
3. CL：默认 1 轮、8 epochs、学习率 8e-6；困难采样、工作点损失、冻结教师稳定性损失。两项 FRR 默认均不可退化，至少一项改善才接受，否则恢复已有模型。随后可做 2 epochs Gate 适配，同样验收/回退。
4. 固定训练集 enrollment 子集重建 wake centroid；最终阈值在开发集冻结后，只测一次盲测集。阈值受控不意味着独立测试一定满足预算，报告会如实标记。

建议保留同一数据划分与种子，分别比较：

| 实验 | 参数 |
| --- | --- |
| FAH 目标对齐基线 | `--hard-positive-fraction 0 --hard-negative-fraction 0 --positive-weight 0 --negative-weight 0 --ranking-weight 0 --cl-rounds 0` |
| 仅困难正负采样 | 保留默认采样，`--positive-weight 0 --negative-weight 0 --ranking-weight 0` |
| 工作点正负损失 | 默认正负权重，`--ranking-weight 0` |
| 完整版本 | 默认参数 |
| KWS-only 对照 | `--disable-gate` |

默认 25% 困难比例与损失权重是实验起点，尚未在你的数据上验证最优。全局时序结构没有无依据地扩大；如近音词仍是主要瓶颈，再比较音节辅助监督等架构实验。

## 输出与推理

```text
best_system.pt                # 模型、可选Gate、训练centroid、阈值、结构与参数
operating_points.json          # real / real_cw / shared 三种工作点
val_calibrated_metrics.json    # 开发集校准结果
test_frozen_metrics.json       # 可选，冻结阈值盲测结果
val_scores.csv / .npz          # 样本得分，可离线分析
test_scores.csv / .npz         # 可选，盲测得分
history.jsonl                  # 每epoch损失、两项指标、挖掘覆盖率
data_summary.json              # 样本数、类别数、时长来源和帧数
config.json / runtime.json
stage1_best.pt / gate_best.pt / cl_01_best.pt / gate_cl_01_best.pt
```

`real` 使用 ID0+ID21，预算 1 FA/h；`real_cw` 使用 ID0+ID21+ID25，预算 20 FA/h。开发集无 ID25 时可以 ID1～20 替代，定义和 test 协议变化写入结果。

两个 benchmark 的 Gate/KWS 阈值可以不同。部署只有一个系统时使用 `shared`，它在开发集同时满足两个预算。CL 验收针对两个 benchmark；shared 也会报告，需单独检查其测试效果。

每个指标包含 FRR、FA/h、FP/FN、负音频小时数、是否达标，以及 `gate_rejected_positive` 和 `kws_rejected_positive` 漏检分解。低误报需要足够负音频时长，零误报并不证明真实误报率为零。

供应用集成的 `predict_batch()` 位于主脚本中，输入已有特征张量，使用 checkpoint 中的 `operating_points['shared']`。该函数将分数转为 float64 后比较，保持精确阈值边界；不要把保存的阈值手动截成 4 位小数或直接用 float32 标量比较。

## 验证

```text
python -m unittest discover -s . -p "test_fah*_v2.py" -v
python train_mymodel_fah_v2.py --smoke-test
```

合成测试会创建独立临时目录、使用小网络和 CPU，执行训练、Gate、CL、保存/加载及冻结评测。输出仅证明执行链路，不作为真实准确率或 FAH 改善证据。真实训练/GPU速度/两个指标的提升仍需要你的数据与权重验证。
