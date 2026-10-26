# v2：正常训练、FP32 测试与量化前权重导出

本版使用原来的 `2ch_wavelet_preemph_spec_rewake/train`、`val`、`test` 三个划分训练。`FUKE_FAH` 是后续量化后模拟上板的独立评测集，本训练入口不会加载它，也不要求 ID25。

保留原 lite2 的双分支 DS-CNN + TCN + AttentiveStatsPooling、归一化 embedding、GE2E 和 Tiny Environmental Gate。改进集中在困难样本采样、训练损失、开发集选模和阈值校准；采样和辅助损失不会增加推理网络算子。原脚本保留。

## 文件与环境

以下文件应放在同一目录，例如 `D:\myproject\project\Net\MyModel\Full1\fah_v2`：

```text
train_mymodel_fah_v2.py      主训练入口
fah_model_v2.py             网络、GE2E 和训练辅助损失
fah_data_v2.py              特征加载、标签审计接口、困难采样
fah_labels_v2.py            标签元数据审计
training_metrics_v2.py     正常训练的 sample-FAR 代理指标
training_export_v2.py      FP32 旧式权重导出
fah_metrics_v2.py           精确阈值预算等底层工具
requirements_fah_v2.txt
```

Python 3.10+，依赖 NumPy、PyTorch。在原模型可训练的 Python 环境中执行：

```bat
python -m pip install -r requirements_fah_v2.txt
```

GPU 的 PyTorch/CUDA 构建需与显卡、驱动匹配，参照 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)。本版训练不需要 librosa、PyWavelets、sklearn 或 matplotlib。

## 默认数据与运行

脚本默认使用：

```text
D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake\train
D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake\val
D:\myproject\project\Data\Datafeature\2ch_wavelet_preemph_spec_rewake\test
```

Windows CMD 中进入代码目录即可启动，无需填写测试集参数：

```bat
cd /d "D:\myproject\project\Net\MyModel\Full1\fah_v2"
python train_mymodel_fah_v2.py
```

默认在当前工作目录的 `runs` 下新建时间戳输出目录。需要指定输出位置时，使用新目录或空目录：

```bat
python "D:\myproject\project\Net\MyModel\Full1\fah_v2\train_mymodel_fah_v2.py" --output "D:\myproject\project\mode_pth\fah_v2_run01"
```

请使用原模型训练时的 Python 解释器；如果有多个环境，可将命令中的 `python` 换成该解释器的绝对路径。

PyCharm 中选同一个解释器，Script path 指向本版入口，Parameters 可以留空或只填 `--output "D:\runs\fah_v2_run02"`。**删除旧运行配置中的 FUKE_FAH 路径，以及旧的 `--adapt-dir`、`--hours-json`、`--allow-test-cw-substitute` 参数。** 本版已移除这些参数，且会拒绝指向 `FUKE_FAH` 的 train/val/test 路径。

如普通三划分实际位于其他目录，可分别用 `--train-dir`、`--val-dir`、`--test-dir` 覆盖。`--skip-test` 仅训练和验证；默认会在全部训练完成后评测一次普通 test。

## 数据与标签约定

每个划分目录直接包含 `ID_*/*.npz`，每个 NPZ 需要：

- `mfcc`：`[20,T]`；`logmel`：`[40,T]`；`mfcc_gate`：`[60,T]`。
- 同一划分帧数一致，三划分采用相同特征提取、归一化和裁窗方式。原 16kHz/1.5s/512 FFT/160 hop/`center=False` 对应 T=147。
- ID0 为唤醒词；ID1～20 用作 GE2E 混淆词类，每个 ID 应对应一个固定词或短语。默认 N=10，至少需要 9 个非空 hard ID；M 默认 12。
- 背景类别沿用 ID21/22/23/24 等；没有 ID25 可以正常训练。`ID_25_noise` 排除，存在的 ID25 混淆词可作为异质负样本。

训练入口默认 `--label-policy folder`，恢复原脚本按文件夹 ID 生成类别/二值标签的行为。NPZ 中过期或冲突的 `class_id`/`label` 会警告并写入 `*_label_conflicts.csv`，不会重写 NPZ。请查看报告；若文件确实放错目录，应根据原始标注修正归类。需要元数据和目录完全一致时，使用 `--label-policy strict`，发现冲突会停止。

按说话人和原始录音划分数据，增强兄弟样本放在同一划分。脚本检查跨划分路径重复；`--verify-content-disjoint` 可增加 NPZ 字节级重复检查，但不能识别重新裁切或增强后的同源录音。

## 三划分职责与训练指标

- **train**：参数更新、困难正负样本挖掘、CL 重放式微调、wake enrollment。CL 也只使用 train。
- **val**：选 checkpoint、早停、校准 Gate/KWS 阈值，不参与参数更新或挖掘。
- **test**：全部训练和阈值选择完成后，仅以冻结阈值做一次普通 FP32 测试。

训练代理指标默认是普通负样本 `sample-FAR≤0.0005`、加入混淆词后的全部负样本 `sample-FAR≤0.005`，可用 `--ordinary-far`、`--mixed-far` 调整。这里 sample-FAR＝错误接受的负样本数÷负样本数，**不能当作 1 FA/h、20 FA/h，或量化后指标已改善的证据**。

结果内部保留 `real`、`real_cw` 名称以兼容 checkpoint/损失接口，其当前含义如下：

| 结果键 | 训练代理协议 |
| --- | --- |
| `real` | ID0 对所有普通负类，即排除 ID1～20、ID25 后的非0类，包含 ID21/22/23/24 等 |
| `real_cw` | ID0 对全部非0类，包含当前可用混淆词；无需 ID25 |
| `shared` | 同一 Gate/KWS 阈值对，在 val 同时满足上述预算；另输出全部类别的分组结果 |

独立协议可各自有阈值。导出和 `predict_batch()` 使用 `shared`。每项报告 FRR、实际 sample-FAR、FP/FN、是否满足预算，以及 Gate/KWS 漏检分解；test 不重新扫描或调阈值。时长元数据可保留用于数据核查，但不用于本入口的代理指标换算。

## 优化与对照实验

GE2E 预热后渐增背景和工作点损失，定期从 train 挖掘低分正样本、高分及边界负样本；默认各预留 25% 困难采样。每批保持 GE2E 的 N×M 结构。Gate 在 val 联合校准，旁路也是候选。CL 默认 1 轮、8 epochs、学习率 8e-6；默认要求两个代理 FRR 均不退化且均值改善，否则回退。

建议同一划分、同一种子逐项比较：困难采样关闭、仅困难采样、采样加工作点损失、完整版本、`--disable-gate`。采样对照可设置 `--positive-weight 0 --negative-weight 0 --ranking-weight 0`；关闭困难采样再加 `--hard-positive-fraction 0 --hard-negative-fraction 0`。默认权重是实验起点，需结合真实数据验证。

从可信的旧系统权重开始，跳过从零训练并继续 Gate/CL：

```bat
python train_mymodel_fah_v2.py --init-checkpoint "D:\models\best_ge2e_kws_tiny_gate_system.pth" --epochs 0 --output "D:\runs\fah_v2_finetune01"
```

这是权重 warm-start，优化器重新初始化；旧权重的结构配置必须匹配。完整参数见 `python train_mymodel_fah_v2.py --help`。

## 输出与后续量化

主要输出：`best_system.pt`、`operating_points.json`、`val_calibrated_metrics.json`、`test_frozen_metrics.json`、`val_scores.*`、`test_scores.*`、`history.jsonl`、`data_summary.json`、`data_roles.json` 和标签审计 CSV。

训练结束还会在输出目录导出供后续独立流程使用的 FP32 文件：

```text
best_ge2e_kws.pth                  原式 model 键、wake_centroid、KWS阈值
best_tiny_gate.pth                 原式 gate 键、Gate阈值（存在Gate时）
best_ge2e_kws_tiny_gate_system.pth  kws_model/gate_model、centroid、共享阈值
export_manifest.json              特征接口、模型配置、阈值来源和量化状态
```

导出使用 CPU FP32 权重并保留训练 enrollment centroid。它们不是 INT8、ONNX 或上板二进制。若 Gate 被旁路，阈值可能为0；如明确关闭 Gate，Gate 文件不会生成，下游加载器需支持缺失 Gate。

把导出权重交给你现有的量化/模拟上板流程；量化校准使用合适的 train/val 数据。量化会改变 embedding 或分数时，按该流程重建/校准 centroid 和阈值，再使用 **FUKE_FAH** 做独立的量化后模拟上板验收，最终确认 FRR@≤1 FA/h 和 FRR@≤20 FA/h。

## 验证

```bat
python -m unittest discover -s . -p "test_*v2.py" -v
python train_mymodel_fah_v2.py --smoke-test
```

CPU 合成数据已用于验证训练、采样、Gate/CL、保存加载、冻结测试和 FP32 导出链路。真实数据的效果、GPU 训练与量化后模拟上板尚未验证；最终指标需要按你的独立量化评测流程实测。
