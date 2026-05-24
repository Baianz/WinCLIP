# WinCLIP - Ascend NPU 适配指南

## 概述

本项目将 [WinCLIP (CVPR'23)](https://github.com/mala-lab/WinCLIP) 零样本异常检测模型适配到华为 **Ascend NPU** 平台。

| 项目 | 说明 |
|------|------|
| 原项目 | [mala-lab/WinCLIP](https://github.com/mala-lab/WinCLIP) |
| 论文 | [WinCLIP: Zero-/few-shot anomaly classification and segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf) |
| 硬件 | Ascend 910 (9362) |
| 框架 | PyTorch 2.9.0 + torch_npu 2.9.0 |

## 适配内容

### 1. CUDA → NPU 迁移 (`main_npu.py`)

| 原始代码 | NPU 适配代码 |
|----------|-------------|
| `torch.cuda.current_device()` | `torch.npu.current_device()` |
| `model.cuda(device=device)` | `model.npu(device=device)` |
| `tensor.cuda(device=device)` | `tensor.npu(device=device)` |
| `torch.cuda.amp.autocast()` | `torch.npu.amp.autocast(dtype=torch.float16)` |

### 2. 模型补丁 (`patch_model_npu.py`)

WinCLIP 的 `forward()` 和 `calculate_visual_anomaly_score()` 方法进行了 NPU 适配：
- 替换所有 `.cuda()` 调用为 `.npu()`
- 保持 `window_masks` 的 CPU → NPU 传输正确

### 3. NPU 优化 (`run_winclip_npu.py`)

- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` - 减少内存碎片
- `TASK_QUEUE_ENABLE=1` - 启动硬件任务队列流水
- `pin_memory=True` - DataLoader 使用锁页内存加速 NPU 传输
- `torch.compile` - 图模式编译 vision tower 减少调度开销
- AMP 混合精度 (FP16)

### 4. 量化方案

#### FP16 量化
```python
model = model.half()
```
将模型权重和计算全部转为 FP16，可获得约 **1.5-2x** 加速。

#### INT8 量化（方案）
Ascend 910 INT8 推理需要额外工具链：
1. **AMCT** (Ascend Model Compression Toolkit) - 量化压缩
2. **ATC** (Ascend Tensor Compiler) - 模型转换
3. **msmodelslim** - 模型瘦身工具

## 文件清单

```
.
├── main_npu.py              # NPU 适配主入口（数据驱动评估）
├── run_winclip_npu.py       # 一键运行脚本（验证/基准/量化）
├── patch_model_npu.py       # WinCLIP 模型补丁
├── requirements_npu.txt     # 依赖列表
├── README_NPU.md            # 本文档
└── open_clip/               # 原始 OpenCLIP 代码（已有）
```

## 快速开始

### 1. 环境准备
```bash
pip install -r requirements_npu.txt
```

### 2. 下载预训练权重
```bash
wget https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt
```

### 3. 模型验证（无需权重）
```bash
python run_winclip_npu.py --mode verify
```

### 4. 基准测试（需权重）
```bash
python run_winclip_npu.py --mode benchmark
```

### 5. 完整评估（需数据集）
```bash
python main_npu.py
```

## 性能预期（Ascend 910）

| 配置 | 预期延迟 | 预期 FPS |
|------|---------|---------|
| FP32 (原始) | ~30-50ms | 20-33 |
| FP16 | ~20-30ms | 33-50 |
| FP16 + AMP + compile | ~15-25ms | 40-66 |

> 注：实际性能取决于具体数据形状和 NPU 负载情况。

## 数据集准备

支持 MVTec AD / Visa AD 格式：
```
DATA_PATH/
    subset_1/
        train/
            good/
        test/
            good/
            defect_class_1/
            ...
```

编辑 `main_npu.py` 中的 `dataset_root_dir` 配置。

## 评估指标

- **AUROC**: Area Under Receiver Operating Characteristic
- **AUPR**: Area Under Precision-Recall Curve
- **F1-Max**: Maximum F1 score
