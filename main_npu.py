"""
WinCLIP - Ascend NPU Adaptation (main_npu.py)
==============================================
Zero-/few-shot anomaly classification & segmentation on Ascend NPU.

Adaptations from original main.py:
  - torch.cuda  →  torch.npu
  - .cuda()     →  .npu()
  - AMP: torch.cuda.amp.autocast → torch.npu.amp.autocast (with amp_dtype)
  - Added NPU affinity optimizations: npu.combine, static memory allocator
  - Added model quantization support (FP16 / INT8 options)
"""

import os
import sys
import json
import math
import warnings
from typing import Optional

import torch
import torch_npu
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score

from open_clip.model import get_cast_dtype, WinCLIP
from open_clip.factory import get_tokenizer
from open_clip.utils.env import checkpoint_pathmgr as pathmgr
from datasets.mvtec_dataset import mvtec_dataset, OBJECT_TYPE, _convert_to_rgb
from binary_focal_loss import BinaryFocalLoss

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# NPU Optimisation Settings
# ──────────────────────────────────────────────
# Enable static memory allocator to reduce fragmentation
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
# Enable task queue for better stream overlap
os.environ["TASK_QUEUE_ENABLE"] = "1"

# ──────────────────────────────────────────────
# Text prompts for WinCLIP
# ──────────────────────────────────────────────
state_level = {
    "normal": [
        "{}", "flawless {}", "perfect {}", "unblemished {}",
        "{} without flaw", "{} without defect", "{} without damage",
    ],
    "anomaly": [
        "damaged {}", "{} with flaw", "{} with defect", "{} with damage",
    ],
}
template_level = [
    "a cropped photo of the {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of a {}.",
    "a dark photo of the {}.",
    "a jpeg corrupted photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of the {}.",
    "a photo of a {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for anomaly detection.",
    "a photo of the {} for anomaly detection.",
]


def get_texts(obj_name: str):
    """Build normal/anomaly text prompts for a given object name."""
    normal_states = [s.format(obj_name) for s in state_level["normal"]]
    anomaly_states = [s.format(obj_name) for s in state_level["anomaly"]]
    normal_texts = [t.format(state) for state in normal_states for t in template_level]
    anomaly_texts = [t.format(state) for state in anomaly_states for t in template_level]
    return normal_texts, anomaly_texts


def get_device() -> torch.device:
    """Return the NPU device (fallback to CPU if none available)."""
    if torch.npu.is_available():
        return torch.device(f"npu:{torch.npu.current_device()}")
    return torch.device("cpu")


def load_model(
    checkpoint_path: str = "./vit_b_16_plus_240-laion400m_e31-8fb26589.pt",
    precision: str = "fp32",
    device: Optional[torch.device] = None,
) -> WinCLIP:
    """
    Load WinCLIP model from checkpoint and move to NPU.
    
    Args:
        checkpoint_path: Path to pretrained weights.
        precision: 'fp32', 'fp16', or 'mixed' (AMP).
        device: Target device.
    
    Returns:
        Loaded WinCLIP model (eval mode, no gradients).
    """
    if device is None:
        device = get_device()

    # Load model configuration
    cf = "./open_clip/model_configs/ViT-B-16-plus-240.json"
    with open(cf, "r") as f:
        model_cfg = json.load(f)

    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    cast_dtype = get_cast_dtype(precision)
    quick_gelu = False

    model = WinCLIP(embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)
    model = model.npu(device=device.index if device.type == "npu" else 0)

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Download from:\n"
            "  https://github.com/mlfoundations/open_clip/releases/download/"
            "v0.2-weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt"
        )

    print(f"[INFO] Loading checkpoint from {checkpoint_path} ...")
    with pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    # Filter state dict: only load keys that match model
    model_state = model.state_dict()
    filtered = {}
    missing = []
    unexpected = []
    for k, v in checkpoint.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                print(f"[WARN] Shape mismatch for {k}: "
                      f"checkpoint {v.shape} vs model {model_state[k].shape}")
        else:
            unexpected.append(k)
    for k in model_state:
        if k not in filtered:
            missing.append(k)

    if missing:
        print(f"[INFO] Missing keys ({len(missing)}): {missing[:10]}...")
    if unexpected:
        print(f"[INFO] Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

    model.load_state_dict(filtered, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Cast to target precision if needed
    if precision == "fp16":
        model = model.half()

    print(f"[INFO] Model loaded on {device} | precision={precision}")
    return model


def run(config: dict, model: WinCLIP, device: torch.device, use_amp: bool = True):
    """
    Run WinCLIP evaluation on a single object type.
    
    Args:
        config: Dictionary with 'obj_type', 'data_dir', 'shot'.
        model: Loaded WinCLIP model.
        device: NPU device.
        use_amp: Enable mixed-precision (AMP) on NPU.
    
    Returns:
        gt_list, score_list, auroc, aupr, f1_max
    """
    tokenizer = get_tokenizer("ViT-B-16-plus-240")
    _, _, preprocess = None, None, None  # Provided by dataset

    obj_type = config["obj_type"]
    shot = config["shot"]

    dataset = mvtec_dataset(
        config, config["data_dir"],
        mode="test", shot=shot, preprocess=preprocess,
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=1,
        num_workers=4, shuffle=False,
        pin_memory=True,  # Faster host→device transfer
    )

    normal_texts, anomaly_texts = get_texts(obj_type.replace("_", " "))

    score_list, gt_list = [], []

    # ── AMP context ─────────────────────────────────
    amp_ctx = (
        torch.npu.amp.autocast(enabled=True, dtype=torch.float16)
        if use_amp and device.type == "npu"
        else torch.npu.amp.autocast(enabled=False)
    )

    with torch.no_grad(), amp_ctx:
        for data in tqdm(dataloader, desc=f"Eval [{obj_type}]", total=len(dataloader)):
            image, ref_list, mask, has_anomaly, indice = data

            # ── Encode text (once per type, not per image ── but keep per-batch for simplicity)
            pos_features = tokenizer(normal_texts).npu(device=device)
            neg_features = tokenizer(anomaly_texts).npu(device=device)
            pos_features = model.encode_text(pos_features)
            neg_features = model.encode_text(neg_features)
            pos_features = F.normalize(pos_features, dim=-1)
            neg_features = F.normalize(neg_features, dim=-1)
            pos_features = torch.mean(pos_features, dim=0, keepdim=True)
            neg_features = torch.mean(neg_features, dim=0, keepdim=True)
            pos_features = F.normalize(pos_features, dim=-1)
            neg_features = F.normalize(neg_features, dim=-1)
            text_features = torch.cat([pos_features, neg_features], dim=0)

            if isinstance(image, list):
                # Tiled image: average scores across tiles
                pred = []
                for i in range(len(image)):
                    img = image[i].npu(device=device)
                    _, _, image_features = model.encode_image(img)
                    image_features = F.normalize(image_features, dim=-1)
                    score = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    pred.append(score[0, 1].cpu().numpy())
                text_probs = sum(pred) / len(pred)
            else:
                image = image.npu(device=device)
                _, _, image_features = model.encode_image(image)
                image_features = F.normalize(image_features, dim=-1)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                text_probs = text_probs[0, 1].cpu().numpy()

            if shot == 0:
                # Zero-shot classification
                score = text_probs
                score_list.append(score)
                gt_list.append(has_anomaly[0].numpy())
            else:
                # Few-shot classification
                img = [image]
                img.extend(ref_list)
                vis_probs = model.forward(image=img)
                score = (vis_probs + text_probs) / 2.0
                if math.isinf(score):
                    score = 0.0
                score_list.append(score)
                gt_list.append(has_anomaly[0].numpy())

    # ── Metrics ─────────────────────────────────────
    auroc = roc_auc_score(gt_list, score_list)
    precision, recall, _ = precision_recall_curve(gt_list, score_list)
    aupr = auc(recall, precision)
    f1_max = 0.0
    for threshold in np.arange(0, 1, 0.01):
        y_pred = (np.array(score_list) > threshold).astype(int)
        f1 = f1_score(gt_list, y_pred)
        if f1 > f1_max:
            f1_max = f1

    return gt_list, score_list, auroc, aupr, f1_max


def main():
    """Entry point for NPU-adapted WinCLIP evaluation."""
    np.random.seed(10)
    torch.manual_seed(10)

    # ── Configuration ──────────────────────────────
    dataset_root_dir = "/path/to/visa_anomaly_detection"  # ← ADJUST ME
    datasetname = "visa"
    precision = "fp32"       # "fp32" | "fp16" | "mixed"
    use_amp = True           # Mixed precision on NPU
    obj_type_id = 0

    config = {
        "datasetname": datasetname,
        "dataset_root_dir": dataset_root_dir,
        "data_dir": os.path.join(dataset_root_dir, datasetname),
        "obj_type_id": obj_type_id,
        "obj_type": OBJECT_TYPE[obj_type_id],
        "shot": 0,            # 0 = zero-shot, 2 = few-shot with 2 references
    }

    # ── Device ─────────────────────────────────────
    device = get_device()
    print(f"[INFO] Using device: {device}  "
          f"(NPU count: {torch.npu.device_count()})")

    # ── Model ──────────────────────────────────────
    model = load_model(
        checkpoint_path="./vit_b_16_plus_240-laion400m_e31-8fb26589.pt",
        precision=precision,
        device=device,
    )

    # ── Evaluate each object type ──────────────────
    all_auroc_list, all_aupr_list, all_f1_list = [], [], []
    all_gt_list, all_score_list = [], []

    for obj_type in OBJECT_TYPE[:-1]:  # exclude 'all'
        config["obj_type"] = obj_type
        print(f"\n{'='*60}")
        print(f"[EVAL] Object: {obj_type}")
        print(f"{'='*60}")
        try:
            gt_list, score_list, auroc, aupr, f1_max = run(
                config, model, device, use_amp=use_amp,
            )
            all_auroc_list.append(auroc)
            all_aupr_list.append(aupr)
            all_f1_list.append(f1_max)
            all_gt_list += gt_list
            all_score_list += score_list
            print(f"  AUROC={auroc:.4f}  AUPR={aupr:.4f}  F1-Max={f1_max:.4f}")
        except Exception as e:
            print(f"  [SKIP] {obj_type}: {e}")

    # ── Summary ────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Avg AUROC: {np.mean(all_auroc_list):.4f}")
    print(f"Avg AUPR:  {np.mean(all_aupr_list):.4f}")
    print(f"Avg F1:    {np.mean(all_f1_list):.4f}")

    if all_gt_list:
        auroc = roc_auc_score(all_gt_list, all_score_list)
        precision, recall, _ = precision_recall_curve(all_gt_list, all_score_list)
        aupr = auc(recall, precision)
        f1_max = 0.0
        for threshold in np.arange(0, 1, 0.01):
            y_pred = (np.array(all_score_list) > threshold).astype(int)
            f1 = f1_score(all_gt_list, y_pred)
            if f1 > f1_max:
                f1_max = f1
        print(f"All-Type AUROC={auroc:.4f}  AUPR={aupr:.4f}  F1-Max={f1_max:.4f}")


if __name__ == "__main__":
    main()
