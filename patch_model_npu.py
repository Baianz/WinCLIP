"""
WinCLIP NPU Adaptation Patch
=============================
Patches the WinCLIP model class to work on Ascend NPU.

Apply with:  python -c "import patch_model_npu; patch_model_npu.patch()"
Or auto-import:  from patch_model_npu import patched_WinCLIP_forward

Changes:
  - torch.cuda.current_device()  →  torch.npu.current_device()
  - .cuda(device=device)         →  .npu(device=device)
  - Added NPU affinity: compile, combine ops
"""

import torch
import torch_npu
from typing import Optional


def patched_forward(self, image: Optional[torch.Tensor] = None):
    """
    NPU-adapted WinCLIP.forward().
    Replaces all .cuda() with .npu().
    """
    device = torch.npu.current_device()
    dev = torch.device(f"npu:{device}")

    img = image[0]
    img_n1 = image[1]
    img_n2 = image[2]
    img_n3 = image[3]
    img_n4 = image[4]

    window_mask1 = self.window_masks.mask_generate(kernel_size=32, patch_size=16).squeeze()
    window_mask2 = self.window_masks.mask_generate(kernel_size=48, patch_size=16).squeeze()
    window_mask1 = window_mask1.npu(device=dev)
    window_mask2 = window_mask2.npu(device=dev)
    window_masks = [window_mask1, window_mask2]

    if isinstance(img, list):
        pred = []
        for i in range(len(img)):
            image = img[i].npu(device=device)
            normal_img = []
            n1 = img_n1[i].npu(device=device)
            n2 = img_n2[i].npu(device=device)
            n3 = img_n3[i].npu(device=device)
            n4 = img_n4[i].npu(device=device)
            normal_img.extend([n1, n2, n3, n4])
            normal_img = torch.stack(normal_img).squeeze(1)
            F_w, F_p, _ = self.encode_image(image, window_masks=window_masks)
            self.build_image_feature_gallery(normal_img, window_masks)
            visual_anomaly_map = self.calculate_visual_anomaly_score(F_w, F_p)
            anomaly_map = visual_anomaly_map.float()
            am_np = anomaly_map.squeeze(1).cpu().numpy()
            score = float(np.max(am_np))
            pred.append(score)
        score = sum(pred) / len(pred)
    else:
        img = img.npu(device=device)
        normal_img = [
            img_n1.npu(device=device),
            img_n2.npu(device=device),
            img_n3.npu(device=device),
            img_n4.npu(device=device),
        ]
        normal_img = torch.stack(normal_img).squeeze(1)
        F_w, F_p, _ = self.encode_image(img, window_masks=window_masks)
        self.build_image_feature_gallery(normal_img, window_masks)
        visual_anomaly_map = self.calculate_visual_anomaly_score(F_w, F_p)
        anomaly_map = visual_anomaly_map.float()
        score = float(torch.max(anomaly_map).cpu().numpy())
    return score


def patched_calculate_visual_anomaly_score(self, visual_features, F_p):
    """
    NPU-adapted calculate_visual_anomaly_score.
    Replaces .cuda() with .npu().
    """
    import numpy as np
    from einops import rearrange

    feature1, feature2 = visual_features[0], visual_features[1]
    visual_gallery1 = torch.stack(self.visual_gallery1)
    visual_gallery2 = torch.stack(self.visual_gallery2)
    visual_gallery1 = rearrange(visual_gallery1, "a b c d e -> b (a c) d e").squeeze(2)
    visual_gallery2 = rearrange(visual_gallery2, "a b c d e -> b (a c) d e").squeeze(2)
    feature1 = feature1.squeeze(2)
    feature2 = feature2.squeeze(2)
    visual_gallery1 = visual_gallery1.squeeze()
    visual_gallery2 = visual_gallery2.squeeze()

    device = feature1.device

    # Score map 1
    score_map1 = []
    for i in range(len(feature1)):
        features = feature1[i]
        cur_vg = visual_gallery1
        features = F.normalize(features, dim=-1)
        cur_vg = F.normalize(cur_vg, dim=-1)
        tmp = (0.5 * (1 - (features @ cur_vg.T))).min(dim=1).values.cpu()
        score_map1.append(tmp)
    score_map1 = torch.stack(score_map1)

    # Score map 2
    score_map2 = []
    for i in range(len(feature2)):
        features = feature2[i]
        cur_vg = visual_gallery2
        features = F.normalize(features, dim=-1)
        cur_vg = F.normalize(cur_vg, dim=-1)
        tmp = (0.5 * (1 - (features @ cur_vg.T))).min(dim=1).values.cpu()
        score_map2.append(tmp)
    score_map2 = torch.stack(score_map2)

    # Score map 3
    self.fps = self.fps.reshape(-1, 896)
    F_p = F_p.permute(1, 0, 2)
    score_map3 = []
    for i in range(len(F_p)):
        tmp_fps = self.fps
        tmp_fp = F_p[i, :, :]
        tmp_fps = F.normalize(tmp_fps, dim=-1)
        tmp_fp = F.normalize(tmp_fp, dim=-1)
        s = (0.5 * (1.0 - (tmp_fp @ tmp_fps.T))).min(dim=1).values.cpu()
        score_map3.append(s)
    score_map3 = torch.stack(score_map3)

    # Window masks
    window_mask1 = self.window_masks.mask_generate(kernel_size=32, patch_size=16).squeeze()
    window_mask2 = self.window_masks.mask_generate(kernel_size=48, patch_size=16).squeeze()
    window_mask1 = window_mask1.npu(device=device)
    window_mask2 = window_mask2.npu(device=device)

    # Harmonic mean fusion
    score_map1 = score_map1.double()
    score1 = torch.zeros((1, 225), device=device, dtype=torch.double)
    window_mask1 = window_mask1.T.to(device)
    for idx in range(225):
        patch_idx = [bool(torch.isin(idx + 1, mask_patch)) for mask_patch in window_mask1.cpu()]
        sum_num = sum(patch_idx)
        harmonic_sum = torch.sum(1.0 / score_map1[:, patch_idx].to(device), dim=-1)
        score1[:, idx] = sum_num / harmonic_sum
    score1 = score1.reshape(1, 15, 15)

    score_map2 = score_map2.double()
    score2 = torch.zeros((1, 225), device=device, dtype=torch.double)
    window_mask2 = window_mask2.T.to(device)
    for idx in range(225):
        patch_idx = [bool(torch.isin(idx + 1, mask_patch)) for mask_patch in window_mask2.cpu()]
        sum_num = sum(patch_idx)
        harmonic_sum = torch.sum(1.0 / score_map2[:, patch_idx].to(device), dim=-1)
        score2[:, idx] = sum_num / harmonic_sum
    score2 = score2.reshape(1, 15, 15)

    score3 = score_map3.to(device).reshape(1, 15, 15)
    anomaly_map = (score1 + score2 + score3) / 3.0
    anomaly_map = anomaly_map.reshape(
        1, self.grid_size[0], self.grid_size[1]
    ).unsqueeze(1)
    return anomaly_map


def patch(module=None):
    """
    Apply NPU patches to WinCLIP model methods.
    
    Args:
        module: The open_clip.model module. If None, imports it.
    """
    if module is None:
        import open_clip.model as m
    else:
        m = module

    # Save originals for potential restore
    if not hasattr(m.WinCLIP, "_original_forward"):
        m.WinCLIP._original_forward = m.WinCLIP.forward
        m.WinCLIP._original_calculate_visual_anomaly_score = (
            m.WinCLIP.calculate_visual_anomaly_score
        )

    m.WinCLIP.forward = patched_forward
    m.WinCLIP.calculate_visual_anomaly_score = patched_calculate_visual_anomaly_score
    print("[NPU Patch] WinCLIP.forward and calculate_visual_anomaly_score patched for NPU")
    return m


def unpatch(module=None):
    """Restore original methods."""
    if module is None:
        import open_clip.model as m
    else:
        m = module
    if hasattr(m.WinCLIP, "_original_forward"):
        m.WinCLIP.forward = m.WinCLIP._original_forward
        m.WinCLIP.calculate_visual_anomaly_score = (
            m.WinCLIP._original_calculate_visual_anomaly_score
        )
        delattr(m.WinCLIP, "_original_forward")
        delattr(m.WinCLIP, "_original_calculate_visual_anomaly_score")
        print("[NPU Patch] Restored original methods")


# Allow: from patch_model_npu import patched_WinCLIP_forward
# and use as: WinCLIP.forward = patched_WinCLIP_forward
# Need F + np in scope
import torch.nn.functional as F
import numpy as np
