"""
生成两份不混合的 IAP .pt 训练集（用于与混合训练对比）:
    pt_llcp_iap/   <- llcp_gray (train180 / val30 / test30)
    pt_laser_iap/  <- laser     (train120 / val20 / test20)

处理与 preprocess_mixed_to_pt.py 完全一致: 中心裁剪1024 -> IAP -> 256补丁(stride256) -> patch4ch .pt
运行: python preprocess_separate_to_pt.py
"""
import os
import math
import cv2
import torch
import numpy as np
from tqdm import tqdm

# ==================== 配置 ====================
BASE = os.path.dirname(os.path.abspath(__file__))
# 每份数据集: (源根, 输出目录, 图像扩展名)
DATASETS = [
    (os.path.join(BASE, 'llcp_gray'), os.path.join(BASE, 'pt_llcp_iap'), '.png'),
    (os.path.join(BASE, 'laser'),     os.path.join(BASE, 'pt_laser_iap'), '.bmp'),
]

APPLY_IAP = True
IMAGE_SIZE = 1024
CROP_SIZE = 256
STRIDE = 256
SPLITS = ['train', 'val', 'test']
ANGLES = ['0', '45', '90', '135']
# =============================================


def load_4ch(folder_path, ext):
    tensors = []
    for ang in ANGLES:
        p = os.path.join(folder_path, f"{ang}{ext}")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(p)
        h, w = img.shape[:2]
        sy, sx = (h - IMAGE_SIZE) // 2, (w - IMAGE_SIZE) // 2
        img = img[sy:sy + IMAGE_SIZE, sx:sx + IMAGE_SIZE]
        tensors.append(torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0))
    return torch.cat(tensors, dim=0)


def apply_iap_4ch(img, C=0.95, safety_margin=0.8, k=2.0, eps=1e-8):
    s0 = img.mean(dim=0, keepdim=True)
    mean = s0.mean()
    std = s0.std()
    v = torch.clamp(mean + k * std, min=eps)
    gamma = (C * safety_margin) / v
    return torch.clamp(img * gamma, 0.0, 1.0), gamma


def extract_patches(img, crop_size, stride):
    C, H, W = img.shape
    num_x = math.floor((H - crop_size) / stride) + 1
    patches = []
    for i in range(num_x):
        for j in range(num_x):
            cy = min(i * stride, H - crop_size)
            cx = min(j * stride, W - crop_size)
            patches.append(img[:, cy:cy + crop_size, cx:cx + crop_size])
    return torch.stack(patches, dim=0)


def save_sample(gt_dir, noisy_dir, sample, out_dir, ext):
    clean_4 = load_4ch(os.path.join(gt_dir, sample), ext)
    noisy_4 = load_4ch(os.path.join(noisy_dir, sample), ext)
    if APPLY_IAP:
        clean_4, _ = apply_iap_4ch(clean_4)
        noisy_4, _ = apply_iap_4ch(noisy_4)
    clean_patch = extract_patches(clean_4, CROP_SIZE, STRIDE)
    noisy_patch = extract_patches(noisy_4, CROP_SIZE, STRIDE)
    data = {
        'clean': (clean_patch * 255.0).round().to(torch.uint8),
        'noisy': (noisy_patch * 255.0).round().to(torch.uint8),
        'format': 'patch4ch',
        'image_size': IMAGE_SIZE,
        'crop_size': CROP_SIZE,
        'stride': STRIDE,
        'name': sample,
        'iap': APPLY_IAP,
    }
    torch.save(data, os.path.join(out_dir, f"{sample}.pt"))


def process_dataset(src_root, out_root, ext):
    for split in SPLITS:
        gt_dir = os.path.join(src_root, 'gt', split)
        noisy_dir = os.path.join(src_root, 'noisy', split)
        if not os.path.isdir(gt_dir) or not os.path.isdir(noisy_dir):
            print(f"[SKIP] {src_root}: 缺 {split} 目录")
            continue
        out_dir = os.path.join(out_root, split)
        os.makedirs(out_dir, exist_ok=True)
        samples = sorted(f for f in os.listdir(gt_dir)
                         if os.path.isdir(os.path.join(gt_dir, f)))
        for sample in tqdm(samples, desc=f"{os.path.basename(out_root)}/{split}"):
            save_sample(gt_dir, noisy_dir, sample, out_dir, ext)
        print(f"[OK] {out_root}/{split}: {len(samples)} 个样本")


def main():
    for src_root, out_root, ext in DATASETS:
        print(f"===== {os.path.basename(out_root)} (源: {src_root}) =====")
        process_dataset(src_root, out_root, ext)


if __name__ == "__main__":
    main()
