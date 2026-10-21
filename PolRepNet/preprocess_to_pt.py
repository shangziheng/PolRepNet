"""
预处理脚本: 将 data/gt 与 data/noisy 下的 BMP 数据批量转换为 .pt 文件（4通道灰度偏振）

IAP: 对每张图（clean 与 noisy 各自独立）施加亮度归一化，消除两数据集间的亮度差异，
     gamma = (C*margin) / (mean(s0) + k*std(s0))，同一 gamma 缩放全部4通道并裁剪到 [0,1]。
     归一化后训练可关闭模型内的 IAP（iap_enabled=False）。

每个样本保存为一个 .pt 文件（含预裁切的补丁，训练时不再读 BMP / 不再实时裁切）:
    {
        'clean': uint8 [N, 4, crop_size, crop_size],   # 4方向 (I0,I45,I90,I135) 灰度 (IAP后)
        'noisy': uint8 [N, 4, crop_size, crop_size],
        'format': 'patch4ch',
        'image_size': 1024,
        'crop_size': 256,
        'stride': 256,
        'name': folder,
        'iap': True,
    }

输出目录: data/pt_iap/{train,val,test}/{sample_id}.pt
运行: python preprocess_to_pt.py
"""
import os
import math
import cv2
import torch
import numpy as np
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))

CLEAN_ROOT = os.path.join(BASE, 'data', 'gt')
NOISY_ROOT = os.path.join(BASE, 'data', 'noisy')
# 是否对 clean 和 noisy 分别施加 IAP 亮度归一化（消除两数据集亮度差异，训练时关掉模型内 IAP）
APPLY_IAP = True
OUTPUT_ROOT = os.path.join(BASE, 'data', 'pt_iap')

IMAGE_SIZE = 1024
CROP_SIZE = 256
STRIDE = 256
SPLITS = ['train', 'val', 'test']
ANGLES = ['0', '45', '90', '135']


def load_4ch(folder_path):
    """加载4张灰度BMP，中心裁剪到 IMAGE_SIZE×IMAGE_SIZE，返回 float32 [4,H,W] (0~1)"""
    tensors = []
    for ang in ANGLES:
        p = os.path.join(folder_path, f"{ang}.bmp")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)  # 单通道灰度
        if img is None:
            raise FileNotFoundError(p)
        h, w = img.shape[:2]
        sy, sx = (h - IMAGE_SIZE) // 2, (w - IMAGE_SIZE) // 2
        img = img[sy:sy + IMAGE_SIZE, sx:sx + IMAGE_SIZE]
        tensors.append(torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0))  # [1,H,W]
    return torch.cat(tensors, dim=0)  # [4,H,W]


def apply_iap_4ch(img, C=0.95, safety_margin=0.8, k=2.0, eps=1e-8):
    """
    对 [4,H,W] 灰度偏振图施加与模型 apply_iap_stable 完全一致的 IAP 归一化:
        s0 = 四方向平均;  v = mean(s0) + k*std(s0);  gamma = (C*margin) / v
        同一 gamma 缩放全部4通道并裁剪到 [0,1]
    """
    s0 = img.mean(dim=0, keepdim=True)          # [1,H,W]
    mean = s0.mean()
    std = s0.std()                              # 无偏标准差，与 torch.std 默认一致
    v = torch.clamp(mean + k * std, min=eps)
    gamma = (C * safety_margin) / v
    return torch.clamp(img * gamma, 0.0, 1.0), gamma


def extract_patches(img, crop_size, stride):
    """按行列顺序提取补丁，返回 [N, C, crop, crop]（与数据集 crop_positions 顺序一致）"""
    C, H, W = img.shape
    num_x = math.floor((H - crop_size) / stride) + 1
    patches = []
    for i in range(num_x):          # 行 (y)
        for j in range(num_x):      # 列 (x)
            cy = min(i * stride, H - crop_size)
            cx = min(j * stride, W - crop_size)
            patches.append(img[:, cy:cy + crop_size, cx:cx + crop_size])
    return torch.stack(patches, dim=0)  # [N, C, crop, crop]


def process_split(split):
    clean_dir = os.path.join(CLEAN_ROOT, split)
    noisy_dir = os.path.join(NOISY_ROOT, split)
    if not os.path.isdir(clean_dir) or not os.path.isdir(noisy_dir):
        print(f"[SKIP] 目录不存在: {clean_dir} 或 {noisy_dir}")
        return
    out_dir = os.path.join(OUTPUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)

    folders = sorted(f for f in os.listdir(clean_dir)
                     if os.path.isdir(os.path.join(clean_dir, f)))
    for folder in tqdm(folders, desc=f"预处理 {split}"):
        clean_4 = load_4ch(os.path.join(clean_dir, folder))
        noisy_4 = load_4ch(os.path.join(noisy_dir, folder))
        if APPLY_IAP:
            # clean 与 noisy 各自独立施加 IAP（各自计算各自的 gamma）
            clean_4, _ = apply_iap_4ch(clean_4)
            noisy_4, _ = apply_iap_4ch(noisy_4)
        clean_patch = extract_patches(clean_4, CROP_SIZE, STRIDE)
        noisy_patch = extract_patches(noisy_4, CROP_SIZE, STRIDE)
        # 存为 uint8（值域 0~1 -> 0~255，无损）缩小文件体积
        data = {
            'clean': (clean_patch * 255.0).round().to(torch.uint8),
            'noisy': (noisy_patch * 255.0).round().to(torch.uint8),
            'format': 'patch4ch',
            'image_size': IMAGE_SIZE,
            'crop_size': CROP_SIZE,
            'stride': STRIDE,
            'name': folder,
            'iap': APPLY_IAP,
        }
        torch.save(data, os.path.join(out_dir, f"{folder}.pt"))
    print(f"[OK] {split}: {len(folders)} 个样本 -> {out_dir}")


def main():
    for split in SPLITS:
        process_split(split)


if __name__ == "__main__":
    main()
