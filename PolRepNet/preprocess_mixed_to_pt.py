"""
Step 3: 对混合数据集 mixed 施加 IAP 亮度归一化，并转为 patch4ch .pt 训练格式。

输入 (4方向灰度 .bmp, 1024x1224):
    mixed/gt/{train,val,test}/{sample}/0.bmp,45.bmp,90.bmp,135.bmp
    mixed/noisy/{train,val,test}/...

处理 (与 preprocess_to_pt.py / 模型 apply_iap_stable 完全一致):
    1. 加载 4 方向灰度，中心裁剪到 IMAGE_SIZE x IMAGE_SIZE (1024x1024)
    2. 对 clean 与 noisy 各自独立施加 IAP:
         s0 = 四方向平均;  v = mean(s0) + k*std(s0);  gamma = (C*margin)/v
         同一 gamma 缩放全部4通道并裁剪到 [0,1]
    3. 提取 CROP_SIZE x CROP_SIZE 补丁 (stride=STRIDE) -> [N,4,crop,crop]
    4. 存 uint8 patch4ch .pt: {'clean','noisy','format':'patch4ch',...,'iap':True}

输出: pt_mixed_iap/{train,val,test}/{sample}.pt   (IAP 后，训练时关闭模型内 IAP)
运行: python preprocess_mixed_to_pt.py
"""
import os
import math
import cv2
import torch
import numpy as np
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))

# ==================== 配置 ====================
MIXED_ROOT = os.path.join(BASE, 'mixed')
OUTPUT_ROOT = os.path.join(BASE, 'pt_mixed_iap')

APPLY_IAP = True        # 离线 IAP，训练时 iap_enabled=False
IMAGE_SIZE = 1024
CROP_SIZE = 256
STRIDE = 256
SPLITS = ['train', 'val', 'test']
ANGLES = ['0', '45', '90', '135']
EXT = '.bmp'
# =============================================


def load_4ch(folder_path):
    """加载4张灰度BMP，中心裁剪到 IMAGE_SIZE×IMAGE_SIZE，返回 float32 [4,H,W] (0~1)"""
    tensors = []
    for ang in ANGLES:
        p = os.path.join(folder_path, f"{ang}{EXT}")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(p)
        h, w = img.shape[:2]
        sy, sx = (h - IMAGE_SIZE) // 2, (w - IMAGE_SIZE) // 2
        img = img[sy:sy + IMAGE_SIZE, sx:sx + IMAGE_SIZE]
        tensors.append(torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0))
    return torch.cat(tensors, dim=0)  # [4,H,W]


def apply_iap_4ch(img, C=0.95, safety_margin=0.8, k=2.0, eps=1e-8):
    """
    与模型 apply_iap_stable 一致的 IAP 归一化 (对 [4,H,W]):
        s0 = 四方向平均;  v = mean(s0) + k*std(s0);  gamma = (C*margin)/v
        同一 gamma 缩放全部4通道并裁剪到 [0,1]
    """
    s0 = img.mean(dim=0, keepdim=True)          # [1,H,W]
    mean = s0.mean()
    std = s0.std()
    v = torch.clamp(mean + k * std, min=eps)
    gamma = (C * safety_margin) / v
    return torch.clamp(img * gamma, 0.0, 1.0), gamma


def extract_patches(img, crop_size, stride):
    """按行列顺序提取补丁，返回 [N, C, crop, crop]"""
    C, H, W = img.shape
    num_x = math.floor((H - crop_size) / stride) + 1
    patches = []
    for i in range(num_x):          # 行 (y)
        for j in range(num_x):      # 列 (x)
            cy = min(i * stride, H - crop_size)
            cx = min(j * stride, W - crop_size)
            patches.append(img[:, cy:cy + crop_size, cx:cx + crop_size])
    return torch.stack(patches, dim=0)  # [N, C, crop, crop]


def save_sample(gt_dir, noisy_dir, sample, out_dir):
    clean_4 = load_4ch(os.path.join(gt_dir, sample))
    noisy_4 = load_4ch(os.path.join(noisy_dir, sample))
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


def process_split(split):
    gt_dir = os.path.join(MIXED_ROOT, 'gt', split)
    noisy_dir = os.path.join(MIXED_ROOT, 'noisy', split)
    if not os.path.isdir(gt_dir) or not os.path.isdir(noisy_dir):
        print(f"[SKIP] 目录不存在: {gt_dir} 或 {noisy_dir}")
        return
    out_dir = os.path.join(OUTPUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)
    samples = sorted(f for f in os.listdir(gt_dir)
                     if os.path.isdir(os.path.join(gt_dir, f)))
    for sample in tqdm(samples, desc=f"预处理 {split}"):
        save_sample(gt_dir, noisy_dir, sample, out_dir)
    print(f"[OK] {split}: {len(samples)} 个样本 -> {out_dir}")


def main():
    for split in SPLITS:
        process_split(split)


if __name__ == "__main__":
    main()
