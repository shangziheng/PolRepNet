"""
将 LLCP 彩色 PNG 数据集转换为灰度版本（不裁切、不做 .pt、不缩尺寸）

LLCP 源数据 (彩色 RGB, 1024x1224):
    LLCP/GT/{train,test}/{sample}/0.png, 45.png, 90.png, 135.png
    LLCP/Noisy/{train,test}/{sample}/0.png, 45.png, 90.png, 135.png

输出 (灰度, 单通道, 原尺寸, 与原 data/gt + data/noisy 相同的文件夹结构):
    data/llcp_gray/gt/{train,test}/{sample}/0.png, 45.png, 90.png, 135.png
    data/llcp_gray/noisy/{train,test}/{sample}/...

灰度转换与代码内 rgb_to_gray_weights [0.299,0.587,0.114] (BT.601) 一致，
即 cv2.cvtColor(..., COLOR_BGR2GRAY)。

运行: python convert_llcp_to_gray.py
"""
import os
import cv2
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))

# ==================== 配置 ====================
LLCP_ROOT = os.path.join(BASE, 'LLCP')
OUTPUT_ROOT = os.path.join(BASE, 'data', 'llcp_gray')
SPLITS = ['train', 'test']
ANGLES = ['0', '45', '90', '135']
# =============================================


def convert_dir(src_dir, dst_dir):
    if not os.path.isdir(src_dir):
        print(f"[SKIP] 目录不存在: {src_dir}")
        return 0
    samples = sorted(f for f in os.listdir(src_dir)
                     if os.path.isdir(os.path.join(src_dir, f)))
    for sample in tqdm(samples, desc=os.path.basename(os.path.dirname(dst_dir)) + "/" + os.path.basename(dst_dir)):
        src_sample = os.path.join(src_dir, sample)
        dst_sample = os.path.join(dst_dir, sample)
        os.makedirs(dst_sample, exist_ok=True)
        for ang in ANGLES:
            img = cv2.imread(os.path.join(src_sample, f"{ang}.png"), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(os.path.join(src_sample, f"{ang}.png"))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # BT.601, 与代码内权重一致
            cv2.imwrite(os.path.join(dst_sample, f"{ang}.png"), gray)
    return len(samples)


def main():
    for split in SPLITS:
        n_gt = convert_dir(os.path.join(LLCP_ROOT, 'GT', split),
                           os.path.join(OUTPUT_ROOT, 'gt', split))
        n_noisy = convert_dir(os.path.join(LLCP_ROOT, 'Noisy', split),
                              os.path.join(OUTPUT_ROOT, 'noisy', split))
        print(f"[OK] {split}: gt={n_gt}, noisy={n_noisy}")


if __name__ == "__main__":
    main()