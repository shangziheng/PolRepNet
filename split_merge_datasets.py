"""
Step 1+2: 拆分 llcp_gray 的 test(60) -> val(30)+test(30)，再与 laser 数据集合并为混合数据集。

llcp_gray (4方向灰度, .png, 1024x1224):
    gt/{train(180),test(60)}, noisy/{train(180),test(60)}   (无 val)
laser (4方向灰度, .bmp, 1024x1224):
    gt/{train(120),val(20),test(20)}, noisy/{train(120),val(20),test(20)}

操作:
  A. 把 llcp_gray/gt/test 与 noisy/test 的 60 个样本拆成 val(30) + test(30):
     ID 0001-0030 -> val, 0031-0060 -> test (确定性划分)
  B. 合并成 mixed 数据集，统一为 0/45/90/135.bmp 灰度:
     train = llcp 180 + laser 120 = 300
     val   = llcp  30 + laser  20 =  50
     test  = llcp  30 + laser  20 =  50
     样本名加前缀 llc_ / las_ 防冲突。

运行: python split_merge_datasets.py
"""
import os
import cv2
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))

LLCP_GRAY = os.path.join(BASE, 'llcp_gray')
LASER = os.path.join(BASE, 'laser')
MIXED = os.path.join(BASE, 'mixed')

ANGLES = ['0', '45', '90', '135']
# llcp 的 test 样本 ID (拆分: 前30->val, 后30->test)
LLCP_TEST_IDS = sorted(f for f in os.listdir(os.path.join(LLCP_GRAY, 'gt', 'test'))
                       if os.path.isdir(os.path.join(LLCP_GRAY, 'gt', 'test', f)))
LLCP_VAL_IDS = LLCP_TEST_IDS[:30]
LLCP_TEST_IDS_SPLIT = LLCP_TEST_IDS[30:]


def copy_sample(src_sample, dst_sample):
    """读取4方向图像并统一写出为灰度 .bmp"""
    os.makedirs(dst_sample, exist_ok=True)
    for ang in ANGLES:
        src = None
        for ext in ('.png', '.bmp'):
            p = os.path.join(src_sample, f"{ang}{ext}")
            if os.path.exists(p):
                src = p
                break
        if src is None:
            raise FileNotFoundError(f"{ang} 图像缺失于 {src_sample}")
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取: {src}")
        cv2.imwrite(os.path.join(dst_sample, f"{ang}.bmp"), img)


def split_llcp_test():
    """把 llcp_gray 的 test(60) 拆成 val(30)+test(30)（就地移动）"""
    for sub in ['gt', 'noisy']:
        src_dir = os.path.join(LLCP_GRAY, sub, 'test')
        val_dir = os.path.join(LLCP_GRAY, sub, 'val')
        os.makedirs(val_dir, exist_ok=True)
        for sid in tqdm(LLCP_VAL_IDS, desc=f"拆分 llcp_{sub} test->val"):
            s = os.path.join(src_dir, sid)
            d = os.path.join(val_dir, sid)
            if os.path.isdir(s) and not os.path.isdir(d):
                os.replace(s, d)  # 就地移动到 val
        print(f"[OK] {sub}: val={len(os.listdir(val_dir))}, test={len(os.listdir(src_dir))}")


def merge_subset(sub, split, src_root, prefix):
    """把 src_root/{split} 的样本复制进 mixed/{sub}/{split}，带前缀"""
    src_dir = os.path.join(src_root, sub, split)
    dst_dir = os.path.join(MIXED, sub, split)
    if not os.path.isdir(src_dir):
        print(f"[SKIP] 不存在: {src_dir}")
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    samples = sorted(f for f in os.listdir(src_dir)
                     if os.path.isdir(os.path.join(src_dir, f)))
    for sid in tqdm(samples, desc=f"合并 {prefix} {split}->mixed/{split}"):
        copy_sample(os.path.join(src_dir, sid), os.path.join(dst_dir, f"{prefix}_{sid}"))
    return len(samples)


def build_mixed():
    """合并 llcp_gray(拆分后) 与 laser 为 mixed 数据集"""
    for split in ['train', 'val', 'test']:
        n_las = merge_subset('gt', split, LASER, 'las')
        n_llc = merge_subset('gt', split, LLCP_GRAY, 'llc')
        n_las_n = merge_subset('noisy', split, LASER, 'las')
        n_llc_n = merge_subset('noisy', split, LLCP_GRAY, 'llc')
        print(f"[OK] mixed/{split}: gt=laser{n_las}+llcp{n_llc}={n_las+n_llc}, "
              f"noisy=laser{n_las_n}+llcp{n_llc_n}={n_las_n+n_llc_n}")


if __name__ == "__main__":
    print("===== A. 拆分 llcp_gray test -> val/test =====")
    split_llcp_test()
    print("\n===== B. 合并 mixed 数据集 =====")
    build_mixed()
    print("\n[DONE]")
