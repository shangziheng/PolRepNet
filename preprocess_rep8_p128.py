# -*- coding: utf-8 -*-
"""按 patch=128 重新预处理 LLCP / LASER 的 IAP .pt 数据。

.pt 是预裁切补丁包（clean/noisy 形如 [N,4,crop,crop]），patch 大小在预处理阶段就固化；
训练时 BranchDataset 走 _pt_is_patch 分支只做索引，crop_size 参数不生效，
因此 patch=128 必须在数据侧实现。

做法：import preprocess_separate_to_pt 后覆盖其模块级 CROP_SIZE / STRIDE / DATASETS，
其余（中心裁 1024 -> IAP -> 分块）复用同一实现，保证与 pt_*_iap 同一口径。

    源 llcp_gray -> pt_llcp_iap_p128    (train180 / val30 / test30)
    源 laser     -> pt_laser_iap_p128   (train120 / val20 / test20)

用法: python preprocess_rep8_p128.py
"""
import os
import sys
import importlib.util
import shutil

import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

CROP = 128
STRIDE = 128

_spec = importlib.util.spec_from_file_location(
    "_pp_sep", os.path.join(BASE, "preprocess_separate_to_pt.py"))
PP = importlib.util.module_from_spec(_spec)
sys.modules["_pp_sep"] = PP
_spec.loader.exec_module(PP)

# 只改这三处，其余逻辑复用 preprocess_separate_to_pt
PP.CROP_SIZE = CROP
PP.STRIDE = STRIDE
PP.DATASETS = [
    (os.path.join(BASE, "llcp_gray"), os.path.join(BASE, "pt_llcp_iap_p128"), ".png"),
    (os.path.join(BASE, "laser"), os.path.join(BASE, "pt_laser_iap_p128"), ".bmp"),
]

if __name__ == "__main__":
    print(f"重新预处理: CROP_SIZE={PP.CROP_SIZE} STRIDE={PP.STRIDE}")
    for _s, out, _e in PP.DATASETS:
        if os.path.isdir(out):
            print(f"[CLEAN] 清空旧目录 {out}")
            shutil.rmtree(out)
    PP.main()

    # 校验：样本数与张量形状；若本地还留有 patch=256 的旧数据，再核对其像素是否一致
    print("\n===== 校验 =====")
    for tag, new_root, old_root in (("LLCP", "pt_llcp_iap_p128", "pt_llcp_iap"),
                                    ("LASER", "pt_laser_iap_p128", "pt_laser_iap")):
        nd = os.path.join(BASE, new_root, "train")
        od = os.path.join(BASE, old_root, "train")
        f = sorted(os.listdir(nd))[0]
        new = torch.load(os.path.join(nd, f), weights_only=True)
        print(f"[{tag}] {f}: patch {tuple(new['clean'].shape)} "
              f"crop={new['crop_size']} stride={new['stride']} iap={new['iap']}")
        if os.path.isfile(os.path.join(od, f)):
            # patch=256 的第 0 块左上角 128x128 应与新 patch=128 的第 0 块逐像素一致
            old = torch.load(os.path.join(od, f), weights_only=True)
            a = new["clean"][0, :, :CROP, :CROP].float() / 255.0
            b = old["clean"][0, :, :CROP, :CROP].float() / 255.0
            d = float((a - b).abs().max())
            print(f"       patch 对齐检查 max|Δ| = {d:.6f} "
                  f"({'一致' if d == 0 else '不一致'})")
        for split in ("train", "val", "test"):
            p = os.path.join(BASE, new_root, split)
            if os.path.isdir(p):
                print(f"       {split}: {len(os.listdir(p))} 个样本")
