# -*- coding: utf-8 -*-
r"""在混合集上训练同一个网络（双分支、patch=128、batch=4、200 epoch）。

复用 train_rep8 的全部训练机制，只往它的 DOMAINS 里追加一个 mixed 档：超参、
日志格式与 ckpt 命名与单域训练完全相同，唯一差别是数据集。

数据: pt_mixed_iap_p128（由 preprocess_rep8_p128_mixed.py 生成，300/50/50，已离线 IAP）
输出: best_model_4ch_rep8_mixed.pth + _rep8_mixed.log + _rep8_mixed_summary.json
用法: python train_rep8_mixed.py
"""
import os
import sys
import json
import glob

import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import train_rep8 as R                                     # noqa: E402
from train_rep8 import run, DOMAINS                        # noqa: E402

ROOT = os.path.join(BASE, "pt_mixed_iap_p128")
DOMAINS["mixed"] = dict(root=ROOT,
                        ckpt=os.path.join(BASE, "best_model_4ch_rep8_mixed.pth"),
                        tag="MIX")


def preflight():
    """训练前检查数据目录与 patch 大小。"""
    if not os.path.isdir(ROOT):
        raise SystemExit(f"[FAIL] 缺数据目录 {ROOT}，先跑 preprocess_rep8_p128_mixed.py")
    fs = sorted(glob.glob(os.path.join(ROOT, "train", "*.pt")))
    if not fs:
        raise SystemExit(f"[FAIL] {ROOT}/train 里没有 .pt")
    d = torch.load(fs[0], weights_only=True)
    crop = int(d["crop_size"])
    print(f"[PRE] train {len(fs)} 个样本, patch {tuple(d['clean'].shape)}, "
          f"crop={crop} stride={d['stride']} iap={d['iap']}")
    if crop != 128:
        raise SystemExit(f"[FAIL] crop={crop}，不是 128")
    if not d.get("iap"):
        raise SystemExit("[FAIL] 数据不是离线 IAP 的，训练口径会错")
    # 混合集应是两域之和：LLCP 180 + LASER 120 = 300
    n_llc = len([f for f in fs if os.path.basename(f).startswith("llc_")])
    n_las = len([f for f in fs if os.path.basename(f).startswith("las_")])
    print(f"[PRE] 其中 llc_ {n_llc} + las_ {n_las} = {n_llc + n_las}")


if __name__ == "__main__":
    preflight()
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}"
          + (f"  {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    rec = run("mixed")
    p = os.path.join(BASE, "_rep8_mixed_summary.json")
    json.dump([rec], open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote", p)
