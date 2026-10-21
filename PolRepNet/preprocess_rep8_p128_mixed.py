# -*- coding: utf-8 -*-
r"""按 patch=128 预处理混合集的 IAP .pt。

单独一份是为了不重跑已有数据：本脚本只做混合集，pt_llcp_iap_p128 / pt_laser_iap_p128
一概不碰。

做法与 preprocess_rep8_p128.py 同源：import preprocess_mixed_to_pt 后覆盖
CROP_SIZE / STRIDE / OUTPUT_ROOT，其余（中心裁 1024 -> 离线 IAP -> 分块 -> 存 uint8）
全部复用，保证与 pt_mixed_iap 同一口径、只差 patch 大小。

    源 mixed/gt|noisy -> pt_mixed_iap_p128   (train300 / val50 / test50)

用法: python preprocess_rep8_p128_mixed.py
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
NEW_ROOT = "pt_mixed_iap_p128"
OLD_ROOT = "pt_mixed_iap"

_spec = importlib.util.spec_from_file_location(
    "_pp_mix", os.path.join(BASE, "preprocess_mixed_to_pt.py"))
PP = importlib.util.module_from_spec(_spec)
sys.modules["_pp_mix"] = PP
_spec.loader.exec_module(PP)

# 只改这三处，其余逻辑复用 preprocess_mixed_to_pt
PP.CROP_SIZE = CROP
PP.STRIDE = STRIDE
PP.OUTPUT_ROOT = os.path.join(BASE, NEW_ROOT)

if __name__ == "__main__":
    out = PP.OUTPUT_ROOT
    print(f"重新预处理混合集: CROP_SIZE={PP.CROP_SIZE} STRIDE={PP.STRIDE}")
    print(f"  源 {PP.MIXED_ROOT}")
    print(f"  -> {out}")
    if os.path.isdir(out):
        print(f"[CLEAN] 清空旧目录 {out}")
        shutil.rmtree(out)
    PP.main()

    print("\n===== 校验 =====")
    ok = True
    for split in PP.SPLITS:
        nd = os.path.join(out, split)
        n_new = len(os.listdir(nd)) if os.path.isdir(nd) else -1
        if n_new <= 0:
            ok = False
        print(f"[{split}] {n_new} 个样本  {'OK' if n_new > 0 else '!! 缺数据'}")

    nd = os.path.join(out, "train")
    f = sorted(os.listdir(nd))[0]
    new = torch.load(os.path.join(nd, f), weights_only=True)
    print(f"[train] {f}: patch {tuple(new['clean'].shape)} "
          f"crop={new['crop_size']} stride={new['stride']} iap={new['iap']}")
    if int(new["crop_size"]) != CROP or not new.get("iap"):
        ok = False

    # 若本地还留有 patch=256 的旧数据，核对第 0 块左上 128x128 是否逐像素一致
    od = os.path.join(BASE, OLD_ROOT, "train")
    if os.path.isfile(os.path.join(od, f)):
        old = torch.load(os.path.join(od, f), weights_only=True)
        d = float((new["clean"][0, :, :CROP, :CROP].float()
                   - old["clean"][0, :, :CROP, :CROP].float()).abs().max())
        print(f"       patch 对齐检查 max|Δ| = {d:.6f} "
              f"({'一致' if d == 0 else '不一致'})")
        if d != 0:
            ok = False

    if not ok:
        raise SystemExit("[FAIL] 预处理校验未通过")
    print("[OK] 混合集 p128 数据就绪")
