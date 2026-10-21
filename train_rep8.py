# -*- coding: utf-8 -*-
"""
在单个域上训练 PolRepNet-2（base_ch=8, rep_scale=8, patch=128, batch_size=4）。

训练主循环、损失、优化器与调度器全部来自 trainer.Trainer；本脚本只注入模型的
两个超参（base_ch / rep_scale）与 ckpt 落点，故与默认配方相比只有 patch 大小不同。

数据：pt_llcp_iap_p128 / pt_laser_iap_p128（已离线施加 IAP 的 uint8 补丁包）
输出：best_model_4ch_rep8_<域>.pth + _rep8_<域>.log + _rep8_summary.json

用法:
    python train_rep8.py              # LLCP -> LASER 顺序跑
    python train_rep8.py llcp         # 只跑 LLCP
"""
import os
import sys
import json
import time

import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import trainer as T                                        # noqa: E402
from trainer import Trainer                                # noqa: E402

REP_SCALE = 8
BASE_CH = 8
CROP = 128         # 仅用于日志记录；真正的 patch 大小来自 .pt 数据
STRIDE = 128
EPOCHS = 200
BATCH = 4          # 训练循环内固定为 4

DOMAINS = {
    "llcp": dict(root=os.path.join(BASE, "pt_llcp_iap_p128"),
                 ckpt=os.path.join(BASE, "best_model_4ch_rep8_llcp.pth"),
                 tag="LLCP"),
    "laser": dict(root=os.path.join(BASE, "pt_laser_iap_p128"),
                  ckpt=os.path.join(BASE, "best_model_4ch_rep8_laser.pth"),
                  tag="LASER"),
}

# --------------------------------------------------------------------------
# 注入 base_ch / rep_scale：替换 trainer 模块里的模型构造符号
# --------------------------------------------------------------------------
_ORIG_MODEL = T.PolRepNet


def _polrepnet_rep8(*args, **kwargs):
    kwargs["base_ch"] = BASE_CH
    kwargs["rep_scale"] = REP_SCALE
    return _ORIG_MODEL(*args, **kwargs)


T.PolRepNet = _polrepnet_rep8


class _Tee:
    """把 stdout 同时写到控制台与日志文件。"""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


class TrainerRep8(Trainer):
    """只改动 ckpt 落点的 Trainer。"""

    def __init__(self, *args, ckpt, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = ckpt
        n_tr = sum(p.numel() for p in self.model.parameters())
        n_inf = None
        try:                       # 训练期参数量 / 融合后推理参数量
            n_inf = sum(p.numel() for p in self.model.slim().parameters())
        except Exception as e:
            print(f"[WARN] slim() 统计失败: {e}")
        ds = self.train_dataset
        sample = ds[0][0]        # 实际取出的 patch，用于打印 patch 大小
        print(f"[REP8] base_ch={BASE_CH} rep_scale={REP_SCALE} batch={BATCH}  "
              f"log 记 crop={CROP} stride={STRIDE}")
        print(f"[REP8] 实际 patch {tuple(sample.shape)} × "
              f"{ds.num_crops_per_image}/图, 训练集 {len(ds)} 个 patch, "
              f"验证集 {len(self.val_dataset)} 个")
        print(f"[REP8] 训练期参数量 {n_tr:,} ({n_tr / 1e6:.3f} M)"
              + (f" / 融合后推理参数量 {n_inf:,} ({n_inf / 1e6:.3f} M)"
                 if n_inf else ""))


def run(key):
    d = DOMAINS[key]
    log = os.path.join(BASE, f"_rep8_{key}.log")
    print("\n" + "=" * 78)
    print(f"[REP8] {d['tag']}  root={os.path.basename(d['root'])}  "
          f"epochs={EPOCHS}  log={os.path.basename(log)}")
    print("=" * 78)
    tee = _Tee(log)
    old = sys.stdout
    sys.stdout = tee
    t0 = time.time()
    try:
        tr = TrainerRep8(d["root"], "",
                         use_pt=True, channels="4ch",
                         use_amp=False, iap_enabled=False,
                         pretrained_path=None,
                         normalize_brightness=False,
                         freeze_branch=None,
                         ckpt=d["ckpt"])
        tr.train(EPOCHS)
        ok = True
        err = None
    except Exception as e:
        import traceback
        traceback.print_exc()
        ok, err = False, f"{type(e).__name__}: {e}"
    finally:
        sys.stdout = old
        tee.flush()
    dt = time.time() - t0
    rec = dict(key=key, tag=d["tag"], rep_scale=REP_SCALE, base_ch=BASE_CH,
               crop=CROP, stride=STRIDE, batch=BATCH, epochs=EPOCHS,
               ckpt=os.path.basename(d["ckpt"]), seconds=dt, ok=ok, error=err)
    print(f"[REP8] {d['tag']} 完成={ok}  用时 {dt / 60:.1f} min"
          + (f"  err={err}" if err else ""))
    del tr
    torch.cuda.empty_cache()
    return rec


if __name__ == "__main__":
    want = sys.argv[1:] or ["llcp", "laser"]
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}"
          + (f"  {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    out = [run(k) for k in want if k in DOMAINS]
    p = os.path.join(BASE, "_rep8_summary.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote", p)
