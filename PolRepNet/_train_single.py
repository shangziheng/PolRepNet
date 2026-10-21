# -*- coding: utf-8 -*-
"""单分支消融（PolRepNet-1）与同种子双分支对照的训练脚本。

论文把双分支主干作为贡献之一：s0_branch 出 S0，polar_branch 出四方向偏振。而 S0 定义上
就是四方向均值，单分支自己也能算出，因此需要一组对照来判断这条分支到底贡献了什么。

三档（同一份数据 pt_mixed_iap、同一个 Trainer、同一个种子、200 epoch）：

  single    单分支 num_blocks=[4]*6  -> best_model_4ch_single.pth
  single_w  单分支 num_blocks=[5]*6  -> best_model_4ch_single_w.pth
            （容量对齐档：训练期 2.9117 M，比双分支 2.7655 M 还大 5.3%，
              用于排除"单分支只是参数少所以输"这个解释）
  dual      双分支（PolRepNet-2 原样，从零重训）-> best_model_4ch_dual_ctrl.pth

唯一变量是模型结构：PolarizationLoss 内部已经用 S0=四方向均值算 DoLP/AoP，
单分支只要令 pred_s0 = clamp(mean(pred_4ch))，损失、优化器与数据管线都不用改。

输出（不覆盖任何已有文件）：
    best_model_4ch_single.pth / best_model_4ch_single_w.pth / best_model_4ch_dual_ctrl.pth
    _ablate_single_<arm>.log + _ablate_single_summary.json

用法：
    python _train_single.py                # 默认 single + dual 顺序跑
    python _train_single.py single         # 只跑单分支
    python _train_single.py single_w       # 只跑容量对齐的宽单分支
"""
import os
import sys
import json
import time
import random

import numpy as np
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
from _model_single_branch import (SingleBranch, NUM_BLOCKS_BASE,   # noqa: E402
                                  NUM_BLOCKS_WIDE)

CLEAN_ROOT = os.path.join(BASE, "pt_mixed_iap")
EPOCHS = 200
SEED = 2026                # 三档共用；只固定初始化，不追求逐位可复现

ARMS = {
    "single":   dict(kind="single", nb=NUM_BLOCKS_BASE,
                     tag="单分支（与基线 polar_branch 同宽）"),
    "single_w": dict(kind="single", nb=NUM_BLOCKS_WIDE,
                     tag="单分支·容量对齐档（+5.3% 参数）"),
    "dual":     dict(kind="dual", nb=None,
                     tag="双分支·同种子从零重训（种子对照）"),
}

DEFAULT_QUEUE = ["single", "dual"]

_ORIG_MODEL = T.PolRepNet
_OrigSingle = SingleBranch


def _make_patch(nb):
    """把 SingleBranch 接到 trainer 的模型构造调用上。

    trainer 里写的是 PolRepNet(in_chans_s0=1, in_chans_polar=4, ...)，而 SingleBranch
    没有独立的 S0 输入通道，故吞掉该参数并注入 num_blocks。
    """
    def _factory(*args, **kwargs):
        kwargs.pop("in_chans_s0", None)
        kwargs["num_blocks"] = nb
        return _OrigSingle(*args, **kwargs)
    return _factory


class _Tee:
    """stdout 同时写控制台与日志文件（控制台句柄可能已失效，故两处都容错）。"""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, s):
        try:
            if sys.__stdout__ is not None:
                sys.__stdout__.write(s)
        except Exception:
            pass
        self.f.write(s)

    def flush(self):
        try:
            if sys.__stdout__ is not None:
                sys.__stdout__.flush()
        except Exception:
            pass
        self.f.flush()


class SingleTrainer(Trainer):
    """只改两件事的 Trainer：ckpt 落点，以及跳过自动加载预训练。

    model_path 须在 super().__init__ 之前生效，否则会静默地载入旧权重的同名键，
    那就不是从零训练了。
    """

    def __init__(self, *args, ckpt, **kwargs):
        self._skip_pretrain = True          # 必须在 super().__init__ 之前
        super().__init__(*args, **kwargs)
        self.model_path = ckpt

    def load_pretrained(self, model_path):
        if getattr(self, "_skip_pretrain", False):
            print(f"[ABL] 从零训练，跳过自动加载: {os.path.basename(model_path)}")
            return
        super().load_pretrained(model_path)


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(arm):
    cfg = ARMS[arm]
    ckpt = os.path.join(BASE, f"best_model_4ch_{arm}.pth"
                        if arm != "dual" else "best_model_4ch_dual_ctrl.pth")
    log = os.path.join(BASE, f"_ablate_single_{arm}.log")

    print("\n" + "=" * 78)
    print(f"[ABL-{arm}] {cfg['tag']}   kind={cfg['kind']}  "
          f"num_blocks={cfg['nb']}  seed={SEED}  epochs={EPOCHS}")
    print(f"[ABL-{arm}] 落点 {os.path.basename(ckpt)}   日志 {os.path.basename(log)}")
    print("=" * 78)

    if os.path.exists(ckpt):
        print(f"[ABL-{arm}] [SKIP] 已存在 {os.path.basename(ckpt)}，删除后可重跑")
        return

    # ---- 替换模型构造：single 档接 SingleBranch，dual 档用原样 ----
    if cfg["kind"] == "single":
        T.PolRepNet = _make_patch(cfg["nb"])
    else:
        T.PolRepNet = _ORIG_MODEL
    try:
        tee = _Tee(log)
        old = sys.stdout
        sys.stdout = tee
        t0 = time.time()
        ok, err = True, None
        try:
            _seed_all(SEED)                  # 三档同一初始化
            tr = SingleTrainer(
                CLEAN_ROOT, "",
                use_pt=True, channels="4ch",
                use_amp=False, iap_enabled=False,
                pretrained_path=None,
                freeze_branch=None,
                ckpt=ckpt,
            )
            n_tr = sum(p.numel() for p in tr.model.parameters())
            n_inf = sum(p.numel() for p in tr.model.slim().parameters())
            print(f"[ABL-{arm}] 训练期参数量 {n_tr:,} ({n_tr/1e6:.4f} M)；"
                  f"融合后推理 {n_inf:,} ({n_inf/1e6:.4f} M)")
            tr.train(EPOCHS)
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok, err = False, f"{type(e).__name__}: {e}"
        finally:
            sys.stdout = old
            tee.flush()
    finally:
        T.PolRepNet = _ORIG_MODEL        # 无论成败都还原，不影响后续档

    dt = time.time() - t0
    print(f"[ABL-{arm}] 完成={ok}  用时 {dt/60:.1f} min"
          + (f"  err={err}" if err else ""))
    p = os.path.join(BASE, "_ablate_single_summary.json")
    rec = dict(arm=arm, kind=cfg["kind"],
               num_blocks=(cfg["nb"] if cfg["nb"] else "baseline"),
               seed=SEED, epochs=EPOCHS, ckpt=os.path.basename(ckpt),
               seconds=dt, ok=ok, error=err)
    try:
        prev = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []
    except Exception:
        prev = []
    prev = [r for r in prev if r.get("arm") != arm] + [rec]
    json.dump(prev, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("updated", p)


def main():
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}"
          + (f"  {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    queue = sys.argv[1:] or DEFAULT_QUEUE
    bad = [a for a in queue if a not in ARMS]
    if bad:
        raise SystemExit(f"未知档位 {bad}；可选 {list(ARMS)}")
    print(f"[ABL] 队列 {queue}（~29 h/档）")
    for a in queue:
        run(a)


if __name__ == "__main__":
    main()
