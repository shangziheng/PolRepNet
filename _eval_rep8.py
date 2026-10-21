# -*- coding: utf-8 -*-
"""
评测 PolRepNet-2 的 rep_scale=8 权重，并与论文表中的数值并列对照。

评测口径与 test.py 一致：全图 -> 中心裁 1024 -> 对 noisy 与 clean 同时施加外部 IAP
-> 逐图算 S0/DoLP/AoP 的 PSNR+SSIM -> 取平均（1 图 1 组指标）。

权重是训练形态（多分支未融合）的 state_dict：必须按训练时的 rep_scale / base_ch
建模型载入（strict），再 .slim() 转推理融合版。用错 rep_scale 会直接 shape 不匹配报错。

用法:
    python _eval_rep8.py                          # 默认 LLCP rep_scale=8
    python _eval_rep8.py <ckpt> <domain> <rep_scale>
"""
import os
import sys
import json
import time

import numpy as np
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from test import (load_4ch_sample, apply_iap_4ch_batch,      # noqa: E402
                  compute_polarization_from_4ch, calculate_metrics)
from polrepnet import PolRepNet                                 # noqa: E402

DOMAINS = {
    "llcp": dict(tag="LLCP", n=30,
                 gt=os.path.join(BASE, "llcp_gray", "gt", "test"),
                 noisy=os.path.join(BASE, "llcp_gray", "noisy", "test")),
    "laser": dict(tag="LASER", n=20,
                  gt=os.path.join(BASE, "laser", "gt", "test"),
                  noisy=os.path.join(BASE, "laser", "noisy", "test")),
}

# 论文表 1 / 表 2 的数值，用于并列对照
PAPER = {
    "llcp": {
        "noisy":  dict(s0_psnr=25.83, s0_ssim=0.696, dolp_psnr=12.60, dolp_ssim=0.134,
                       aop_psnr=13.96, aop_ssim=0.039),
        "base4":  dict(s0_psnr=33.41, s0_ssim=0.943, dolp_psnr=31.06, dolp_ssim=0.803,
                       aop_psnr=19.54, aop_ssim=0.247),
        "best":   dict(name="Restormer", s0_psnr=34.12, s0_ssim=0.947, dolp_psnr=31.91,
                       dolp_ssim=0.820, aop_psnr=19.65, aop_ssim=0.264),
    },
    "laser": {
        "noisy":  dict(s0_psnr=24.66, s0_ssim=0.601, dolp_psnr=19.03, dolp_ssim=0.406,
                       aop_psnr=17.76, aop_ssim=0.156),
        "base4":  dict(s0_psnr=26.15, s0_ssim=0.877, dolp_psnr=24.18, dolp_ssim=0.595,
                       aop_psnr=23.54, aop_ssim=0.441),
        "best":   dict(name="Restormer", s0_psnr=26.28, s0_ssim=0.885, dolp_psnr=24.49,
                       dolp_ssim=0.600, aop_psnr=23.33, aop_ssim=0.443),
    },
}

KEYS = ["s0_psnr", "s0_ssim", "dolp_psnr", "dolp_ssim", "aop_psnr", "aop_ssim"]


def build(base_ch, rep_scale, iap_enabled=True):
    # 本脚本已对输入显式做外部 IAP，模型内再开一遍就是"两遍 IAP"，与 test.py 走 .pt
    # 时的"一遍"口径不同（差约 0.5 dB）；第 4 个命令行参数可关掉内置 IAP 做定标。
    return PolRepNet(base_ch=base_ch, rep_scale=rep_scale,
                    in_chans_s0=1, in_chans_polar=4,
                    use_4ch_input=True, use_12ch_input=False,
                    iap_enabled=iap_enabled)


def infer_polrepnet(m, x):
    """前向并取出 S0 / DoLP / AoP。"""
    b1, b2 = m(x)
    if b1.dim() == 4 and b1.shape[1] > 1:
        b1 = b1.mean(dim=1, keepdim=True)
    _, d, a = compute_polarization_from_4ch(b2)
    return b1, d, a


def pol_from_4ch(t):
    return t.mean(dim=1, keepdim=True), *compute_polarization_from_4ch(t)[1:]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        BASE, "best_model_4ch_rep8_llcp.pth")
    dom = sys.argv[2] if len(sys.argv) > 2 else "llcp"
    rep_scale = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    # 第 4 个参数：模型内 IAP 开关，默认 1；传 0 则输入只经外部 IAP 一遍。
    iap_enabled = (sys.argv[4].lower() not in ("0", "false", "no")
                   if len(sys.argv) > 4 else True)
    # 评测设备：默认 GPU，可设 EVAL_DEVICE=cpu
    want = os.environ.get("EVAL_DEVICE", "").strip().lower()
    if not os.path.exists(ckpt):
        raise SystemExit(f"找不到权重: {ckpt}")
    d = DOMAINS[dom]
    if want == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"ckpt   = {os.path.basename(ckpt)}  ({os.path.getsize(ckpt) / 1e6:.2f} MB)")
    print(f"域     = {d['tag']} (n={d['n']})   rep_scale={rep_scale}  device={device}  "
          f"模型内 IAP={'开(外部+内置=两遍)' if iap_enabled else '关(仅外部=一遍)'}")

    m = build(8, rep_scale, iap_enabled).to(device).eval()
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    if all(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    n_tr = sum(p.numel() for p in m.parameters())
    m = m.slim().to(device).eval()
    n_inf = sum(p.numel() for p in m.parameters())
    print(f"参数量 = {n_tr:,} ({n_tr/1e6:.3f} M) 训练态 -> {n_inf:,} "
          f"({n_inf/1e6:.4f} M) 融合后\n")

    items = sorted(f for f in os.listdir(d["gt"])
                   if os.path.isdir(os.path.join(d["gt"], f)))
    recs, t0 = [], time.time()
    with torch.no_grad():
        for it in items:
            noisy = load_4ch_sample(os.path.join(d["noisy"], it), 1024,
                                    center_crop=True).to(device)
            clean = load_4ch_sample(os.path.join(d["gt"], it), 1024,
                                    center_crop=True).to(device)
            noisy_iap, _ = apply_iap_4ch_batch(noisy)
            clean_iap, _ = apply_iap_4ch_batch(clean)
            p_s0, p_dl, p_ap = infer_polrepnet(m, noisy_iap)
            c_s0, c_dl, c_ap = pol_from_4ch(clean_iap)
            n_s0, n_dl, n_ap = pol_from_4ch(noisy_iap)
            recs.append(calculate_metrics(p_s0, p_dl, p_ap, c_s0, c_dl, c_ap,
                                          n_s0, n_dl, n_ap, lpips_model=None))
    agg = {k: float(np.mean([r[k] for r in recs])) for k in KEYS}
    print(f"评测完成：{len(items)} 张图，用时 {time.time()-t0:.1f}s\n")

    ref = PAPER[dom]
    print(f"{'':22s} {'S0-PSNR':>8s} {'S0-SSIM':>8s} {'DoLP-PSNR':>10s} "
          f"{'DoLP-SSIM':>10s} {'AoP-PSNR':>9s} {'AoP-SSIM':>9s}")
    for tag, label, col in (("noisy", "含噪输入（参考）", None),
                            ("base4", "PolRepNet-2 (rep_scale=4)", "\033[2m"),
                            ("best", ref["best"]["name"], "\033[2m")):
        r = ref[tag]
        line = (f"{label:22s} {r['s0_psnr']:8.2f} {r['s0_ssim']:8.3f} "
                f"{r['dolp_psnr']:10.2f} {r['dolp_ssim']:10.3f} "
                f"{r['aop_psnr']:9.2f} {r['aop_ssim']:9.3f}")
        print(("\033[2m" + line + "\033[0m") if col else line)
    line = (f"{'rep_scale=8 (本次)':22s} {agg['s0_psnr']:8.2f} {agg['s0_ssim']:8.3f} "
            f"{agg['dolp_psnr']:10.2f} {agg['dolp_ssim']:10.3f} "
            f"{agg['aop_psnr']:9.2f} {agg['aop_ssim']:9.3f}")
    print("\033[1m" + line + "\033[0m")

    b = ref["base4"]
    print(f"\n相对 rep_scale=4 的差值（正=更好；基线是 patch256，本次是 patch128，"
          f"非单一变量）：")
    for k in KEYS:
        print(f"  {k:10s} {agg[k] - b[k]:+7.3f}")
    w = ref["best"]
    print(f"相对 {w['name']}（逐列最优）的差值：")
    for k in KEYS:
        print(f"  {k:10s} {agg[k] - w[k]:+7.3f}")

    # ---- 产物命名：带上权重标识，避免跨权重评测时互相覆盖 ----
    tag = "" if iap_enabled else "_noinit_iap"
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    out = os.path.join(BASE, f"_eval_rep{rep_scale}_{dom}__{stem}{tag}.json")
    json.dump({"ckpt": ckpt, "domain": dom, "rep_scale": rep_scale,
               "iap_enabled": iap_enabled,
               "n": len(items), "average_metrics": agg,
               "paper_base4": b, "paper_best": w},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote", out)

    legacy = os.path.join(BASE, f"_eval_rep{rep_scale}_{dom}{tag}.json")
    # 短名只在该域自训练的权重下写出：stem 必须正好是
    # best_model_4ch_rep{rep_scale}_{dom}。
    if stem == f"best_model_4ch_rep{rep_scale}_{dom}":
        json.dump({"ckpt": ckpt, "domain": dom, "rep_scale": rep_scale,
                   "iap_enabled": iap_enabled,
                   "n": len(items), "average_metrics": agg,
                   "paper_base4": b, "paper_best": w},
                  open(legacy, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1)
        print("wrote", legacy)
    else:
        print(f"[!] 未写 {os.path.basename(legacy)}：这是跨域评测，写短名会覆盖"
              f"该域自训练的结果。")


if __name__ == "__main__":
    main()
