# -*- coding: utf-8 -*-
"""单分支消融模型 PolRepNet-1：只保留 polar_branch，S0 由四方向均值反推。

双分支里 s0_branch 细化灰度强度（1ch->1ch），polar_branch 去噪四方向偏振（4ch->4ch）。
而 S0 = (I0+I45+I90+I135)/4 本身就能从 polar_branch 的 4 通道算出，因此用本模型检验
s0_branch 的贡献：pred_s0 = clamp(mean(pred_4ch))，损失、优化器与数据管线完全不动，
唯一变量是"有没有那条 1 通道的 S0 分支"。

参数量（base_ch=8, rep_scale=4, num_blocks=[4]*6）：
  双分支   训练 2,765,457 (2.7655 M)   推理 294,993 (0.2950 M)
  单分支   训练 2,367,500 (2.3675 M)   推理 258,412 (0.2584 M)   ← 去掉 s0 分支

用法:
    python _model_single_branch.py        # 自检：前向形状 + slim 一致性 + 参数量 + 损失兼容
"""
import os
import sys

import torch
import torch.nn as nn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from polrepnet import (PolRepNet, PolRepNetSlim,      # noqa: E402
                      PolarBranchMultiScale)

NUM_BLOCKS_BASE = [4, 4, 4, 4, 4, 4]               # 与双分支 polar_branch 相同
NUM_BLOCKS_WIDE = [5, 5, 5, 5, 5, 5]               # 容量对齐档


class SingleBranch(nn.Module):
    """只有 polar_branch 的单分支模型。接口与 PolRepNet 一致（返回 2 元组），
    因此 PolarizationLoss / Trainer 无需任何改动。

    forward(x) -> (s0_out, polar_out)
        polar_out = x + polar_branch(x)          # 与双分支同式，不 clamp
        s0_out    = clamp(mean(polar_out), 0, 1) # 双分支的 s0_out 同样有界
    """

    def __init__(self, in_chans_polar=4, base_ch=8, rep_scale=4,
                 num_blocks=None, use_4ch_input=True, use_12ch_input=False,
                 iap_enabled=False, iap_C=0.95, iap_safety_margin=0.8,
                 iap_eps=1e-8, iap_k=2):
        super().__init__()
        self.use_4ch_input = use_4ch_input
        self.use_12ch_input = use_12ch_input
        self.iap_enabled = iap_enabled
        self.iap_C, self.iap_safety_margin = iap_C, iap_safety_margin
        self.iap_eps, self.iap_k = iap_eps, iap_k
        self.base_ch, self.rep_scale = base_ch, rep_scale
        self.num_blocks = num_blocks or NUM_BLOCKS_BASE

        self.polar_branch = PolarBranchMultiScale(
            in_ch=in_chans_polar, out_ch=in_chans_polar,
            base_ch=base_ch, rep_scale=rep_scale,
            num_blocks=self.num_blocks,
        )

    # 与 PolRepNet 相同的 IAP（训练时 iap_enabled=False，数据已离线 IAP）
    def apply_iap_stable(self, s0_rgb, C, safety_margin):
        luminance = s0_rgb.mean(dim=1, keepdim=True)
        mean = luminance.mean(dim=(2, 3), keepdim=True)
        std = luminance.std(dim=(2, 3), keepdim=True)
        v = torch.clamp(mean + self.iap_k * std, min=self.iap_eps)
        gamma = (C * safety_margin) / v
        return torch.clamp(s0_rgb * gamma, 0.0, 1.0), gamma

    def forward(self, x):
        if self.use_4ch_input:
            B = x.shape[0]
            polar = x
            s0 = x.mean(dim=1, keepdim=True)
            if self.iap_enabled:
                s0, gamma = self.apply_iap_stable(s0, self.iap_C,
                                                  self.iap_safety_margin)
                polar = torch.clamp(polar * gamma.view(B, 1, 1, 1), 0.0, 1.0)
        elif self.use_12ch_input:
            B, C, H, W = x.shape
            data = x.view(B, 4, 3, H, W)
            if self.iap_enabled:
                s0 = data.mean(dim=1)
                s0, gamma = self.apply_iap_stable(s0, self.iap_C,
                                                  self.iap_safety_margin)
                data = torch.clamp(data * gamma.view(B, 1, 1, 1, 1), 0.0, 1.0)
            s0 = data.mean(dim=1)
            w = torch.tensor([0.299, 0.587, 0.114],
                             device=x.device).view(1, 1, 3, 1, 1)
            polar = (data * w).sum(dim=2)
        else:
            raise ValueError("单分支消融只支持 use_4ch_input / use_12ch_input")

        polar_out = polar + self.polar_branch(polar)
        s0_out = torch.clamp(polar_out.mean(dim=1, keepdim=True), 0.0, 1.0)
        return s0_out, polar_out

    def slim(self):
        """融合为推理版：复用 PolRepNet.slim() 的 BN 折叠，只取偏振段。

        注意：PolRepNet 的 polar_branch 固定 4 块/阶段，容量档要先给 proxy 换一个
        同 num_blocks 的 polar_branch，否则 5 块/阶段的权重灌不进 4 块/阶段的槽里。
        """
        proxy = PolRepNet(base_ch=self.base_ch, rep_scale=self.rep_scale,
                         in_chans_s0=1, in_chans_polar=4,
                         use_4ch_input=self.use_4ch_input,
                         use_12ch_input=self.use_12ch_input,
                         iap_enabled=self.iap_enabled)
        proxy.polar_branch = PolarBranchMultiScale(
            in_ch=4, out_ch=4, base_ch=self.base_ch, rep_scale=self.rep_scale,
            num_blocks=self.num_blocks)
        # 只灌 polar 权重；proxy 的 s0_branch 保持随机，稍后整个丢弃
        proxy.polar_branch.load_state_dict(self.polar_branch.state_dict())
        full = proxy.slim()

        net = SingleBranchSlim(
            in_chans_polar=4, base_ch=self.base_ch, num_blocks=self.num_blocks,
            use_4ch_input=self.use_4ch_input,
            use_12ch_input=self.use_12ch_input, iap_enabled=self.iap_enabled)
        sd, src = net.state_dict(), full.state_dict()
        polar_keys = [k for k in sd if k.startswith("polar_")]
        missing = [k for k in polar_keys if k not in src]
        if missing:
            raise RuntimeError(f"融合键对不上，slim() 可能已变: {missing[:5]}")
        for k in polar_keys:
            sd[k] = src[k]
        net.load_state_dict(sd, strict=True)
        return net

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_M": total / 1e6}


def _make_slim_class():
    """构造 SingleBranchSlim：继承 PolRepNetSlim，构造后删掉 s0_* 子模块，
    forward 只跑偏振路径，键名与 PolRepNetSlim 的 polar_ 段完全一致。"""

    class _Slim(PolRepNetSlim):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            for name in [k for k in list(self._modules) if k.startswith("s0_")]:
                delattr(self, name)

        def forward(self, x):
            # ---- 输入切分（同 PolRepNetSlim）----
            if self.use_4ch_input:
                B = x.shape[0]
                polar = x
                s0 = x.mean(dim=1, keepdim=True)
                if self.iap_enabled:
                    s0, gamma = self.apply_iap_stable(s0, self.iap_C,
                                                      self.iap_safety_margin)
                    polar = torch.clamp(polar * gamma.view(B, 1, 1, 1), 0.0, 1.0)
            elif self.use_12ch_input:
                B, C, H, W = x.shape
                data = x.view(B, 4, 3, H, W)
                if self.iap_enabled:
                    s0 = data.mean(dim=1)
                    s0, gamma = self.apply_iap_stable(s0, self.iap_C,
                                                      self.iap_safety_margin)
                    data = torch.clamp(data * gamma.view(B, 1, 1, 1, 1), 0.0, 1.0)
                s0 = data.mean(dim=1)
                w = torch.tensor([0.299, 0.587, 0.114],
                                 device=x.device).view(1, 1, 3, 1, 1)
                polar = (data * w).sum(dim=2)
            else:
                raise ValueError("单分支消融只支持 4ch/12ch")

            # ---- 偏振路径（同 PolRepNetSlim）----
            e0 = polar
            for block in self.polar_enc0:
                e0 = block(e0)
            e1 = self.polar_enc1(e0)
            for block in self.polar_enc1_blocks:
                e1 = block(e1)
            e2 = self.polar_enc2(e1)
            for block in self.polar_enc2_blocks:
                e2 = block(e2)
            b = e2
            for block in self.polar_bottleneck_blocks:
                b = block(b)
            d2 = self.polar_dec2(b)
            e1_att = e1 * self.polar_skip_att2(e1)
            d2 = self.polar_skip2(torch.cat([d2, e1_att], dim=1))
            for block in self.polar_dec2_blocks:
                d2 = block(d2)
            d1 = self.polar_dec1(d2)
            e0_att = e0 * self.polar_skip_att1(e0)
            d1 = self.polar_skip1(torch.cat([d1, e0_att], dim=1))
            for block in self.polar_dec1_blocks:
                d1 = block(d1)
            polar_out = polar + self.polar_tail(d1)

            s0_out = torch.clamp(polar_out.mean(dim=1, keepdim=True), 0.0, 1.0)
            return s0_out, polar_out

    return _Slim


# SingleBranch.slim() 里引用的名字，延迟构造以避免类定义顺序问题
SingleBranchSlim = _make_slim_class()


def build(base_ch=8, rep_scale=4, num_blocks=None, **kw):
    return SingleBranch(base_ch=base_ch, rep_scale=rep_scale,
                        num_blocks=num_blocks, **kw)


def param_report():
    for tag, nb in (("single  ", NUM_BLOCKS_BASE), ("single_w", NUM_BLOCKS_WIDE)):
        m = build(num_blocks=nb, iap_enabled=False)
        n_tr = sum(p.numel() for p in m.parameters())
        n_inf = sum(p.numel() for p in m.slim().parameters())
        print(f"[{tag}] num_blocks={nb}  训练 {n_tr:,} ({n_tr/1e6:.4f} M)  "
              f"融合后推理 {n_inf:,} ({n_inf/1e6:.4f} M)")
    dual = PolRepNet(base_ch=8, rep_scale=4, in_chans_s0=1, in_chans_polar=4,
                    use_4ch_input=True, use_12ch_input=False, iap_enabled=False)
    n_tr = sum(p.numel() for p in dual.parameters())
    n_inf = sum(p.numel() for p in dual.slim().parameters())
    print(f"[dual    ] 双分支 PolRepNet-2  训练 {n_tr:,} ({n_tr/1e6:.4f} M)  "
          f"融合后推理 {n_inf:,} ({n_inf/1e6:.4f} M)")


def selfcheck():
    torch.manual_seed(0)

    print("== [1] 前向形状 ==")
    m = build(num_blocks=NUM_BLOCKS_BASE, iap_enabled=False).eval()
    x = torch.rand(2, 4, 64, 64)
    with torch.no_grad():
        s0, pol = m(x)
    print(f"   输入 {tuple(x.shape)} -> s0 {tuple(s0.shape)}  polar {tuple(pol.shape)}")
    assert s0.shape == (2, 1, 64, 64) and pol.shape == (2, 4, 64, 64)
    # s0 是有意 clamp 过的（双分支的 s0_out 同样有界），polar_out 不 clamp
    ref = pol.mean(1, keepdim=True).clamp(0.0, 1.0)
    d = (s0 - ref).abs().max().item()
    print(f"   s0 == clamp(mean(polar)) ? max|差| = {d:.3e}  {'OK' if d < 1e-6 else '!!'}")
    raw = pol.mean(1, keepdim=True)
    print(f"   （未 clamp 的 mean(polar) 越界幅度 {(raw - ref).abs().max().item():.3f}，"
          f"随机权重下正常）")

    print("\n== [2] 训练态 vs slim 融合态 前向一致性（打破初始化，随机权重） ==")
    m2 = build(num_blocks=NUM_BLOCKS_BASE, iap_enabled=False)
    with torch.no_grad():
        for p in m2.parameters():
            p.normal_(0, 0.05)
    m2.eval()
    slim = m2.slim().eval()
    with torch.no_grad():
        a1, a2 = m2(x)
        b1, b2 = slim(x)
    for nm, a, b in (("s0", a1, b1), ("polar", a2, b2)):
        dm = (a - b).abs().max().item()
        rel = dm / (a.abs().max().item() + 1e-12)
        print(f"   {nm:6s} max|train-slim| = {dm:.3e}  (相对 {rel:.2e})"
              f"  {'OK' if dm < 1e-4 else '!! 不一致'}")

    print("\n== [3] 参数量 ==")
    param_report()

    print("\n== [4] 与 PolarizationLoss 的兼容性 ==")
    from trainer import PolarizationLoss
    crit = PolarizationLoss()
    tgt = torch.rand(2, 4, 64, 64)
    with torch.no_grad():
        loss, d_ = crit(m2(x), tgt)
    print(f"   loss 可算: total={loss.item():.4f}  " +
          " ".join(f"{k}={v:.4f}" for k, v in d_.items()))
    print(f"   s0_loss 用 mean(pred_4ch) 对 mean(target_4ch)，单分支下仍然自洽")

    print("\n== [5] slim 后的子模块（应只剩 polar_*） ==")
    ks = sorted(set(k.split('.')[0] for k in slim.state_dict()))
    print("   ", ks)

    print("\n== [6] 键名与 PolRepNetSlim 的 polar 段一致（评测脚本可直接用） ==")
    full = PolRepNetSlim(in_chans_polar=4, base_ch=8, num_blocks=NUM_BLOCKS_BASE,
                        use_4ch_input=True, use_12ch_input=False,
                        iap_enabled=False)
    fk = set(k for k in full.state_dict() if k.startswith("polar_"))
    sk = set(slim.state_dict().keys())
    print(f"   完整 slim 有 polar 键 {len(fk)} 个；单分支 slim 共 {len(sk)} 个；"
          f"缺失 {len(fk - sk)} 个  {'OK' if not (fk - sk) else '!!'}")


if __name__ == "__main__":
    selfcheck()
