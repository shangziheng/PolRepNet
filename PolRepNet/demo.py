"""Minimal inference demo for PolRepNet (dual-branch, rep_scale = 8).

    python demo.py --ckpt checkpoints/best_model_4ch_rep8_mixed.pth
    python demo.py --ckpt checkpoints/best_model_4ch_rep8_mixed.pth --image sample.npy --slim

The input is a grayscale four-direction stack [I0, I45, I90, I135] of shape
(4, H, W) in [0, 1].  The script prints the restored intensity and the derived
DoLP/AoP; with --slim it also fuses the parallel branches and reports the
parameters before and after reparameterization.
"""
import argparse
import os

import numpy as np
import torch

from polrepnet import PolRepNet


def build_model(base_ch=8, rep_scale=8, iap_enabled=False):
    return PolRepNet(base_ch=base_ch, rep_scale=rep_scale,
                     in_chans_s0=1, in_chans_polar=4,
                     use_4ch_input=True, use_12ch_input=False,
                     iap_enabled=iap_enabled)


def polarization_from_4ch(x):
    """S0, DoLP and AoP from the four restored directions (S0 is the mean)."""
    s0 = x.mean(dim=1, keepdim=True)
    s1 = x[:, 0:1] - x[:, 2:3]
    s2 = x[:, 1:2] - x[:, 3:4]
    dolp = torch.sqrt(s1 ** 2 + s2 ** 2) / (s0 + 1e-8)
    aop = 0.5 * torch.atan2(s2, s1)
    return s0, dolp, aop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best_model_4ch_rep8_mixed.pth")
    ap.add_argument("--image", default=None,
                    help="path to a 4-direction stack saved as a .npy array of shape "
                         "(4,H,W); if omitted, a random input is used")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--base-ch", type=int, default=8)
    ap.add_argument("--rep-scale", type=int, default=8)
    ap.add_argument("--slim", action="store_true",
                    help="fuse the branches and report the inference-time size")
    args = ap.parse_args()

    model = build_model(args.base_ch, args.rep_scale).to(args.device).eval()
    if os.path.isfile(args.ckpt):
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"loaded {args.ckpt}")
    else:
        print(f"[warning] {args.ckpt} not found — download the weights first "
              f"(see 'Pretrained weights' in README.md); running with random weights")

    if args.image:
        x = torch.from_numpy(np.load(args.image)).float()[None].to(args.device)
    else:
        x = torch.rand(1, 4, 128, 128, device=args.device)

    with torch.no_grad():
        s0_out, dirs_out = model(x)
        s0, dolp, aop = polarization_from_4ch(dirs_out)

    n_train = sum(p.numel() for p in model.parameters())
    print(f"training-time parameters  : {n_train/1e6:.4f} M")
    print(f"restored S0               : {tuple(s0_out.shape)}")
    print(f"restored directions       : {tuple(dirs_out.shape)}")
    print(f"DoLP  mean/min/max        : {dolp.mean():.4f} / {dolp.min():.4f} / {dolp.max():.4f}")
    print(f"AoP   mean (deg)          : {torch.rad2deg(aop.mean()):.2f}")

    if args.slim:
        # the check runs on CPU in float32: TF32 on GPU would add ~1e-3 noise of its own
        m_cpu, x_cpu = model.to("cpu").eval(), x.cpu()
        with torch.no_grad():
            s0_t, dirs_t = m_cpu(x_cpu)
            fused = m_cpu.slim().eval()
            s0_f, dirs_f = fused(x_cpu)
        n_inf = sum(p.numel() for p in fused.parameters())
        print(f"fused inference parameters: {n_inf/1e6:.4f} M "
              f"({100 * (1 - n_inf / n_train):.1f}% fewer)")
        print(f"max|train - fused|        : "
              f"{max(float((s0_t - s0_f).abs().max()), float((dirs_t - dirs_f).abs().max())):.2e}")


if __name__ == "__main__":
    main()
