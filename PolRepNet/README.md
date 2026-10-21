# PolRepNet

Reference implementation of **PolRepNet**, the polarization representation network introduced in

> Z. Shang, J. Han, X. Hao, et al., *Dual-mode passive-active polarimetric imaging with a unified
> blind restoration method*, **Optics Letters** (2026), under review.

PolRepNet restores polarization images from the four directional intensity measurements
($I_0, I_{45}, I_{90}, I_{135}$). It jointly recovers the total intensity $S_0$ and the four
denoised directions, from which the degree of linear polarization (DoLP) and the angle of
polarization (AoP) are obtained through the physical Stokes formulas. The same network serves
both the passive low-light (LLCP) and the active laser (LASER) regime, and a single model
trained on the mixed data replaces the two domain-specific models.

**What is in this repository.** The model, the training/evaluation pipeline, the offline
preprocessing, the trained weights used in the paper, and the single-branch ablation.

## Highlights

- **Multi-branch reparameterized convolution (MBRConv).** Every convolution is built from
  parallel $1\times1$, $3\times3$, $5\times5$, $1\times3$ and $3\times1$ branches plus
  BatchNorm. Training uses all branches (`rep_scale` scales their width); `slim()` fuses them
  into one equivalent plain convolution, so the deployed network is independent of `rep_scale`.
- **Dual-branch decoupled U-Net.** An $S_0$ branch and a polarimetric branch share the same
  three-stage encoder–decoder topology but are optimized separately, and both predict residuals.
  DoLP/AoP are computed analytically, not learned.
- **Intensity Adaptive Preprocessing (IAP).** An offline, per-sample luminance alignment with a
  single gain shared by the four directions, so DoLP/AoP are preserved exactly. No trainable
  parameter, identical in training and inference.

| Model | Branches | `rep_scale` | Training params | Fused inference params |
|---|---|---|---|---|
| PolRepNet-1 (single-branch ablation) | 1 | 4 | 2.3675 M | 0.2584 M |
| **PolRepNet-2 (dual-branch, used in the paper)** | 2 | **8** | **5.5294 M** | **0.2950 M** |
| PolRepNet-2, `rep_scale=4` reference | 2 | 4 | 2.7655 M | 0.2950 M |

## Install

```bash
conda create -n polrepnet python=3.12 -y
conda activate polrepnet
pip install -r requirements.txt
# then install the CUDA build of PyTorch that matches your driver, e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Tested with Python 3.12 and PyTorch 2.7 (CUDA 12.8).

## Data

Two domains are used. The passive low-light data are the public LLCP dataset (converted to
grayscale); the active laser data are collected with a 520 nm linearly polarized laser and are
available from the authors on request. Expected layout (four PNG/BMP files per sample):

```text
llcp_gray/{gt,noisy}/{train,val,test}/<sample>/{0,45,90,135}.png
laser/{gt,noisy}/{train,val,test}/<sample>/{0,45,90,135}.bmp
```

Splits: LLCP 180/30/30, LASER 120/20/20, mixed 300/50/50 (train/val/test). Helper scripts
`convert_llcp_to_gray.py` and `split_merge_datasets.py` prepare the raw folders.

## Quick start

```bash
# 1) offline preprocessing (crop 1024 -> IAP -> 128-pixel patches, one .pt per sample)
python preprocess_rep8_p128.py            # -> pt_llcp_iap_p128/, pt_laser_iap_p128/
python preprocess_rep8_p128_mixed.py      # -> pt_mixed_iap_p128/

# 2) training (PolRepNet-2, rep_scale=8, patch 128)
python train_rep8.py llcp                 # passive-only model
python train_rep8.py laser                # active-only model
python train_rep8_mixed.py                # mixed-domain model

# 3) evaluation (test.py builds the rep_scale=8 network by default; _eval_rep8.py also
#    prints the reference numbers of the paper tables)
python _eval_rep8.py checkpoints/best_model_4ch_rep8_mixed.pth llcp 8
```

Training recipe: AdamW (lr 1e-4, weight decay 1e-4), cosine annealing, batch size 4, 200 epochs,
`base_ch=8`, `rep_scale=8`, IAP applied offline (`iap_enabled=False`).

A one-minute inference check (`--slim` also fuses the branches and reports the
inference-time size, i.e. 5.53 M -> 0.30 M):

```bash
python demo.py --ckpt checkpoints/best_model_4ch_rep8_mixed.pth --slim
```

## Pretrained weights

The three checkpoints (22 MB each, 63 MB in total) are distributed outside this
repository, since a single upload here is limited to 25 MB:

> **Baidu Netdisk:** https://pan.baidu.com/s/XXXXXXXXXX  access code: `XXXX`
> The archive also contains a `README.txt` with the MD5 sums.

Unzip them into `checkpoints/` at the repository root — `demo.py`, `test.py` and
`_eval_rep8.py` read them from there by default:

| File | Model | Domain |
|---|---|---|
| `checkpoints/best_model_4ch_rep8_llcp.pth` | PolRepNet-2, dual branch, `rep_scale=8` | passive (LLCP) |
| `checkpoints/best_model_4ch_rep8_laser.pth` | PolRepNet-2, dual branch, `rep_scale=8` | active (LASER) |
| `checkpoints/best_model_4ch_rep8_mixed.pth` | PolRepNet-2, dual branch, `rep_scale=8` | mixed (one model for both) |

All three store the training-time state dict (parallel branches, not yet fused):
build the model with the matching `rep_scale` / `base_ch`, load with
`strict=True`, then call `slim()`. Every number in the paper comes from these
three files.

## Repository layout

```text
polrepnet.py                model: MBRConv1/3/5, FST/FSTR, the two branches, slim()
trainer.py                  training loop, polarization loss, dataset
train_rep8.py               PolRepNet-2 training (per domain)
train_rep8_mixed.py         mixed-domain training
_eval_rep8.py               evaluation for rep_scale=8 checkpoints
test.py                     evaluation utilities and main entry (set REP_SCALE there)
preprocess_to_pt.py         base offline preprocessing (IAP + patching)
preprocess_separate_to_pt.py / preprocess_mixed_to_pt.py
                            per-domain and mixed wrappers
preprocess_rep8_p128.py / preprocess_rep8_p128_mixed.py
                            the 128-pixel patch pipeline used in the paper
convert_llcp_to_gray.py     LLCP -> grayscale conversion
split_merge_datasets.py     build the mixed-domain dataset
_model_single_branch.py     PolRepNet-1 backbone (single-branch ablation)
_train_single.py            single- vs dual-branch ablation driver
demo.py                     minimal inference example
```

All paths are relative to the repository root; edit the configuration block at the top of the
preprocessing scripts to point at your local data. `checkpoints/` is not part of this archive —
see *Pretrained weights* above.

## Related work by the authors

- Z. Shang, J. Han, X. Hao, et al., "Transformer-based despeckling for laser active polarimetric
  imaging," *Opt. Lett.* **51**, 3397–3400 (2026).
- Z. Shang, J. Han, X. Hao, et al., "One-step lightweight transformer network for laser speckle
  suppression in active polarimetric imaging," *Opt. Laser Technol.* **203**, 115944 (2026).

## License

MIT — see `LICENSE`. If you use this code in your research, please cite the paper above.
