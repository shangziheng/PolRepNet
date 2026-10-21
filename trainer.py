
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import kornia
from polrepnet import PolRepNet
import os
import math
import random
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
from contextlib import nullcontext
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))


class BranchDataset(Dataset):
    """单分支偏振去噪数据集

    channels='4ch':  输出4通道 (4方向灰度 I0,I45,I90,I135) —— 当前数据集
    channels='12ch': 输出12通道 (4方向 × RGB)
    """

    def __init__(self, clean_dir, noisy_dir, mode='train', image_size=1024, crop_size=256, stride=128,
                 normalize_brightness=False, target_brightness=0.1, use_pt=True, preload=True,
                 channels='4ch'):
        """
        Args:
            clean_dir: 清晰图像目录（.pt 文件目录或原始图片目录）
            noisy_dir: 噪声图像目录
            mode: 'train', 'val', 'test'
            image_size: 输入图像大小
            crop_size: 裁剪大小
            stride: 裁剪步长
            normalize_brightness: 是否对输入进行亮度归一化
            target_brightness: 归一化的目标平均亮度
            use_pt: 是否使用 .pt 预处理文件（推荐True）
            preload: 是否把 .pt 一次性预加载进内存（训练时零磁盘IO，推荐True）
            channels: '4ch' 或 '12ch'，决定输出通道数
        """
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.mode = mode
        self.image_size = image_size
        self.crop_size = crop_size
        self.stride = stride
        self.eps = 1e-8
        self.normalize_brightness = normalize_brightness
        self.target_brightness = target_brightness
        self.use_pt = use_pt
        self.preload = preload
        self.channels = channels
        self.channels_per_dir = 1 if channels == '4ch' else 3
        # .pt 相关缓存
        self._pt_is_patch = False          # True: 预裁切补丁格式
        self._pt_cache = OrderedDict()     # 未预加载时的 LRU 缓存（按文件路径）
        self._pt_cache_max = 8
        self._preloaded_clean = None
        self._preloaded_noisy = None

        # RGB转灰度权重 (R, G, B顺序，数据是RGB格式)
        self.rgb_to_gray_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

        self.do_crop = (mode in ['train', 'val', 'test'])  # test模式也裁剪
        self.center_crop_size = 1024

        # 检测数据格式
        if self.use_pt:
            self.sample_files = self._find_pt_files()
            print(f"[SingleBranch] 使用 .pt 文件，找到 {len(self.sample_files)} 个样本")
        else:
            self.sample_folders = self._find_valid_sample_folders()
            print(f"[SingleBranch] 使用原始图片，找到 {len(self.sample_folders)} 个样本")

        if self.normalize_brightness:
            print(f"[SingleBranch] 启用输入亮度归一化（仅S0彩色通道），目标亮度 = {self.target_brightness}")

        # 计算裁剪相关参数
        if self.do_crop:
            self.num_crops_per_image = self._calculate_num_crops()
            self.crop_positions = self._precompute_crop_positions()
        else:
            self.num_crops_per_image = 1
            self.crop_positions = None

        # 预加载 .pt 到内存（训练时零磁盘IO）
        if self.use_pt and self.preload:
            self._preload_all_pt()

        print(f"初始化完成，数据集大小: {len(self)}")
        if self.channels_per_dir == 1:
            print(f"[INFO] 输出格式: 4通道 (4方向灰度偏振 I0,I45,I90,I135)")
        else:
            print(f"[INFO] 输出格式: 12通道 (4方向 × RGB)")

    def _find_pt_files(self):
        """查找所有 .pt 文件"""
        if not os.path.exists(self.clean_dir):
            raise FileNotFoundError(f"clean_dir 不存在: {self.clean_dir}")
        pt_files = sorted([f for f in os.listdir(self.clean_dir) if f.endswith('.pt')])
        return [os.path.join(self.clean_dir, f) for f in pt_files]

    def _find_valid_sample_folders(self):
        """找到所有有效的样本文件夹（兼容模式）"""
        sample_folders = []
        if not os.path.exists(self.clean_dir):
            raise FileNotFoundError(f"clean_dir 不存在: {self.clean_dir}")
        if not os.path.exists(self.noisy_dir):
            raise FileNotFoundError(f"noisy_dir 不存在: {self.noisy_dir}")

        folders = sorted(os.listdir(self.clean_dir))
        folders = [f for f in folders if os.path.isdir(os.path.join(self.clean_dir, f))]

        for folder in folders:
            clean_folder = os.path.join(self.clean_dir, folder)
            noisy_folder = os.path.join(self.noisy_dir, folder)

            clean_files = [
                os.path.join(clean_folder, "0.bmp"),
                os.path.join(clean_folder, "45.bmp"),
                os.path.join(clean_folder, "90.bmp"),
                os.path.join(clean_folder, "135.bmp")
            ]
            noisy_files = [
                os.path.join(noisy_folder, "0.bmp"),
                os.path.join(noisy_folder, "45.bmp"),
                os.path.join(noisy_folder, "90.bmp"),
                os.path.join(noisy_folder, "135.bmp")
            ]

            if all(os.path.exists(f) for f in clean_files + noisy_files):
                sample_folders.append({
                    'folder': folder,
                    'clean_files': clean_files,
                    'noisy_files': noisy_files
                })
        return sample_folders


    def _load_single_image(self, file_path):
        """加载并预处理单张图像（兼容模式，4ch=灰度 / 12ch=彩色）"""
        if self.channels_per_dir == 1:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"无法读取图像: {file_path}")
            h, w = img.shape
            # 中心裁剪
            start_y = (h - self.center_crop_size) // 2
            start_x = (w - self.center_crop_size) // 2
            img = img[start_y:start_y + self.center_crop_size,
                      start_x:start_x + self.center_crop_size]
            # 调整大小
            if img.shape[0] != self.image_size:
                img = cv2.resize(img, (self.image_size, self.image_size),
                                 interpolation=cv2.INTER_CUBIC)
            return img.astype(np.float32)[None, :, :] / 255.0  # [1,H,W]
        else:
            img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"无法读取图像: {file_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, c = img_rgb.shape

            # 中心裁剪
            start_y = (h - self.center_crop_size) // 2
            start_x = (w - self.center_crop_size) // 2
            img_rgb = img_rgb[start_y:start_y + self.center_crop_size,
                              start_x:start_x + self.center_crop_size, :]

            # 调整大小
            if img_rgb.shape[0] != self.image_size:
                img_rgb = cv2.resize(img_rgb, (self.image_size, self.image_size),
                                     interpolation=cv2.INTER_CUBIC)

            tensor = np.transpose(img_rgb.astype(np.float32) / 255.0, (2, 0, 1))
            return tensor

    def _preload_all_pt(self):
        """把所有 .pt 文件一次性预加载进内存（训练时零磁盘IO）

        支持的格式:
          - 'patch4ch':  uint8 [N,4,crop,crop] 预裁切补丁（preprocess_to_pt.py 生成，当前）
          - 'patch12ch': uint8 [N,12,crop,crop] 预裁切补丁（legacy 彩色格式）
          - '12ch':      float32 [12,H,W] 整幅张量（legacy 格式，仍实时裁切）
        """
        print(f"[Preload] 预加载 {len(self.sample_files)} 个 .pt 文件到内存...")
        clean_list, noisy_list = [], []
        for f in self.sample_files:
            data = torch.load(f, weights_only=True)
            data_format = data.get('format', '12ch')
            if data_format in ('patch4ch', 'patch12ch'):
                self._pt_is_patch = True
                if data_format == 'patch4ch':
                    self.channels_per_dir = 1
                else:
                    self.channels_per_dir = 3
                clean_list.append(data['clean'])   # 保持 uint8，降低内存占用
                noisy_list.append(data['noisy'])
            elif data_format == '7ch':
                raise ValueError(
                    f"数据格式为 '7ch'，但数据集要求输出4或12通道。请使用4/12通道格式的 .pt 文件，"
                    f"或重新生成数据。"
                )
            else:  # '12ch'
                clean_list.append(data['clean'])
                noisy_list.append(data['noisy'])

        self._preloaded_clean = torch.stack(clean_list)
        self._preloaded_noisy = torch.stack(noisy_list)

        if self._pt_is_patch:
            # 以文件实际补丁数为准
            self.num_crops_per_image = self._preloaded_clean.shape[1]
            mem_mb = (self._preloaded_clean.numel() + self._preloaded_noisy.numel()) \
                     * self._preloaded_clean.element_size() / 1e6
            print(f"[Preload] 补丁数/样本 = {self.num_crops_per_image}, "
                  f"内存占用 ≈ {mem_mb:.0f} MB")
        print("[Preload] 完成")

    def _load_sample_from_pt(self, idx):
        """从 .pt 文件加载样本（支持预裁切补丁与整幅张量两种格式）"""
        if self._preloaded_clean is not None:
            return self._preloaded_clean[idx], self._preloaded_noisy[idx]

        path = self.sample_files[idx]
        if path in self._pt_cache:
            return self._pt_cache[path]

        data = torch.load(path, weights_only=True)
        data_format = data.get('format', '12ch')  # 默认为12通道
        if data_format in ('patch4ch', 'patch12ch'):
            self._pt_is_patch = True
            self.channels_per_dir = 1 if data_format == 'patch4ch' else 3
            clean = data['clean']   # uint8 [N,C,crop,crop]，取块时转 float
            noisy = data['noisy']
        elif data_format == '7ch':
            # 无法从7通道恢复，抛出错误提示
            raise ValueError(
                f"数据格式为 '7ch'，但数据集要求输出4或12通道。请使用4/12通道格式的 .pt 文件，"
                f"或重新生成数据。"
            )
        else:
            clean, noisy = data['clean'], data['noisy']  # [12,H,W]

        self._pt_cache[path] = (clean, noisy)
        if len(self._pt_cache) > self._pt_cache_max:
            self._pt_cache.popitem(last=False)
        return clean, noisy

    def _load_sample_from_images(self, idx):
        """从原始图片加载样本，输出12通道（4张RGB拼接）"""
        sample = self.sample_folders[idx]
        clean_tensors = [self._load_single_image(f) for f in sample['clean_files']]
        noisy_tensors = [self._load_single_image(f) for f in sample['noisy_files']]
        clean_12ch = np.concatenate(clean_tensors, axis=0)  # [12, H, W]
        noisy_12ch = np.concatenate(noisy_tensors, axis=0)  # [12, H, W]
        # 直接返回12通道，不再转换为7通道
        return torch.from_numpy(clean_12ch), torch.from_numpy(noisy_12ch)

    def _calculate_num_crops(self):
        """计算每张图像的裁剪块数量"""
        num_x = math.floor((self.image_size - self.crop_size) / self.stride) + 1
        num_y = math.floor((self.image_size - self.crop_size) / self.stride) + 1
        return num_x * num_y

    def _precompute_crop_positions(self):
        """预计算所有裁剪位置"""
        crop_positions = []
        num_x = math.floor((self.image_size - self.crop_size) / self.stride) + 1
        for i in range(num_x):
            for j in range(num_x):
                x = j * self.stride
                y = i * self.stride
                x = min(x, self.image_size - self.crop_size)
                y = min(y, self.image_size - self.crop_size)
                crop_positions.append((x, y))
        return crop_positions

    def apply_aop_rotation(self, tensor, shift):
        """AOP旋转增强：滚动方向维
        channels_per_dir=1 (4通道): 直接滚动通道维
        channels_per_dir=3 (12通道): 每方向3通道成组滚动
        """
        if shift == 0:
            return tensor
        if self.channels_per_dir == 1:
            # 4通道灰度偏振：通道维即方向维
            if tensor.dim() == 3:
                return tensor.roll(shifts=shift, dims=0)
            elif tensor.dim() == 4:
                return tensor.roll(shifts=shift, dims=1)
            else:
                raise ValueError("只支持 3D 或 4D 张量")
        # 12通道：每方向3通道成组滚动
        shape = tensor.shape
        if tensor.dim() == 3:
            t = tensor.view(4, 3, *shape[1:])
            t = t.roll(shifts=shift, dims=0)
            return t.view(shape)
        elif tensor.dim() == 4:
            t = tensor.view(tensor.size(0), 4, 3, *shape[2:])
            t = t.roll(shifts=shift, dims=1)
            return t.view(shape)
        else:
            raise ValueError("只支持 3D 或 4D 张量")

    def __len__(self):
        if self.use_pt:
            num_samples = len(self.sample_files)
        else:
            num_samples = len(self.sample_folders)

        if self.do_crop:
            return num_samples * self.num_crops_per_image
        else:
            return num_samples

    def __getitem__(self, idx):
        if self.do_crop:
            # 确定样本和裁剪位置
            if self.use_pt:
                num_samples = len(self.sample_files)
            else:
                num_samples = len(self.sample_folders)

            sample_idx = idx // self.num_crops_per_image
            crop_idx = idx % self.num_crops_per_image

            # 加载 (12通道)
            if self.use_pt:
                clean, noisy = self._load_sample_from_pt(sample_idx)
                if self._pt_is_patch:
                    # 预裁切补丁格式：直接取第 crop_idx 个补丁并转 float（uint8 -> [0,1]）
                    clean_crop = clean[crop_idx].float() / 255.0
                    noisy_crop = noisy[crop_idx].float() / 255.0
                else:
                    x, y = self.crop_positions[crop_idx]
                    clean_crop = clean[:, y:y + self.crop_size, x:x + self.crop_size].clone()
                    noisy_crop = noisy[:, y:y + self.crop_size, x:x + self.crop_size].clone()
            else:
                x, y = self.crop_positions[crop_idx]
                clean, noisy = self._load_sample_from_images(sample_idx)
                clean_crop = clean[:, y:y + self.crop_size, x:x + self.crop_size].clone()
                noisy_crop = noisy[:, y:y + self.crop_size, x:x + self.crop_size].clone()

            # 亮度归一化
            if self.normalize_brightness:
                s0_channels = noisy_crop[:3, :, :]
                current_mean = s0_channels.mean()
                if current_mean > self.eps:
                    scale = self.target_brightness / current_mean
                    noisy_crop[:3, :, :] = torch.clamp(s0_channels * scale, 0.0, 1.0)

            # 训练增强（AOP 旋转）
            if self.mode == 'train':
                shift = random.randint(0, 3)
                clean_crop = self.apply_aop_rotation(clean_crop, shift)
                noisy_crop = self.apply_aop_rotation(noisy_crop, shift)

            return noisy_crop, clean_crop

        else:
            # 无裁剪的测试/全图模式
            if self.use_pt:
                if self._pt_is_patch:
                    # 预裁切补丁无法还原为整幅图，全图模式请使用原始图片或 '12ch' 格式 .pt
                    raise ValueError(
                        "当前 .pt 为预裁切补丁格式（patch12ch），无法用于全图/无裁剪模式。"
                        "请使用 use_pt=False（原始图片）或 '12ch' 格式的 .pt。"
                    )
                clean, noisy = self._load_sample_from_pt(idx)
            else:
                clean, noisy = self._load_sample_from_images(idx)

            if self.normalize_brightness:
                s0_channels = noisy[:3, :, :]
                current_mean = s0_channels.mean()
                if current_mean > self.eps:
                    scale = self.target_brightness / current_mean
                    noisy[:3, :, :] = torch.clamp(s0_channels * scale, 0.0, 1.0)

            return noisy, clean

class PolarizationLoss(nn.Module):
    """
    损失函数 - 适配 12 通道输入 (4方向×RGB)

    模型输出:
        - branch1_out: [B, 3, H, W]  S0彩色图像
        - branch2_out: [B, 4, H, W]  灰度偏振图像 (I0, I45, I90, I135)

    目标格式:
        - target: [B, 12, H, W] = 4方向 × RGB (I0_RGB, I45_RGB, I90_RGB, I135_RGB)

    损失:
        - L1(S0_pred, S0_target)
        - L1(DoLP_pred, DoLP_target)
        - L1(AoP_pred, AoP_target) 周期性处理
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.eps = 1e-8
        # RGB转灰度权重 (R, G, B顺序，数据是RGB格式)
        self.rgb_to_gray_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

    def _extract_s0_rgb_and_gray_polar(self, target):
        """
        从目标中提取 S0 和 灰度偏振
        - 4通道  [B,4,H,W]:   S0灰度(1ch) + 灰度偏振(4ch)
        - 12通道 [B,12,H,W]:  S0彩色(3ch) + 灰度偏振(4ch)
        """
        B, C, H, W = target.shape
        if C == 4:
            # 4通道灰度偏振 (I0,I45,I90,I135): S0 = 四方向平均 (灰度)
            s0 = target.mean(dim=1, keepdim=True)   # [B,1,H,W]
            gray_polar = target                      # [B,4,H,W]
        elif C == 12:
            # 重排为 [B, 4, 3, H, W]
            polar_rgb = target.view(B, 4, 3, H, W)
            s0 = polar_rgb.mean(dim=1)               # [B,3,H,W]
            # 灰度偏振: 每个方向的RGB转灰度
            gray_polar = (polar_rgb * self.rgb_to_gray_weights.to(target.device)).sum(dim=2)
        else:
            raise ValueError(f"不支持的目标通道数: {C}（仅支持 4 或 12）")
        return s0, gray_polar

    def _compute_polarization(self, gray_polar):
        """
        从4通道灰度偏振图像计算 DoLP 和 AoP
        Args:
            gray_polar: [B,4,H,W] (I0, I45, I90, I135)
        Returns:
            DoLP: [B,1,H,W]
            AoP:  [B,1,H,W] 范围 [0, π]
        """
        i0 = gray_polar[:, 0:1, :, :]
        i45 = gray_polar[:, 1:2, :, :]
        i90 = gray_polar[:, 2:3, :, :]
        i135 = gray_polar[:, 3:4, :, :]

        S1 = (i0 - i90) * 0.5
        S2 = (i45 - i135) * 0.5
        S0 = (i0 + i45 + i90 + i135) * 0.25

        DoLP = torch.sqrt(S1 ** 2 + S2 ** 2 + self.eps) / (S0 + self.eps)
        DoLP = torch.clamp(DoLP, 0.0, 1.0)

        AoP = 0.5 * torch.atan2(S2, S1 + self.eps)
        AoP = AoP % math.pi
        return DoLP, AoP

    def forward(self, pred, target):
        """
        Args:
            pred: tuple (branch1_out, branch2_out)
                - branch1_out: [B,1,H,W] (4ch模式) 或 [B,3,H,W] (12ch模式)  S0预测
                - branch2_out: [B,4,H,W]  灰度偏振预测
            target: [B,4,H,W] (4ch模式) 或 [B,12,H,W] (12ch模式)，也可为 tuple
        Returns:
            total_loss, loss_dict
        """
        # 解包目标（兼容 tuple 传入）
        if isinstance(target, tuple):
            _, target = target

        # 从目标提取 S0 和 灰度偏振（自动识别 4/12 通道）
        target_s0, target_gray_polar = self._extract_s0_rgb_and_gray_polar(target)

        # 解包预测
        pred_s0, pred_gray_polar = pred

        # 1. S0 L1 损失
        s0_loss = self.l1(pred_s0, target_s0)

        # 2. 计算预测和目标偏振参数
        pred_dolp, pred_aop = self._compute_polarization(pred_gray_polar)
        target_dolp, target_aop = self._compute_polarization(target_gray_polar)

        # 3. DoLP L1 损失
        dolp_loss = self.l1(pred_dolp, target_dolp)

        # 4. AoP 周期性 L1 损失
        aop_diff = torch.abs(pred_aop - target_aop)
        aop_diff = torch.min(aop_diff, math.pi - aop_diff)
        aop_loss = aop_diff.mean() / math.pi

        # 5. 强度损失
        intensity_loss = self.l1(pred_gray_polar, target_gray_polar)

        # 加权总loss
        total_loss = s0_loss +  dolp_loss +  aop_loss + 0.1 * intensity_loss

        loss_dict = {
            's0': s0_loss.item(),
            'dolp': dolp_loss.item(),
            'aop': aop_loss.item(),
            'physics': intensity_loss.item(),
            'total': total_loss.item()
        }
        return total_loss, loss_dict

class Trainer:
    def __init__(self, clean_root, noisy_root, config=None, use_amp=False,
                 pretrained_path=None, use_pt=True, normalize_brightness=False,
                 target_brightness=0.1, freeze_branch=None, channels='4ch',
                 iap_enabled=False):
        """
        Args:
            clean_root: 清晰图像根目录（包含 train/ 和 test/ 子目录）
            noisy_root: 噪声图像根目录（若 use_pt=True 可留空）
            config: 配置字典（可选）
            use_amp: 是否使用混合精度
            pretrained_path: 预训练模型路径
            use_pt: 是否使用预处理好的 .pt 文件
            normalize_brightness: 是否对输入进行亮度归一化
            target_brightness: 归一化目标亮度
            freeze_branch: 冻结分支 ('s0', 'polar', None)
            channels: '4ch' (4方向灰度偏振) 或 '12ch' (4方向×RGB)
            iap_enabled: 是否在模型内施加 IAP。若数据集已离线施加 IAP（如 data/pt_iap），
                         应设为 False（输入与target已在同一归一化亮度空间）
        """
        self.use_pt = use_pt
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.channels = channels
        self.model_path = "best_model_4ch.pth" if channels == '4ch' else "best_model_12ch.pth"
        self.freeze_branch = freeze_branch

        # 创建模型
        if channels == '4ch':
            # 4通道灰度偏振: S0分支输入1通道(灰度S0)，偏振分支输入4通道
            self.model = PolRepNet(in_chans_s0=1, in_chans_polar=4,
                                  use_4ch_input=True, use_12ch_input=False,
                                  iap_enabled=iap_enabled).to(self.device)
            print("[INFO] 使用 PolRepNet (灰度偏振双分支去噪网络，4通道输入)")
            print("[INFO] 分支1: S0灰度图像 (1通道)")
            print("[INFO] 分支2: 灰度偏振图像 (4通道)")
        else:
            self.model = PolRepNet(iap_enabled=iap_enabled).to(self.device)  # 12通道 (4方向×RGB)
            print("[INFO] 使用 PolRepNet (彩色偏振双分支去噪网络，12通道输入)")
            print("[INFO] 分支1: S0彩色图像 (3通道)")
            print("[INFO] 分支2: 灰度偏振图像 (4通道)")

        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)
        elif os.path.exists(self.model_path):
            self.load_pretrained(self.model_path)
        else:
            print("[INFO] 未找到预训练模型，从头开始训练")

        # 分支冻结
        self._apply_freeze()

        # 优化器
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)

        # 损失函数
        self.criterion = PolarizationLoss().to(self.device)

        # 混合精度
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()
            print("[OK] 启用混合精度训练 (AMP)")
        else:
            self.scaler = None
            print("[INFO] 未启用混合精度训练")

        # 数据集
        # 说明: clean_root / noisy_root 应指向包含 train/val/test 子目录的根目录
        #   use_pt=True 时指向预处理后的 .pt 目录 (如 .../data/pt)
        #   use_pt=False 时指向原始图片目录 (如 .../data/gt 与 .../data/noisy)
        self.train_dataset = BranchDataset(
            clean_dir=os.path.join(clean_root, "train"),
            noisy_dir=os.path.join(noisy_root, "train") if noisy_root else "",
            mode='train',
            image_size=1024,
            crop_size=256,
            stride=256,
            use_pt=self.use_pt,
            normalize_brightness=normalize_brightness,
            target_brightness=target_brightness,
            channels=channels
        )
        # 验证集: 优先使用 val 划分，没有则回退到 test
        val_split = "val" if os.path.isdir(os.path.join(clean_root, "val")) else "test"
        self.val_dataset = BranchDataset(
            clean_dir=os.path.join(clean_root, val_split),
            noisy_dir=os.path.join(noisy_root, val_split) if noisy_root else "",
            mode='val',
            image_size=1024,
            crop_size=256,
            stride=256,
            use_pt=self.use_pt,
            normalize_brightness=normalize_brightness,
            target_brightness=target_brightness,
            channels=channels
        )
        print(f"训练集: {len(self.train_dataset)}, 验证集({val_split}): {len(self.val_dataset)}")

    # ---------- 辅助函数 ----------
    def _extract_target_from_12ch(self, target):
        """从目标提取S0和灰度偏振（与损失函数一致），支持4/12通道"""
        B, C, H, W = target.shape
        if C == 4:
            s0 = target.mean(dim=1, keepdim=True)   # [B,1,H,W]
            gray_polar = target                      # [B,4,H,W]
        elif C == 12:
            polar_rgb = target.view(B, 4, 3, H, W)
            s0 = polar_rgb.mean(dim=1)
            weights = torch.tensor([0.299, 0.587, 0.114], device=target.device).view(1, 3, 1, 1)
            gray_polar = (polar_rgb * weights).sum(dim=2)
        else:
            raise ValueError(f"不支持的目标通道数: {C}（仅支持 4 或 12）")
        return s0, gray_polar

    def compute_polarization_parameters(self, gray_polar):
        """从4通道灰度偏振图像计算 S0(灰度), DoLP, AoP（用于验证指标）"""
        eps = 1e-8
        i0 = gray_polar[:, 0:1, :, :]
        i45 = gray_polar[:, 1:2, :, :]
        i90 = gray_polar[:, 2:3, :, :]
        i135 = gray_polar[:, 3:4, :, :]
        S0 = (i0 + i45 + i90 + i135) * 0.25
        S1 = (i0 - i90) * 0.5
        S2 = (i45 - i135) * 0.5
        DoLP = torch.sqrt(S1 ** 2 + S2 ** 2 + eps) / (S0 + eps)
        DoLP = torch.clamp(DoLP, 0.0, 1.0)
        AoP = 0.5 * torch.atan2(S2, S1 + eps)
        AoP = AoP % math.pi
        return S0, DoLP, AoP

    # ---------- 模型加载/保存 ----------
    def load_pretrained(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            if all(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[OK] 加载模型: {model_path}")
        except Exception as e:
            print(f"[FAIL] 加载失败: {e}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"[OK] 模型已保存: {path}")

    # ---------- 分支冻结 ----------
    def _apply_freeze(self):
        if self.freeze_branch is None:
            print("[INFO] 训练所有分支")
            return
        if self.freeze_branch == 's0':
            self.model.set_freeze('s0')
            print("[INFO] 冻结S0分支，只训练偏振分支")
        elif self.freeze_branch == 'polar':
            self.model.set_freeze('polar')
            print("[INFO] 冻结偏振分支，只训练S0分支")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] 可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def unfreeze_all(self):
        self.model.set_freeze(None)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        print("[INFO] 已解冻所有分支")

    def freeze_s0_branch(self):
        self.model.set_freeze('s0')
        print("[INFO] 已冻结S0分支")

    def freeze_polar_branch(self):
        self.model.set_freeze('polar')
        print("[INFO] 已冻结偏振分支")

    # ---------- 训练主循环 ----------
    def train(self, epochs, resume_epoch=0):
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # .pt 已预加载到内存，多进程反而在 Windows 下重复加载/开销更大
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        best_loss = float('inf')
        early_stop_counter = 0
        accumulation_steps = 1

        for epoch in range(resume_epoch, resume_epoch + epochs):
            self.model.train()
            train_loss = 0.0

            for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
                noisy_12ch, clean_12ch = batch_data  # [B,4,H,W] 或 [B,12,H,W]
                noisy_12ch = noisy_12ch.to(self.device)
                clean_12ch = clean_12ch.to(self.device)

                # 前向传播（AMP 时自动混合精度）
                autocast_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                                if self.use_amp else nullcontext())
                with autocast_ctx:
                    branch1_out, branch2_out = self.model(noisy_12ch)
                    output = (branch1_out, branch2_out)
                    # 计算损失
                    loss, _ = self.criterion(output, clean_12ch)
                loss = loss / accumulation_steps

                # 反向传播
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                train_loss += loss.item() * accumulation_steps

            # ---------- 验证 ----------
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                autocast_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                                if self.use_amp else nullcontext())
                with autocast_ctx:
                    for batch_data in val_loader:
                        noisy_12ch, clean_12ch = batch_data
                        noisy_12ch = noisy_12ch.to(self.device)
                        clean_12ch = clean_12ch.to(self.device)

                        branch1_out, branch2_out = self.model(noisy_12ch)
                        output = (branch1_out, branch2_out)
                        loss, _ = self.criterion(output, clean_12ch)
                        val_loss += loss.item()

            # 平均
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 简化的打印信息
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

            # 保存最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                early_stop_counter = 0
                self.save_model(self.model_path)
                print(f"  [SAVED] 最佳模型 (Val Loss: {avg_val_loss:.6f})")
            else:
                early_stop_counter += 1
                if early_stop_counter >= 100:
                    print(f"早停于 epoch {epoch+1}")
                    break

def main():

    # 预处理后的 .pt 数据根目录 (由 preprocess_to_pt.py 生成)
    # data/pt_iap: 已对 clean/noisy 各自离线施加 IAP 亮度归一化 -> 训练关闭模型内 IAP (iap_enabled=False)
    # data/pt:     未施加 IAP 的原始数据 -> 训练需开模型内 IAP (iap_enabled=True)
    clean_root = os.path.join(BASE, 'pt_laser_iap')
    noisy_root = ""  # .pt 模式下不再需要 noisy 目录

    # 创建训练器
    trainer1 = Trainer(
        clean_root, noisy_root,
        use_pt=True,       # 使用预处理 .pt 文件（训练时零解码/零裁切）
        channels='4ch',    # 4通道灰度偏振 (I0,I45,I90,I135)
        use_amp=False,      # 混合精度：完整训练 batch=4 峰值~14.5GB，16GB显存刚够
        iap_enabled=False, # 数据集已离线施加 IAP，模型内不再做 IAP
        pretrained_path=None,  # 仅偏振分支可复用，S0分支从零训练
        normalize_brightness=False,  # 可选：亮度归一化
        freeze_branch=None  # 完整训练（4通道模型结构不同，需全量训练）
    )

    # 开始训练
    trainer1.train(200)

if __name__ == "__main__":
    main()