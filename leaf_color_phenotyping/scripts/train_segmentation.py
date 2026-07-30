#!/usr/bin/env python3
"""
U-Net 叶片分割模型训练脚本。

支持的数据集格式:
    - 目录结构: images/ + masks/ (同名PNG, mask为二值图)

训练策略:
    - Encoder: EfficientNet-B3 (推荐, 精度/速度平衡) 或 ResNet-50
    - Loss: BCE + Dice Loss (混合损失, 对边界更敏感)
    - Augmentation: Albumentations 增强流水线

Usage:
    python train_segmentation.py \
        --images data/train/images/ \
        --masks data/train/masks/ \
        --backbone efficientnet-b3 \
        --epochs 100 \
        --output models/unet_cabbage.pth
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import inspect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.segmentation import IMAGENET_MEAN, IMAGENET_STD, normalize_imagenet_rgb
from src.utils import read_image_gray, read_image_rgb, split_pairs_by_sample

# 需要额外安装:
#   pip install segmentation-models-pytorch albumentations


class LeafSegmentationDataset(Dataset):
    """叶片分割数据集.

    要求:
        images/ 和 masks/ 目录下的文件名一一对应.
        mask 为单通道二值图 (0=背景, 255=叶片).
    """

    def __init__(self,
                 images_dir: str,
                 masks_dir: str,
                 image_size: Tuple[int, int] = (512, 512),
                 augment: bool = False,
                 pairs: Optional[List[Tuple[Path, Path]]] = None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augment = augment

        # 查找匹配的图像-掩膜对
        self.pairs: List[Tuple[Path, Path]] = list(pairs or [])
        if pairs is None:
            for img_path in sorted(self.images_dir.glob("*")):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                    continue
                mask_path = self.masks_dir / f"{img_path.stem}.png"
                if not mask_path.exists():
                    mask_path = self.masks_dir / f"{img_path.stem}.jpg"
                if mask_path.exists():
                    self.pairs.append((img_path, mask_path))

        print(f"Found {len(self.pairs)} image-mask pairs")

        if augment:
            try:
                import albumentations as A
                crop_parameters = inspect.signature(A.RandomResizedCrop).parameters
                if "size" in crop_parameters:
                    random_crop = A.RandomResizedCrop(
                        size=tuple(image_size), scale=(0.8, 1.0)
                    )
                else:
                    random_crop = A.RandomResizedCrop(
                        height=image_size[0], width=image_size[1], scale=(0.8, 1.0)
                    )

                noise_parameters = inspect.signature(A.GaussNoise).parameters
                if "std_range" in noise_parameters:
                    gaussian_noise = A.GaussNoise(std_range=(0.012, 0.028), p=0.2)
                else:
                    # Albumentations 1.x expresses this as variance in the
                    # image's native scale. Inputs are float RGB [0, 1].
                    gaussian_noise = A.GaussNoise(
                        var_limit=(0.012**2, 0.028**2), p=0.2
                    )

                self.transform = A.Compose([
                    random_crop,
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10,
                                         sat_shift_limit=20,
                                         val_shift_limit=10, p=0.3),
                    gaussian_noise,
                    A.Blur(blur_limit=3, p=0.2),
                    A.Normalize(mean=IMAGENET_MEAN.tolist(),
                                std=IMAGENET_STD.tolist(),
                                max_pixel_value=1.0),
                ])
            except ImportError:
                print("WARNING: albumentations not installed, using basic transform")
                self.transform = None
        else:
            try:
                import albumentations as A
                self.transform = A.Compose([
                    A.Resize(height=image_size[0], width=image_size[1]),
                    A.Normalize(mean=IMAGENET_MEAN.tolist(),
                                std=IMAGENET_STD.tolist(),
                                max_pixel_value=1.0),
                ])
            except ImportError:
                self.transform = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # 读取图像
        # Decode every supported integer bit depth to one common float [0, 1]
        # representation before augmentation and ImageNet normalization.
        img = read_image_rgb(img_path, as_float=True)

        # 读取掩膜
        mask = read_image_gray(mask_path)
        mask = (mask > 127).astype(np.float32)

        # 应用变换
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            # 基础处理
            target_size = (self.image_size[1], self.image_size[0])
            img = cv2.resize(img, target_size)
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            img = normalize_imagenet_rgb(img)
            img = img.transpose(2, 0, 1)  # HWC → CHW

        # To tensor
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).float()
            if img_tensor.ndim == 3 and img_tensor.shape[0] not in (1, 3):
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC → CHW
        else:
            img_tensor = img

        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return img_tensor, mask_tensor


# ============================================================
# 损失函数: BCE + Dice Loss (混合损失)
# ============================================================
class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE + Dice 混合损失."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.bce_weight * self.bce(pred, target) +
                self.dice_weight * self.dice(pred, target))


# ============================================================
# 训练指标
# ============================================================
@torch.no_grad()
def compute_iou(pred: torch.Tensor, target: torch.Tensor,
                 threshold: float = 0.5) -> float:
    """计算IoU (Intersection over Union)."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = (pred_bin + target).clamp(0, 1).sum(dim=(1, 2, 3))
    per_sample = torch.where(
        union > 0, intersection / union.clamp_min(1e-8), torch.ones_like(union)
    )
    return float(per_sample.mean().item())


# ============================================================
# 训练主循环
# ============================================================
def train(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ---- 数据 ----
    source_dataset = LeafSegmentationDataset(
        args.images, args.masks, image_size=args.image_size, augment=False
    )
    if len(source_dataset) < 2:
        raise ValueError("At least two image-mask pairs are required for train/validation split")

    train_pairs, val_pairs = split_pairs_by_sample(source_dataset.pairs)

    train_dataset = LeafSegmentationDataset(
        args.images, args.masks, image_size=args.image_size, augment=True,
        pairs=train_pairs,
    )
    val_dataset = LeafSegmentationDataset(
        args.images, args.masks, image_size=args.image_size, augment=False,
        pairs=val_pairs,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=pin_memory)

    # ---- 模型 ----
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name=args.backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    model = model.to(device)

    # ---- 优化器 & 调度器 ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs // 3), T_mult=2, eta_min=args.lr * 0.01
    )

    # ---- 损失 ----
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    # ---- 训练 ----
    best_iou = float("-inf")
    history = {"train_loss": [], "val_loss": [], "val_iou": []}

    for epoch in range(1, args.epochs + 1):
        # -- Training --
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # -- Validation --
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                val_iou += compute_iou(preds, masks)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        scheduler.step()

        # -- 保存最佳模型 --
        if val_iou > best_iou:
            best_iou = val_iou
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "backbone": args.backbone,
                "image_size": list(args.image_size),
                "normalization": {
                    "mean": IMAGENET_MEAN.tolist(),
                    "std": IMAGENET_STD.tolist(),
                },
                "threshold": 0.5,
                "best_val_iou": float(best_iou),
            }, str(output_path))
            improved = " *"
        else:
            improved = ""

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val IoU: {val_iou:.4f}{improved}")

    # ---- 保存训练历史 ----
    import json
    hist_path = Path(args.output).with_suffix(".json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete!")
    print(f"   Best Val IoU: {best_iou:.4f}")
    print(f"   Model saved:  {args.output}")
    print(f"   History saved: {hist_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="U-Net叶片分割模型训练")
    parser.add_argument("--images", required=True,
                        help="训练图像目录")
    parser.add_argument("--masks", required=True,
                        help="训练掩膜目录")
    parser.add_argument("--backbone", default="efficientnet-b3",
                        choices=["efficientnet-b0", "efficientnet-b3",
                                 "resnet34", "resnet50", "mobilenet_v2"],
                        help="编码器骨架 (default: efficientnet-b3)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512],
                        help="输入尺寸 (H W) (default: 512 512)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率 (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="权重衰减 (default: 1e-4)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader工作线程数 (default: 4)")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="训练设备 (default: cuda)")
    parser.add_argument("--output", default="./models/unet_leaf.pth",
                        help="模型输出路径 (default: ./models/unet_leaf.pth)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
