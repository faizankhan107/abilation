#!/usr/bin/env python3
"""
Race Segmentation Training Script

This script implements a complete training pipeline for race segmentation with:
- Fixed CUDA initialization and memory management
- Proper JSON serialization handling
- Updated PyTorch functions (torch.amp instead of torch.cuda.amp)
- Enhanced model architecture with multi-scale features
- Complete training loop with validation and detailed metrics
- Per-class statistics and professional logging
"""

import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, lr_scheduler
import torch.amp as amp  # Updated import to avoid deprecation

import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from tqdm import tqdm
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RaceSegmentationDataset(Dataset):
    """Dataset for race segmentation with class-based directory structure"""
    
    def __init__(self, data_dir, transform=None, is_training=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_training = is_training
        
        # Race classes mapping
        self.classes = {
            'white': 0,
            'black': 1, 
            'east_asian': 2,
            'indian': 3,
            'latino_hispanic': 4,
            'middle_eastern': 5,
            'southeast_asian': 6
        }
        
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """Load image and mask pairs from class-based directory structure"""
        images_dir = self.data_dir / 'images'
        masks_dir = self.data_dir / 'masks'
        
        for class_name, class_idx in self.classes.items():
            class_images_dir = images_dir / class_name
            class_masks_dir = masks_dir / class_name
            
            if class_images_dir.exists() and class_masks_dir.exists():
                for img_file in class_images_dir.glob('*.jpg'):
                    mask_file = class_masks_dir / f"{img_file.stem}.png"
                    if mask_file.exists():
                        self.samples.append({
                            'image': str(img_file),
                            'mask': str(mask_file),
                            'class': class_idx
                        })
        
        logger.info(f"Loaded {len(self.samples)} samples for {'training' if self.is_training else 'validation'}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image']).convert('RGB')
        
        # Load mask
        mask = Image.open(sample['mask']).convert('L')
        mask = np.array(mask)
        
        # Convert color mask to class indices if needed
        mask = self._mask_to_class_indices(mask)
        
        if self.transform:
            # Apply transforms to both image and mask
            transformed = self.transform(image=np.array(image), mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask).long()
            
        return image, mask
    
    def _mask_to_class_indices(self, mask):
        """Convert color mask to class indices"""
        # Assuming grayscale mask with pixel values corresponding to class indices
        # Adjust this based on your actual mask format
        return mask


class EnhancedRaceSegmentationModel(nn.Module):
    """Enhanced race segmentation model with multi-scale features"""
    
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone - ResNet50 with pretrained weights
        self.backbone = resnet50(pretrained=pretrained)
        
        # Remove the final classification layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Multi-scale feature extraction
        self.aspp = ASPP(2048, 256)
        
        # Decoder
        self.decoder = Decoder(256, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply ASPP for multi-scale features
        features = self.aspp(features)
        
        # Decode to segmentation map
        output = self.decoder(features)
        
        # Upsample to input size
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return output
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        size = x.shape[2:]
        
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))
        x4 = F.relu(self.bn4(self.conv4(x)))
        
        x5 = self.global_avg_pool(x)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = F.relu(self.bn_out(self.conv_out(x)))
        x = self.dropout(x)
        
        return x


class Decoder(nn.Module):
    """Decoder module"""
    
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, num_classes, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.conv3(x)
        return x


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Calculate dice coefficient for each class
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Focal Loss and Dice Loss"""
    
    def __init__(self, focal_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice


def calculate_metrics(predictions, targets, num_classes=7):
    """Calculate per-class metrics"""
    metrics = {}
    
    for class_id in range(num_classes):
        # Binary masks for current class
        pred_mask = (predictions == class_id).float()
        target_mask = (targets == class_id).float()
        
        # True positives, false positives, false negatives
        tp = (pred_mask * target_mask).sum()
        fp = (pred_mask * (1 - target_mask)).sum()
        fn = ((1 - pred_mask) * target_mask).sum()
        
        # Calculate metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        
        metrics[class_id] = {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'iou': iou.item()
        }
    
    return metrics


def get_data_transforms():
    """Get data augmentation transforms"""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        train_transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        val_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        return train_transform, val_transform
        
    except ImportError:
        logger.warning("Albumentations not available, using basic transforms")
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    intersection = torch.zeros(7).to(device)
    union = torch.zeros(7).to(device)
    
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if device.type == 'cuda':
            with amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
        else:
            with amp.autocast('cpu'):
                outputs = model(images)
                loss = criterion(outputs, masks)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate metrics
        predictions = torch.argmax(outputs, dim=1)
        correct_pixels += (predictions == masks).sum().item()
        total_pixels += masks.numel()
        
        # Calculate IoU
        for class_id in range(7):
            pred_mask = (predictions == class_id)
            target_mask = (masks == class_id)
            inter = (pred_mask & target_mask).sum().float()
            uni = (pred_mask | target_mask).sum().float()
            intersection[class_id] += inter
            union[class_id] += uni
        
        running_loss += loss.item()
        
        # Update progress bar
        accuracy = correct_pixels / total_pixels
        iou = (intersection / (union + 1e-8)).mean().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{accuracy:.4f}',
            'iou': f'{iou:.4f}'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_pixels / total_pixels
    epoch_iou = (intersection / (union + 1e-8)).mean().item()
    
    return epoch_loss, epoch_acc, epoch_iou


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    intersection = torch.zeros(7).to(device)
    union = torch.zeros(7).to(device)
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(val_loader, desc=f'Val Epoch {epoch}')
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            if device.type == 'cuda':
                with amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                with amp.autocast('cpu'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            
            predictions = torch.argmax(outputs, dim=1)
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()
            
            # Store for detailed metrics
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())
            
            # Calculate IoU
            for class_id in range(7):
                pred_mask = (predictions == class_id)
                target_mask = (masks == class_id)
                inter = (pred_mask & target_mask).sum().float()
                uni = (pred_mask | target_mask).sum().float()
                intersection[class_id] += inter
                union[class_id] += uni
            
            running_loss += loss.item()
            
            # Update progress bar
            accuracy = correct_pixels / total_pixels
            iou = (intersection / (union + 1e-8)).mean().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{accuracy:.4f}',
                'iou': f'{iou:.4f}'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct_pixels / total_pixels
    epoch_iou = (intersection / (union + 1e-8)).mean().item()
    
    # Calculate detailed metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    detailed_metrics = calculate_metrics(all_predictions, all_targets)
    
    return epoch_loss, epoch_acc, epoch_iou, detailed_metrics


def print_metrics_table(metrics, epoch):
    """Print formatted metrics table"""
    print(f"\n{'='*60}")
    print(f"Epoch {epoch} Results:")
    print(f"{'':11} | {'Precision':10} | {'Recall':10} | {'F1':10} | {'IoU':10}")
    print(f"{'-'*60}")
    
    mean_iou = 0
    for class_id in range(7):
        if class_id in metrics:
            m = metrics[class_id]
            print(f"Class {class_id:1}    | {m['precision']:.4f}     | {m['recall']:.4f}     | {m['f1']:.4f}     | {m['iou']:.4f}")
            mean_iou += m['iou']
        else:
            print(f"Class {class_id:1}    | 0.0000     | 0.0000     | 0.0000     | 0.0000")
    
    mean_iou /= 7
    overall_acc = sum(m['precision'] for m in metrics.values()) / len(metrics) if metrics else 0
    
    print(f"{'-'*60}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"{'='*60}")


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, accuracy, iou, checkpoint_dir, is_best=False):
    """Save model checkpoint with proper device handling"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Convert device objects to strings for JSON serialization
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'iou': iou,
        'device': str(next(model.parameters()).device)  # Convert device to string
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(state, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        torch.save(state, best_path)
        logger.info(f"Saved best model with IoU: {iou:.4f}")
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def setup_cuda():
    """Setup CUDA with proper error handling"""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return torch.device('cpu')
    
    try:
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Set CUDA device
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        
        # Test CUDA functionality
        test_tensor = torch.randn(1, 1, device=device)
        del test_tensor
        torch.cuda.empty_cache()
        
        logger.info(f"CUDA initialized successfully on device: {device}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        
        return device
        
    except Exception as e:
        logger.error(f"CUDA initialization failed: {e}")
        logger.warning("Falling back to CPU")
        return torch.device('cpu')


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Race Segmentation Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup CUDA
    device = setup_cuda()
    
    # Create data transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = RaceSegmentationDataset(
        args.data_dir + '/train',
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = RaceSegmentationDataset(
        args.data_dir + '/val',
        transform=val_transform,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    model = EnhancedRaceSegmentationModel(num_classes=7, pretrained=True)
    model = model.to(device)
    
    # Create loss function
    criterion = CombinedLoss(focal_weight=1.0, dice_weight=1.0)
    
    # Create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Create gradient scaler for mixed precision
    if device.type == 'cuda':
        scaler = amp.GradScaler('cuda')
    else:
        scaler = amp.GradScaler('cpu')
    
    # Resume from checkpoint if provided
    start_epoch = 1
    best_iou = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['iou']
        logger.info(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")
    
    # Training loop
    train_log = []
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train
        train_loss, train_acc, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_iou, detailed_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Step scheduler
        scheduler.step()
        
        # Print detailed metrics
        print_metrics_table(detailed_metrics, epoch)
        
        # Log training progress
        epoch_log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_iou': train_iou,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_iou': val_iou,
            'learning_rate': scheduler.get_last_lr()[0],
            'detailed_metrics': {str(k): v for k, v in detailed_metrics.items()}  # Convert keys to strings
        }
        train_log.append(epoch_log)
        
        # Save checkpoint
        is_best = val_iou > best_iou
        if is_best:
            best_iou = val_iou
        
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch,
            val_loss, val_acc, val_iou, args.checkpoint_dir, is_best
        )
        
        # Save training log
        log_path = Path(args.checkpoint_dir) / 'training_log.json'
        with open(log_path, 'w') as f:
            json.dump(train_log, f, indent=2)
        
        # Early stopping check
        if epoch > 10 and val_iou < 0.1:
            logger.warning("Validation IoU too low, consider checking data or model")
    
    logger.info("Training completed!")
    logger.info(f"Best validation IoU: {best_iou:.4f}")


if __name__ == '__main__':
    main()