#!/usr/bin/env python3
"""
Enhanced UTKFace Multi-Model Segmentation Comparison Script

This script provides a comprehensive comparison framework for facial race segmentation
using multiple deep learning models with the UTKFace dataset.

Models supported:
1. UNet
2. DeepLabV3+  
3. PSPNet
4. HRNetV2
5. FaceSeg+
6. EnhancedFaceSegmentationWithRaceNet (NEW)

Features:
- Interactive model selection interface
- UTKFace dataset with 5 race classes (White, Black, Asian, Indian, Others)
- Mixed precision training with gradient scaling
- Early stopping and model checkpointing
- Comprehensive visualization (300 DPI and 600 DPI)
- Multi-GPU support with DataParallel
- Results export to CSV and JSON formats
"""

import os
import sys
import json
import csv
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet101

# Image processing and visualization
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns

# Scientific computing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy import ndimage

# Progress tracking
from tqdm import tqdm

# ========================================================================================
# CONFIGURATION AND CONSTANTS
# ========================================================================================

# UTKFace race classes and color mappings (BGR format)
RACE_CLASSES = ['White', 'Black', 'Asian', 'Indian', 'Others']
NUM_RACE_CLASSES = len(RACE_CLASSES)

# Color mappings for race classes (BGR format)
RACE_COLOR_MAP = {
    0: (255, 250, 250),  # White
    1: (139, 69, 19),    # Black  
    2: (255, 215, 0),    # Asian
    3: (255, 153, 51),   # Indian
    4: (152, 251, 152)   # Others
}

# Reverse mapping for color to class
COLOR_TO_CLASS = {v: k for k, v in RACE_COLOR_MAP.items()}

# Segmentation classes (background and face)
SEGMENTATION_CLASSES = ['Background', 'Face']
NUM_SEGMENTATION_CLASSES = len(SEGMENTATION_CLASSES)

# Model configurations
MODEL_CONFIGS = {
    1: {'name': 'UNet', 'display_name': 'U-Net'},
    2: {'name': 'DeepLabV3Plus', 'display_name': 'DeepLabV3+'},
    3: {'name': 'PSPNet', 'display_name': 'PSPNet'},
    4: {'name': 'HRNetV2', 'display_name': 'HRNetV2'},
    5: {'name': 'FaceSegPlus', 'display_name': 'FaceSeg+'},
    6: {'name': 'EnhancedFaceSegmentationWithRaceNet', 'display_name': 'EnhancedFaceSegmentationWithRaceNet'}
}

# Training hyperparameters
DEFAULT_CONFIG = {
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 4e-5,
    'weight_decay': 1e-4,
    'patience': 10,
    'min_delta': 1e-4,
    'image_size': (256, 256),
    'num_workers': 4,
    'pin_memory': True,
    'mixed_precision': True,
    'gradient_clip_value': 1.0,
    'save_every_n_epochs': 5
}

# ========================================================================================
# LOGGING SETUP
# ========================================================================================

def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    """Setup comprehensive logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('UTKFaceSegmentation')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler for detailed logs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'utkface_segmentation_{timestamp}.log')
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ========================================================================================
# ENHANCED FACE SEGMENTATION WITH RACENET MODEL
# ========================================================================================

class SpatialAttentionModule(nn.Module):
    """Spatial attention mechanism for enhanced feature focus."""
    
    def __init__(self, in_channels: int):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention

class AdaptiveFeatureFusion(nn.Module):
    """Adaptive feature fusion module for multi-scale features."""
    
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super(AdaptiveFeatureFusion, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.attention = nn.Conv2d(len(in_channels_list), len(in_channels_list), 1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, features):
        # Ensure all features have the same spatial size
        target_size = features[0].shape[2:]
        aligned_features = []
        
        for i, (feat, conv) in enumerate(zip(features, self.convs)):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(conv(feat))
        
        # Stack and compute attention weights
        stacked = torch.stack(aligned_features, dim=1)  # [B, N, C, H, W]
        B, N, C, H, W = stacked.shape
        
        # Compute attention weights
        weights = torch.mean(stacked, dim=2)  # [B, N, H, W]
        weights = self.attention(weights)
        weights = self.softmax(weights)
        weights = weights.unsqueeze(2)  # [B, N, 1, H, W]
        
        # Apply attention weights
        fused = torch.sum(stacked * weights, dim=1)  # [B, C, H, W]
        return fused

class BoundaryEnhancementModule(nn.Module):
    """Boundary enhancement processing for improved segmentation edges."""
    
    def __init__(self, in_channels: int):
        super(BoundaryEnhancementModule, self).__init__()
        self.edge_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.boundary_conv = nn.Conv2d(in_channels * 2, in_channels, 1)
        
    def forward(self, x):
        # Edge detection
        edge_x = self.edge_conv(x)
        edge_enhanced = torch.cat([x, edge_x], dim=1)
        boundary_enhanced = self.boundary_conv(edge_enhanced)
        return boundary_enhanced

class ProgressiveRefinementHead(nn.Module):
    """Progressive refinement head for multi-scale prediction."""
    
    def __init__(self, in_channels: int, num_classes: int):
        super(ProgressiveRefinementHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1)
        self.classifier = nn.Conv2d(in_channels // 4, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class PyramidPoolingModule(nn.Module):
    """Multi-scale pyramid feature extraction."""
    
    def __init__(self, in_channels: int, pool_sizes: List[int] = [1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.pool_sizes = pool_sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1)
            for _ in pool_sizes
        ])
        self.final_conv = nn.Conv2d(
            in_channels + (in_channels // len(pool_sizes)) * len(pool_sizes),
            in_channels, 3, padding=1
        )
        
    def forward(self, x):
        h, w = x.shape[2:]
        pyramid_features = [x]
        
        for pool_size, conv in zip(self.pool_sizes, self.convs):
            pooled = F.adaptive_avg_pool2d(x, pool_size)
            pooled = conv(pooled)
            pooled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pyramid_features.append(pooled)
        
        concatenated = torch.cat(pyramid_features, dim=1)
        output = self.final_conv(concatenated)
        return output

class EnhancedFaceSegmentationWithRaceNet(nn.Module):
    """
    Enhanced Face Segmentation model with RaceNet integration.
    Combines spatial attention, adaptive feature fusion, boundary enhancement,
    and progressive refinement for superior performance.
    """
    
    def __init__(self, num_segmentation_classes: int = 2, num_race_classes: int = 5):
        super(EnhancedFaceSegmentationWithRaceNet, self).__init__()
        
        self.num_segmentation_classes = num_segmentation_classes
        self.num_race_classes = num_race_classes
        
        # Backbone (ResNet-50 based encoder)
        backbone = resnet50(pretrained=True)
        self.backbone_layers = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),  # 64 channels
            nn.Sequential(backbone.maxpool, backbone.layer1),            # 256 channels  
            backbone.layer2,                                             # 512 channels
            backbone.layer3,                                             # 1024 channels
            backbone.layer4                                              # 2048 channels
        ])
        
        # Feature channels at each level
        self.feature_channels = [64, 256, 512, 1024, 2048]
        
        # Spatial attention modules
        self.spatial_attentions = nn.ModuleList([
            SpatialAttentionModule(ch) for ch in self.feature_channels
        ])
        
        # Pyramid pooling module
        self.ppm = PyramidPoolingModule(self.feature_channels[-1])
        
        # Adaptive feature fusion
        self.feature_fusion = AdaptiveFeatureFusion(self.feature_channels, 256)
        
        # Boundary enhancement
        self.boundary_enhancement = BoundaryEnhancementModule(256)
        
        # Progressive refinement heads
        self.segmentation_head = ProgressiveRefinementHead(256, num_segmentation_classes)
        self.race_head = ProgressiveRefinementHead(256, num_race_classes)
        
        # Global average pooling for race classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.race_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_race_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Extract multi-scale features with spatial attention
        features = []
        current = x
        
        for i, (layer, attention) in enumerate(zip(self.backbone_layers, self.spatial_attentions)):
            current = layer(current)
            attended = attention(current)
            features.append(attended)
        
        # Apply pyramid pooling to the deepest features
        features[-1] = self.ppm(features[-1])
        
        # Adaptive feature fusion
        fused_features = self.feature_fusion(features)
        
        # Boundary enhancement
        enhanced_features = self.boundary_enhancement(fused_features)
        
        # Progressive refinement for segmentation
        segmentation_logits = self.segmentation_head(enhanced_features)
        segmentation_logits = F.interpolate(
            segmentation_logits, size=input_size, mode='bilinear', align_corners=False
        )
        
        # Race classification
        race_features = self.global_pool(enhanced_features)
        race_features = race_features.flatten(1)
        race_logits = self.race_classifier(race_features)
        
        # Progressive refinement for race segmentation
        race_segmentation_logits = self.race_head(enhanced_features)
        race_segmentation_logits = F.interpolate(
            race_segmentation_logits, size=input_size, mode='bilinear', align_corners=False
        )
        
        return {
            'segmentation': segmentation_logits,
            'race_classification': race_logits,
            'race_segmentation': race_segmentation_logits
        }

# ========================================================================================
# UTKFACE DATASET PROCESSING
# ========================================================================================

class UTKFaceDataset(Dataset):
    """UTKFace dataset loader with race segmentation support."""
    
    def __init__(self, 
                 image_dir: str,
                 mask_dir: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 image_size: Tuple[int, int] = (256, 256)):
        """
        Initialize UTKFace dataset.
        
        Args:
            image_dir: Directory containing UTKFace images
            mask_dir: Directory containing segmentation masks
            split: 'train' or 'val'
            transform: Image transformations
            image_size: Target image size (height, width)
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Load image paths and parse metadata
        self.image_paths = []
        self.race_labels = []
        self.ages = []
        self.genders = []
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Load and parse UTKFace dataset files."""
        image_extensions = ['.jpg', '.jpeg', '.png']
        
        for img_path in self.image_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                # Parse UTKFace filename: age_gender_race_date&time.jpg
                try:
                    filename = img_path.stem
                    parts = filename.split('_')
                    
                    if len(parts) >= 3:
                        age = int(parts[0])
                        gender = int(parts[1])  # 0: male, 1: female
                        race = int(parts[2])    # 0: White, 1: Black, 2: Asian, 3: Indian, 4: Others
                        
                        # Check if corresponding mask exists
                        mask_path = self.mask_dir / f"{filename}_mask.png"
                        if mask_path.exists():
                            self.image_paths.append(img_path)
                            self.race_labels.append(race)
                            self.ages.append(age)
                            self.genders.append(gender)
                            
                except (ValueError, IndexError):
                    # Skip files with invalid naming convention
                    continue
    
    def _load_mask(self, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process segmentation mask.
        
        Returns:
            face_mask: Binary mask for face/background segmentation
            race_mask: Mask with race class labels
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Convert BGR to RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Create face/background mask
        face_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        race_mask = np.full(mask.shape[:2], -1, dtype=np.int8)  # -1 for background (ignore)
        
        # Process each race class
        for race_id, color in RACE_COLOR_MAP.items():
            # Convert BGR to RGB for comparison
            rgb_color = (color[2], color[1], color[0])
            
            # Find pixels matching this race color
            color_mask = np.all(mask == rgb_color, axis=2)
            
            if np.any(color_mask):
                face_mask[color_mask] = 1  # Face region
                race_mask[color_mask] = race_id
        
        return face_mask, race_mask
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        image_path = self.image_paths[idx]
        race_label = self.race_labels[idx]
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        # Load mask
        mask_path = self.mask_dir / f"{image_path.stem}_mask.png"
        face_mask, race_mask = self._load_mask(mask_path)
        
        # Apply transformations
        if self.transform:
            # Create PIL Images for torchvision transforms
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Convert masks to tensors
        face_mask = torch.from_numpy(face_mask).long()
        race_mask = torch.from_numpy(race_mask).long()
        
        return {
            'image': image,
            'face_mask': face_mask,
            'race_mask': race_mask,
            'race_label': torch.tensor(race_label, dtype=torch.long),
            'age': torch.tensor(self.ages[idx], dtype=torch.float),
            'gender': torch.tensor(self.genders[idx], dtype=torch.long),
            'image_path': str(image_path)
        }

def create_utkface_transforms(split: str, image_size: Tuple[int, int]) -> transforms.Compose:
    """Create data augmentation transforms for UTKFace dataset."""
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:  # validation
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

def split_utkface_dataset(dataset_dir: str, 
                         train_ratio: float = 0.8,
                         stratify_by_race: bool = True) -> Tuple[List[str], List[str]]:
    """
    Split UTKFace dataset into train and validation sets.
    
    Args:
        dataset_dir: Path to UTKFace dataset directory
        train_ratio: Ratio of training data (0.0 to 1.0)
        stratify_by_race: Whether to stratify split by race
    
    Returns:
        train_files: List of training image filenames
        val_files: List of validation image filenames
    """
    dataset_path = Path(dataset_dir)
    image_files = []
    race_labels = []
    
    # Parse all image files
    for img_path in dataset_path.iterdir():
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                filename = img_path.stem
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    race = int(parts[2])
                    image_files.append(filename)
                    race_labels.append(race)
                    
            except (ValueError, IndexError):
                continue
    
    # Split dataset
    if stratify_by_race and len(set(race_labels)) > 1:
        train_files, val_files = train_test_split(
            image_files,
            test_size=1.0 - train_ratio,
            stratify=race_labels,
            random_state=42
        )
    else:
        train_files, val_files = train_test_split(
            image_files,
            test_size=1.0 - train_ratio,
            random_state=42
        )
    
    return train_files, val_files

# ========================================================================================
# OTHER MODEL IMPLEMENTATIONS
# ========================================================================================

class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution."""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """U-Net architecture for segmentation."""
    
    def __init__(self, n_channels=3, num_segmentation_classes=2, num_race_classes=5, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.num_segmentation_classes = num_segmentation_classes
        self.num_race_classes = num_race_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output heads
        self.segmentation_head = OutConv(64, num_segmentation_classes)
        self.race_head = OutConv(64, num_race_classes)
        
        # Global pooling for race classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.race_classifier = nn.Linear(64, num_race_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Segmentation output
        segmentation_logits = self.segmentation_head(x)
        
        # Race segmentation output
        race_segmentation_logits = self.race_head(x)
        
        # Race classification output
        race_features = self.global_pool(x)
        race_features = race_features.flatten(1)
        race_logits = self.race_classifier(race_features)
        
        return {
            'segmentation': segmentation_logits,
            'race_classification': race_logits,
            'race_segmentation': race_segmentation_logits
        }

# Simplified placeholder implementations for other models
# These would need full implementations in a production system
class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ architecture for segmentation.""" 
    def __init__(self, num_segmentation_classes=2, num_race_classes=5):
        super().__init__()
        # Simplified implementation - would need full DeepLabV3+ architecture
        self.backbone = resnet50(pretrained=True)
        self.segmentation_head = nn.Conv2d(2048, num_segmentation_classes, 1)
        self.race_head = nn.Conv2d(2048, num_race_classes, 1)
        self.race_classifier = nn.Linear(2048, num_race_classes)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # Simplified forward pass
        features = self.backbone.layer4(self.backbone.layer3(self.backbone.layer2(self.backbone.layer1(
            self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))))))
        
        seg_out = self.segmentation_head(features)
        race_seg_out = self.race_head(features)
        
        race_features = self.global_pool(features).flatten(1)
        race_cls_out = self.race_classifier(race_features)
        
        return {
            'segmentation': F.interpolate(seg_out, size=x.shape[2:], mode='bilinear'),
            'race_classification': race_cls_out,
            'race_segmentation': F.interpolate(race_seg_out, size=x.shape[2:], mode='bilinear')
        }

class PSPNet(nn.Module):
    """PSPNet architecture for segmentation."""
    def __init__(self, num_segmentation_classes=2, num_race_classes=5):
        super().__init__()
        # Simplified implementation
        self.backbone = resnet50(pretrained=True)
        self.psp_module = PyramidPoolingModule(2048)
        self.segmentation_head = nn.Conv2d(2048, num_segmentation_classes, 1)
        self.race_head = nn.Conv2d(2048, num_race_classes, 1)
        self.race_classifier = nn.Linear(2048, num_race_classes)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # Simplified forward pass
        features = self.backbone.layer4(self.backbone.layer3(self.backbone.layer2(self.backbone.layer1(
            self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))))))
        
        features = self.psp_module(features)
        
        seg_out = self.segmentation_head(features)
        race_seg_out = self.race_head(features)
        
        race_features = self.global_pool(features).flatten(1)
        race_cls_out = self.race_classifier(race_features)
        
        return {
            'segmentation': F.interpolate(seg_out, size=x.shape[2:], mode='bilinear'),
            'race_classification': race_cls_out,
            'race_segmentation': F.interpolate(race_seg_out, size=x.shape[2:], mode='bilinear')
        }

class HRNetV2(nn.Module):
    """HRNetV2 architecture for segmentation."""
    def __init__(self, num_segmentation_classes=2, num_race_classes=5):
        super().__init__()
        # Simplified implementation
        self.backbone = resnet50(pretrained=True)
        self.segmentation_head = nn.Conv2d(2048, num_segmentation_classes, 1)
        self.race_head = nn.Conv2d(2048, num_race_classes, 1)
        self.race_classifier = nn.Linear(2048, num_race_classes)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # Simplified forward pass
        features = self.backbone.layer4(self.backbone.layer3(self.backbone.layer2(self.backbone.layer1(
            self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))))))
        
        seg_out = self.segmentation_head(features)
        race_seg_out = self.race_head(features)
        
        race_features = self.global_pool(features).flatten(1)
        race_cls_out = self.race_classifier(race_features)
        
        return {
            'segmentation': F.interpolate(seg_out, size=x.shape[2:], mode='bilinear'),
            'race_classification': race_cls_out,
            'race_segmentation': F.interpolate(race_seg_out, size=x.shape[2:], mode='bilinear')
        }

class FaceSegPlus(nn.Module):
    """FaceSeg+ architecture for segmentation."""
    def __init__(self, num_segmentation_classes=2, num_race_classes=5):
        super().__init__()
        # Simplified implementation
        self.backbone = resnet50(pretrained=True)
        self.segmentation_head = nn.Conv2d(2048, num_segmentation_classes, 1)
        self.race_head = nn.Conv2d(2048, num_race_classes, 1)
        self.race_classifier = nn.Linear(2048, num_race_classes)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # Simplified forward pass
        features = self.backbone.layer4(self.backbone.layer3(self.backbone.layer2(self.backbone.layer1(
            self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))))))
        
        seg_out = self.segmentation_head(features)
        race_seg_out = self.race_head(features)
        
        race_features = self.global_pool(features).flatten(1)
        race_cls_out = self.race_classifier(race_features)
        
        return {
            'segmentation': F.interpolate(seg_out, size=x.shape[2:], mode='bilinear'),
            'race_classification': race_cls_out,
            'race_segmentation': F.interpolate(race_seg_out, size=x.shape[2:], mode='bilinear')
        }

# ========================================================================================
# TRAINING AND EVALUATION PIPELINE
# ========================================================================================

class SegmentationTrainer:
    """Training pipeline for segmentation models."""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any],
                 logger: logging.Logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device and enable multi-GPU if available
        self.model = self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if config.get('mixed_precision', False) else None
        
        # Loss functions
        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.race_criterion = nn.CrossEntropyLoss()
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config['patience'] // 2,
            verbose=True
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_seg_loss = 0
        total_race_loss = 0
        num_batches = len(train_loader)
        
        # Segmentation metrics
        seg_correct = 0
        seg_total = 0
        seg_intersection = torch.zeros(NUM_SEGMENTATION_CLASSES)
        seg_union = torch.zeros(NUM_SEGMENTATION_CLASSES)
        
        # Race classification metrics
        race_correct = 0
        race_total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            face_masks = batch['face_mask'].to(self.device)
            race_masks = batch['race_mask'].to(self.device)
            race_labels = batch['race_label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    seg_loss = self.seg_criterion(outputs['segmentation'], face_masks)
                    race_loss = self.race_criterion(outputs['race_classification'], race_labels)
                    loss = seg_loss + race_loss
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip_value'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip_value']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                seg_loss = self.seg_criterion(outputs['segmentation'], face_masks)
                race_loss = self.race_criterion(outputs['race_classification'], race_labels)
                loss = seg_loss + race_loss
                
                loss.backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip_value'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip_value']
                    )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_race_loss += race_loss.item()
            
            # Segmentation accuracy
            seg_pred = torch.argmax(outputs['segmentation'], dim=1)
            valid_pixels = face_masks != -1
            seg_correct += (seg_pred[valid_pixels] == face_masks[valid_pixels]).sum().item()
            seg_total += valid_pixels.sum().item()
            
            # Race classification accuracy
            race_pred = torch.argmax(outputs['race_classification'], dim=1)
            race_correct += (race_pred == race_labels).sum().item()
            race_total += race_labels.size(0)
            
            # IoU calculation for segmentation
            for class_idx in range(NUM_SEGMENTATION_CLASSES):
                pred_mask = seg_pred == class_idx
                true_mask = face_masks == class_idx
                
                intersection = (pred_mask & true_mask).sum().float()
                union = (pred_mask | true_mask).sum().float()
                
                seg_intersection[class_idx] += intersection
                seg_union[class_idx] += union
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Seg_Acc': f'{seg_correct/seg_total:.4f}',
                'Race_Acc': f'{race_correct/race_total:.4f}'
            })
        
        # Calculate final metrics
        avg_loss = total_loss / num_batches
        avg_seg_loss = total_seg_loss / num_batches
        avg_race_loss = total_race_loss / num_batches
        seg_accuracy = seg_correct / seg_total
        race_accuracy = race_correct / race_total
        
        # Calculate IoU
        seg_iou = seg_intersection / (seg_union + 1e-8)
        seg_miou = seg_iou.mean().item()
        
        return {
            'loss': avg_loss,
            'segmentation_loss': avg_seg_loss,
            'race_loss': avg_race_loss,
            'segmentation_accuracy': seg_accuracy,
            'race_accuracy': race_accuracy,
            'segmentation_miou': seg_miou,
            'segmentation_iou_per_class': seg_iou.tolist()
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0
        total_seg_loss = 0
        total_race_loss = 0
        num_batches = len(val_loader)
        
        # Segmentation metrics
        seg_correct = 0
        seg_total = 0
        seg_intersection = torch.zeros(NUM_SEGMENTATION_CLASSES)
        seg_union = torch.zeros(NUM_SEGMENTATION_CLASSES)
        
        # Race classification metrics
        race_correct = 0
        race_total = 0
        race_predictions = []
        race_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                face_masks = batch['face_mask'].to(self.device)
                race_masks = batch['race_mask'].to(self.device)
                race_labels = batch['race_label'].to(self.device)
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        seg_loss = self.seg_criterion(outputs['segmentation'], face_masks)
                        race_loss = self.race_criterion(outputs['race_classification'], race_labels)
                        loss = seg_loss + race_loss
                else:
                    outputs = self.model(images)
                    seg_loss = self.seg_criterion(outputs['segmentation'], face_masks)
                    race_loss = self.race_criterion(outputs['race_classification'], race_labels)
                    loss = seg_loss + race_loss
                
                # Update metrics
                total_loss += loss.item()
                total_seg_loss += seg_loss.item()
                total_race_loss += race_loss.item()
                
                # Segmentation accuracy
                seg_pred = torch.argmax(outputs['segmentation'], dim=1)
                valid_pixels = face_masks != -1
                seg_correct += (seg_pred[valid_pixels] == face_masks[valid_pixels]).sum().item()
                seg_total += valid_pixels.sum().item()
                
                # Race classification accuracy
                race_pred = torch.argmax(outputs['race_classification'], dim=1)
                race_correct += (race_pred == race_labels).sum().item()
                race_total += race_labels.size(0)
                
                # Store predictions for detailed analysis
                race_predictions.extend(race_pred.cpu().numpy())
                race_targets.extend(race_labels.cpu().numpy())
                
                # IoU calculation for segmentation
                for class_idx in range(NUM_SEGMENTATION_CLASSES):
                    pred_mask = seg_pred == class_idx
                    true_mask = face_masks == class_idx
                    
                    intersection = (pred_mask & true_mask).sum().float()
                    union = (pred_mask | true_mask).sum().float()
                    
                    seg_intersection[class_idx] += intersection
                    seg_union[class_idx] += union
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Seg_Acc': f'{seg_correct/seg_total:.4f}',
                    'Race_Acc': f'{race_correct/race_total:.4f}'
                })
        
        # Calculate final metrics
        avg_loss = total_loss / num_batches
        avg_seg_loss = total_seg_loss / num_batches
        avg_race_loss = total_race_loss / num_batches
        seg_accuracy = seg_correct / seg_total
        race_accuracy = race_correct / race_total
        
        # Calculate IoU
        seg_iou = seg_intersection / (seg_union + 1e-8)
        seg_miou = seg_iou.mean().item()
        
        # Calculate per-class race classification metrics
        race_report = classification_report(
            race_targets, race_predictions, 
            target_names=RACE_CLASSES,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'segmentation_loss': avg_seg_loss,
            'race_loss': avg_race_loss,
            'segmentation_accuracy': seg_accuracy,
            'race_accuracy': race_accuracy,
            'segmentation_miou': seg_miou,
            'segmentation_iou_per_class': seg_iou.tolist(),
            'race_classification_report': race_report,
            'race_predictions': race_predictions,
            'race_targets': race_targets
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int) -> Dict[str, List]:
        """Full training loop."""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} "
                f"- Train Loss: {train_metrics['loss']:.4f} "
                f"- Val Loss: {val_metrics['loss']:.4f} "
                f"- Val mIoU: {val_metrics['segmentation_miou']:.4f} "
                f"- Race Acc: {val_metrics['race_accuracy']:.4f} "
                f"- Time: {epoch_time:.1f}s"
            )
            
            # Store metrics
            self.train_metrics.append({**train_metrics, 'epoch': epoch+1, 'time': epoch_time})
            self.val_metrics.append({**val_metrics, 'epoch': epoch+1})
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config['patience']:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_every_n_epochs', 5) == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        return {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if is_best:
            filename = f"{model_to_save.__class__.__name__}_best_model.pth"
            torch.save(checkpoint, filename)
            self.logger.info(f"Best model saved to {filename}")
        else:
            filename = f"{model_to_save.__class__.__name__}_epoch_{epoch+1}.pth"
            torch.save(checkpoint, filename)

# ========================================================================================
# VISUALIZATION AND RESULTS EXPORT
# ========================================================================================

class ResultsVisualizer:
    """Comprehensive visualization and results export."""
    
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set high-quality plot parameters
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 12
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_training_curves(self, 
                           train_metrics: List[Dict],
                           val_metrics: List[Dict],
                           model_name: str,
                           save_600dpi: bool = True):
        """Plot comprehensive training curves."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_name} - Training Progress', fontsize=16, fontweight='bold')
        
        epochs = [m['epoch'] for m in train_metrics]
        
        # Loss curves
        train_losses = [m['loss'] for m in train_metrics]
        val_losses = [m['loss'] for m in val_metrics]
        
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Segmentation mIoU
        train_miou = [m['segmentation_miou'] for m in train_metrics]
        val_miou = [m['segmentation_miou'] for m in val_metrics]
        
        axes[0, 1].plot(epochs, train_miou, 'b-', label='Training', linewidth=2)
        axes[0, 1].plot(epochs, val_miou, 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Segmentation mIoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mIoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Race classification accuracy
        train_race_acc = [m['race_accuracy'] for m in train_metrics]
        val_race_acc = [m['race_accuracy'] for m in val_metrics]
        
        axes[0, 2].plot(epochs, train_race_acc, 'b-', label='Training', linewidth=2)
        axes[0, 2].plot(epochs, val_race_acc, 'r-', label='Validation', linewidth=2)
        axes[0, 2].set_title('Race Classification Accuracy')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Segmentation accuracy
        train_seg_acc = [m['segmentation_accuracy'] for m in train_metrics]
        val_seg_acc = [m['segmentation_accuracy'] for m in val_metrics]
        
        axes[1, 0].plot(epochs, train_seg_acc, 'b-', label='Training', linewidth=2)
        axes[1, 0].plot(epochs, val_seg_acc, 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_title('Segmentation Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss components
        train_seg_loss = [m['segmentation_loss'] for m in train_metrics]
        train_race_loss = [m['race_loss'] for m in train_metrics]
        val_seg_loss = [m['segmentation_loss'] for m in val_metrics]
        val_race_loss = [m['race_loss'] for m in val_metrics]
        
        axes[1, 1].plot(epochs, train_seg_loss, 'b-', label='Train Seg', linewidth=2)
        axes[1, 1].plot(epochs, train_race_loss, 'g-', label='Train Race', linewidth=2)
        axes[1, 1].plot(epochs, val_seg_loss, 'r--', label='Val Seg', linewidth=2)
        axes[1, 1].plot(epochs, val_race_loss, 'm--', label='Val Race', linewidth=2)
        axes[1, 1].set_title('Loss Components')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Training time per epoch
        epoch_times = [m['time'] for m in train_metrics]
        axes[1, 2].bar(epochs, epoch_times, alpha=0.7, color='skyblue')
        axes[1, 2].set_title('Training Time per Epoch')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Time (seconds)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        base_filename = self.results_dir / f"{model_name}_training_curves"
        plt.savefig(f"{base_filename}_300dpi.png", dpi=300, bbox_inches='tight')
        
        if save_600dpi:
            plt.savefig(f"{base_filename}_600dpi.png", dpi=600, bbox_inches='tight')
        
        plt.close()
    
    def plot_confusion_matrix(self, 
                            predictions: List[int],
                            targets: List[int],
                            model_name: str,
                            save_600dpi: bool = True):
        """Plot race classification confusion matrix."""
        
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=RACE_CLASSES,
                   yticklabels=RACE_CLASSES)
        plt.title(f'{model_name} - Race Classification Confusion Matrix', fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Save plots
        base_filename = self.results_dir / f"{model_name}_confusion_matrix"
        plt.savefig(f"{base_filename}_300dpi.png", dpi=300, bbox_inches='tight')
        
        if save_600dpi:
            plt.savefig(f"{base_filename}_600dpi.png", dpi=600, bbox_inches='tight')
        
        plt.close()
    
    def export_metrics_csv(self, 
                          all_results: Dict[str, Any],
                          filename: str = 'model_comparison_results.csv'):
        """Export comprehensive metrics to CSV."""
        
        rows = []
        
        for model_name, results in all_results.items():
            if 'val_metrics' in results and results['val_metrics']:
                # Get final epoch metrics
                final_metrics = results['val_metrics'][-1]
                
                row = {
                    'model_name': model_name,
                    'final_epoch': final_metrics['epoch'],
                    'val_loss': final_metrics['loss'],
                    'val_segmentation_loss': final_metrics['segmentation_loss'],
                    'val_race_loss': final_metrics['race_loss'],
                    'val_segmentation_accuracy': final_metrics['segmentation_accuracy'],
                    'val_segmentation_miou': final_metrics['segmentation_miou'],
                    'val_race_accuracy': final_metrics['race_accuracy'],
                }
                
                # Add per-class IoU
                if 'segmentation_iou_per_class' in final_metrics:
                    for i, class_name in enumerate(SEGMENTATION_CLASSES):
                        row[f'val_{class_name.lower()}_iou'] = final_metrics['segmentation_iou_per_class'][i]
                
                # Add race classification metrics
                if 'race_classification_report' in final_metrics:
                    race_report = final_metrics['race_classification_report']
                    for race_class in RACE_CLASSES:
                        if race_class.lower() in race_report:
                            class_metrics = race_report[race_class.lower()]
                            row[f'race_{race_class.lower()}_precision'] = class_metrics['precision']
                            row[f'race_{race_class.lower()}_recall'] = class_metrics['recall']
                            row[f'race_{race_class.lower()}_f1'] = class_metrics['f1-score']
                
                # Add model parameters and FLOPs (placeholder)
                row['model_parameters'] = results.get('model_parameters', 0)
                row['model_flops'] = results.get('model_flops', 0)
                row['inference_time_ms'] = results.get('inference_time_ms', 0)
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.results_dir / filename, index=False)
        
        return df
    
    def export_metrics_json(self, 
                           all_results: Dict[str, Any],
                           filename: str = 'model_comparison_results.json'):
        """Export comprehensive results to JSON."""
        
        # Clean results for JSON serialization
        json_results = {}
        
        for model_name, results in all_results.items():
            json_results[model_name] = {
                'config': results.get('config', {}),
                'train_metrics': results.get('train_metrics', []),
                'val_metrics': results.get('val_metrics', []),
                'model_info': {
                    'parameters': results.get('model_parameters', 0),
                    'flops': results.get('model_flops', 0),
                    'inference_time_ms': results.get('inference_time_ms', 0)
                }
            }
        
        with open(self.results_dir / filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
    
    def create_model_comparison_plot(self, 
                                   all_results: Dict[str, Any],
                                   save_600dpi: bool = True):
        """Create comprehensive model comparison visualization."""
        
        model_names = list(all_results.keys())
        metrics_to_compare = [
            'val_segmentation_miou',
            'val_race_accuracy', 
            'val_segmentation_accuracy',
            'inference_time_ms'
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics_to_compare):
            ax = axes[idx // 2, idx % 2]
            
            values = []
            labels = []
            
            for model_name, results in all_results.items():
                if 'val_metrics' in results and results['val_metrics']:
                    final_metrics = results['val_metrics'][-1]
                    
                    if metric == 'inference_time_ms':
                        value = results.get('inference_time_ms', 0)
                    else:
                        value = final_metrics.get(metric, 0)
                    
                    values.append(value)
                    labels.append(model_name)
            
            if values:
                bars = ax.bar(labels, values, alpha=0.7)
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plots
        base_filename = self.results_dir / "model_comparison"
        plt.savefig(f"{base_filename}_300dpi.png", dpi=300, bbox_inches='tight')
        
        if save_600dpi:
            plt.savefig(f"{base_filename}_600dpi.png", dpi=600, bbox_inches='tight')
        
        plt.close()

# ========================================================================================
# EXTENDED MAIN APPLICATION CLASS
# ========================================================================================

class UTKFaceSegmentationComparison:
    """Main application class for UTKFace segmentation model comparison."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model registry
        self.model_registry = {
            'UNet': UNet,
            'DeepLabV3Plus': DeepLabV3Plus,
            'PSPNet': PSPNet,
            'HRNetV2': HRNetV2,
            'FaceSegPlus': FaceSegPlus,
            'EnhancedFaceSegmentationWithRaceNet': EnhancedFaceSegmentationWithRaceNet
        }
        
        # Results storage
        self.results = {}
        
        # Initialize visualizer
        self.visualizer = ResultsVisualizer()
        
    def display_banner(self):
        """Display application banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    Enhanced UTKFace Multi-Model Segmentation                     ║
║                              Comparison Framework                                ║  
║                                                                                  ║
║  Models: UNet | DeepLabV3+ | PSPNet | HRNetV2 | FaceSeg+ | EnhancedRaceNet     ║
║  Dataset: UTKFace with 5 race classes (White, Black, Asian, Indian, Others)    ║
║  Features: Mixed Precision | Early Stopping | Multi-GPU | Visualization        ║
╚══════════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        self.logger.info("UTKFace Segmentation Comparison Framework initialized")
    
    def select_models_interactive(self) -> List[int]:
        """Interactive model selection interface."""
        self.display_banner()
        
        print("\n" + "="*80)
        print("                          MODEL SELECTION")
        print("="*80)
        
        for model_id, config in MODEL_CONFIGS.items():
            status = "✓ Available" if config['name'] in self.model_registry else "⚠ Not implemented"
            print(f"  {model_id}. {config['display_name']:<35} {status}")
        
        print("\n" + "-"*80)
        print("Selection Options:")
        print("  • Enter model numbers separated by commas (e.g., 1,2,6)")
        print("  • Enter 'all' to select all available models")
        print("  • Enter 'skip-trained' to skip already trained models")
        print("  • Press Enter for default selection (1,2,3,4,5,6)")
        print("-"*80)
        
        while True:
            try:
                selection = input("\nEnter your selection: ").strip()
                
                if not selection:
                    # Default selection - all models
                    selected = list(MODEL_CONFIGS.keys())
                    break
                elif selection.lower() == 'all':
                    selected = list(MODEL_CONFIGS.keys())
                    break
                elif selection.lower() == 'skip-trained':
                    selected = self._get_untrained_models()
                    break
                else:
                    # Parse comma-separated list
                    selected = [int(x.strip()) for x in selection.split(',')]
                    # Validate selection
                    invalid = [x for x in selected if x not in MODEL_CONFIGS]
                    if invalid:
                        print(f"❌ Invalid model numbers: {invalid}")
                        continue
                    break
                    
            except ValueError:
                print("❌ Invalid input format. Please use comma-separated numbers.")
                continue
        
        # Display selected models
        print(f"\n{'='*80}")
        print("SELECTED MODELS:")
        print(f"{'='*80}")
        for model_id in selected:
            config = MODEL_CONFIGS[model_id]
            print(f"  ✓ {config['display_name']}")
        
        print(f"\nTotal models selected: {len(selected)}")
        self.logger.info(f"Selected models: {[MODEL_CONFIGS[m]['name'] for m in selected]}")
        
        return selected
    
    def _get_untrained_models(self) -> List[int]:
        """Get list of models that haven't been trained yet."""
        # Check for existing model files
        untrained = []
        for model_id, config in MODEL_CONFIGS.items():
            model_file = f"{config['name']}_best_model.pth"
            if not os.path.exists(model_file):
                untrained.append(model_id)
        return untrained if untrained else list(MODEL_CONFIGS.keys())
    
    def setup_dataset(self, dataset_dir: str) -> Tuple[DataLoader, DataLoader]:
        """Setup UTKFace dataset loaders."""
        self.logger.info("Setting up UTKFace dataset...")
        
        # Check if dataset directory exists
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Create subdirectories if needed
        images_dir = os.path.join(dataset_dir, 'images')
        masks_dir = os.path.join(dataset_dir, 'masks')
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            raise FileNotFoundError(
                f"Required subdirectories not found. Expected: {images_dir} and {masks_dir}"
            )
        
        # Create transforms
        train_transform = create_utkface_transforms('train', self.config['image_size'])
        val_transform = create_utkface_transforms('val', self.config['image_size'])
        
        # Split dataset
        train_files, val_files = split_utkface_dataset(images_dir)
        self.logger.info(f"Dataset split - Train: {len(train_files)}, Val: {len(val_files)}")
        
        # Create datasets
        train_dataset = UTKFaceDataset(
            image_dir=images_dir,
            mask_dir=masks_dir,
            split='train',
            transform=train_transform,
            image_size=self.config['image_size']
        )
        
        val_dataset = UTKFaceDataset(
            image_dir=images_dir,
            mask_dir=masks_dir,
            split='val',
            transform=val_transform,
            image_size=self.config['image_size']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
        
        self.logger.info(f"Created data loaders - Train batches: {len(train_loader)}, "
                        f"Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def calculate_model_stats(self, model: nn.Module) -> Tuple[int, int, float]:
        """Calculate model parameters, FLOPs, and inference time."""
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate FLOPs (simplified calculation)
        # This is a rough estimate - would need more sophisticated tools for exact calculation
        flops = 0
        input_size = (1, 3, *self.config['image_size'])
        
        # Simple inference time measurement
        model.eval()
        dummy_input = torch.randn(input_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Measure inference time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_inference_time = ((end_time - start_time) / 100) * 1000  # Convert to milliseconds
        
        return total_params, flops, avg_inference_time
    
    def train_single_model(self, 
                          model_id: int, 
                          train_loader: DataLoader, 
                          val_loader: DataLoader) -> Dict[str, Any]:
        """Train a single model."""
        
        model_config = MODEL_CONFIGS[model_id]
        model_name = model_config['name']
        display_name = model_config['display_name']
        
        self.logger.info(f"Starting training for {display_name}")
        
        # Check if model is already trained
        best_model_path = f"{model_name}_best_model.pth"
        if os.path.exists(best_model_path):
            response = input(f"Model {display_name} already trained. Retrain? (y/n): ")
            if response.lower() != 'y':
                self.logger.info(f"Skipping {display_name} training")
                return self._load_existing_results(model_name)
        
        # Initialize model
        model_class = self.model_registry[model_name]
        model = model_class(
            num_segmentation_classes=NUM_SEGMENTATION_CLASSES,
            num_race_classes=NUM_RACE_CLASSES
        )
        
        # Calculate model statistics
        model_params, model_flops, inference_time = self.calculate_model_stats(model)
        self.logger.info(f"{display_name} - Parameters: {model_params:,}, "
                        f"Inference: {inference_time:.2f}ms")
        
        # Initialize trainer
        trainer = SegmentationTrainer(model, self.config, self.logger)
        
        # Train model
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config['num_epochs']
        )
        
        # Store results
        results = {
            'model_name': model_name,
            'display_name': display_name,
            'config': self.config.copy(),
            'model_parameters': model_params,
            'model_flops': model_flops,
            'inference_time_ms': inference_time,
            **training_results
        }
        
        # Generate visualizations
        self.visualizer.plot_training_curves(
            train_metrics=results['train_metrics'],
            val_metrics=results['val_metrics'],
            model_name=display_name,
            save_600dpi=True
        )
        
        # Generate confusion matrix for final validation results
        if results['val_metrics']:
            final_val_metrics = results['val_metrics'][-1]
            if 'race_predictions' in final_val_metrics and 'race_targets' in final_val_metrics:
                self.visualizer.plot_confusion_matrix(
                    predictions=final_val_metrics['race_predictions'],
                    targets=final_val_metrics['race_targets'],
                    model_name=display_name,
                    save_600dpi=True
                )
        
        self.logger.info(f"Completed training for {display_name}")
        return results
    
    def _load_existing_results(self, model_name: str) -> Dict[str, Any]:
        """Load results from existing model files."""
        # This is a placeholder - would load from saved results files
        return {
            'model_name': model_name,
            'display_name': MODEL_CONFIGS[[k for k, v in MODEL_CONFIGS.items() 
                                         if v['name'] == model_name][0]]['display_name'],
            'train_metrics': [],
            'val_metrics': [],
            'model_parameters': 0,
            'model_flops': 0,
            'inference_time_ms': 0
        }
    
    def run_comparison(self, selected_models: List[int], dataset_dir: str) -> Dict[str, Any]:
        """Run complete model comparison pipeline."""
        
        self.logger.info("Starting UTKFace segmentation model comparison")
        
        # Setup dataset
        try:
            train_loader, val_loader = self.setup_dataset(dataset_dir)
        except FileNotFoundError as e:
            self.logger.error(f"Dataset setup failed: {e}")
            print(f"\n❌ {e}")
            print("\nPlease ensure your dataset is organized as:")
            print("  dataset_dir/")
            print("    ├── images/")
            print("    │   ├── age_gender_race_*.jpg")
            print("    │   └── ...")
            print("    └── masks/")
            print("        ├── age_gender_race_*_mask.png")
            print("        └── ...")
            return {}
        
        # Train selected models
        all_results = {}
        
        for model_id in selected_models:
            model_config = MODEL_CONFIGS[model_id]
            model_name = model_config['name']
            
            try:
                results = self.train_single_model(model_id, train_loader, val_loader)
                all_results[model_name] = results
                
            except Exception as e:
                self.logger.error(f"Training failed for {model_config['display_name']}: {e}")
                print(f"\n❌ Training failed for {model_config['display_name']}: {e}")
                continue
        
        # Generate comparison visualizations and reports
        if all_results:
            self.logger.info("Generating comparison reports and visualizations...")
            
            # Export results
            self.visualizer.export_metrics_csv(all_results)
            self.visualizer.export_metrics_json(all_results)
            
            # Create comparison plots
            self.visualizer.create_model_comparison_plot(all_results, save_600dpi=True)
            
            # Generate summary report
            self._generate_summary_report(all_results)
            
            self.logger.info("Comparison complete! Check the 'results' directory for outputs.")
            print(f"\n✅ Comparison complete! Results saved to: {self.visualizer.results_dir}")
            print("\nGenerated files:")
            print("  📊 Training curves (300 DPI and 600 DPI)")
            print("  📈 Confusion matrices (300 DPI and 600 DPI)")
            print("  📊 Model comparison plots (300 DPI and 600 DPI)")
            print("  📄 Results CSV and JSON files")
            print("  📋 Summary report")
        
        return all_results
    
    def _generate_summary_report(self, all_results: Dict[str, Any]):
        """Generate a comprehensive summary report."""
        
        report_path = self.visualizer.results_dir / "comparison_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("UTKFACE MULTI-MODEL SEGMENTATION COMPARISON REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            for key, value in self.config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("MODEL PERFORMANCE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            # Create summary table
            headers = ["Model", "mIoU", "Race Acc", "Seg Acc", "Params (M)", "Time (ms)"]
            f.write(f"{'Model':<35} {'mIoU':<8} {'Race Acc':<10} {'Seg Acc':<10} {'Params(M)':<12} {'Time(ms)':<10}\n")
            f.write("-" * 85 + "\n")
            
            for model_name, results in all_results.items():
                if results['val_metrics']:
                    final_metrics = results['val_metrics'][-1]
                    miou = final_metrics.get('segmentation_miou', 0)
                    race_acc = final_metrics.get('race_accuracy', 0)
                    seg_acc = final_metrics.get('segmentation_accuracy', 0)
                    params = results.get('model_parameters', 0) / 1e6  # Convert to millions
                    inference_time = results.get('inference_time_ms', 0)
                    
                    f.write(f"{results['display_name']:<35} {miou:<8.4f} {race_acc:<10.4f} "
                           f"{seg_acc:<10.4f} {params:<12.2f} {inference_time:<10.2f}\n")
            
            f.write("\n" + "="*80 + "\n")

def main():
    """Main entry point with command line argument support."""
    
    parser = argparse.ArgumentParser(description='Enhanced UTKFace Multi-Model Segmentation Comparison')
    parser.add_argument('--dataset-dir', type=str, required=False, 
                       help='Path to UTKFace dataset directory')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=4e-5, 
                       help='Learning rate')
    parser.add_argument('--models', type=str, 
                       help='Comma-separated model numbers (e.g., 1,2,6)')
    parser.add_argument('--config', type=str, 
                       help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Update config with command line arguments
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Initialize application
    app = UTKFaceSegmentationComparison(config)
    
    # Select models
    if args.models:
        try:
            selected_models = [int(x.strip()) for x in args.models.split(',')]
        except ValueError:
            print("❌ Invalid model specification. Use comma-separated numbers.")
            return
    else:
        selected_models = app.select_models_interactive()
    
    # Get dataset directory
    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        dataset_dir = input("\nEnter path to UTKFace dataset directory: ").strip()
    
    if not dataset_dir:
        print("❌ Dataset directory is required.")
        return
    
    # Run comparison
    try:
        results = app.run_comparison(selected_models, dataset_dir)
        
        if results:
            print(f"\n🎉 Successfully completed comparison of {len(results)} models!")
        else:
            print("\n⚠️ No models were successfully trained.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()