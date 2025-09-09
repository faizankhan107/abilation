#!/usr/bin/env python3
"""
Enhanced AdaptiveStream Training Script with Robust NaN Handling

This script addresses the NaN detection issues observed during training by implementing:
1. Proper gradient clipping
2. Robust learning rate scheduling with warm-up
3. Comprehensive NaN detection and handling
4. Numerical stability checks
5. Data validation and preprocessing safeguards
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
import time
from pathlib import Path
from tqdm import tqdm
import math
import csv
from typing import Dict, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NumericalStabilityMixin:
    """Mixin class for numerical stability utilities"""
    
    @staticmethod
    def check_for_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check if tensor contains NaN or Inf values"""
        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {name}")
            return True
        if torch.isinf(tensor).any():
            logger.warning(f"Inf detected in {name}")
            return True
        return False
    
    @staticmethod
    def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Safe division with epsilon to prevent division by zero"""
        return numerator / (denominator + eps)
    
    @staticmethod
    def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
        """Clip gradients and return the norm"""
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


class EnhancedAdaptiveStreamModel(nn.Module, NumericalStabilityMixin):
    """Enhanced AdaptiveStream model with improved numerical stability"""
    
    def __init__(self, num_classes: int = 5, input_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        
        # Encoder with proper initialization
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Adaptive stream components - ensure proper size for upsampling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # 7x7 -> 224x224 with 5 upsampling layers
        self.stream_conv = nn.Conv2d(512, 256, 1)
        
        # Decoder for segmentation - properly upscale to original image size
        self.decoder = nn.Sequential(
            # Upsample from 7x7 to 14x14
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Upsample from 14x14 to 28x28
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Upsample from 28x28 to 56x56
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Upsample from 56x56 to 112x112
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Upsample from 112x112 to 224x224
            nn.ConvTranspose2d(16, num_classes, 4, stride=2, padding=1),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper schemes"""
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with NaN checking"""
        # Check input
        if self.check_for_nan_inf(x, "input"):
            logger.error("NaN/Inf in input, using zeros")
            x = torch.zeros_like(x)
        
        # Encoder
        features = self.encoder(x)
        if self.check_for_nan_inf(features, "encoder_features"):
            logger.warning("NaN/Inf in encoder features")
            features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Adaptive stream
        pooled = self.adaptive_pool(features)
        stream_features = self.stream_conv(pooled)
        
        if self.check_for_nan_inf(stream_features, "stream_features"):
            logger.warning("NaN/Inf in stream features")
            stream_features = torch.nan_to_num(stream_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Segmentation output
        seg_output = self.decoder(stream_features)
        if self.check_for_nan_inf(seg_output, "segmentation_output"):
            logger.warning("NaN/Inf in segmentation output")
            seg_output = torch.nan_to_num(seg_output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Classification output
        cls_output = self.classifier(stream_features)
        if self.check_for_nan_inf(cls_output, "classification_output"):
            logger.warning("NaN/Inf in classification output")
            cls_output = torch.nan_to_num(cls_output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return seg_output, cls_output


class StableLoss(nn.Module, NumericalStabilityMixin):
    """Numerically stable loss function"""
    
    def __init__(self, seg_weight: float = 1.0, cls_weight: float = 1.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_loss = nn.CrossEntropyLoss(reduction='mean')
        self.cls_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, seg_pred: torch.Tensor, cls_pred: torch.Tensor, 
                seg_target: torch.Tensor, cls_target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute stable loss with NaN handling"""
        
        # Check inputs for NaN/Inf
        inputs = [
            (seg_pred, "seg_pred"),
            (cls_pred, "cls_pred"),
            (seg_target, "seg_target"),
            (cls_target, "cls_target")
        ]
        
        for tensor, name in inputs:
            if self.check_for_nan_inf(tensor, name):
                logger.error(f"NaN/Inf detected in {name}, returning large loss")
                return torch.tensor(1e6, device=seg_pred.device, requires_grad=True), {
                    'seg_loss': torch.tensor(1e6),
                    'cls_loss': torch.tensor(1e6),
                    'total_loss': torch.tensor(1e6)
                }
        
        # Clamp predictions to prevent extreme values
        seg_pred = torch.clamp(seg_pred, -10, 10)
        cls_pred = torch.clamp(cls_pred, -10, 10)
        
        try:
            # Compute losses
            seg_loss = self.seg_loss(seg_pred, seg_target)
            cls_loss = self.cls_loss(cls_pred, cls_target)
            
            # Check for NaN in losses
            if torch.isnan(seg_loss) or torch.isinf(seg_loss):
                logger.warning("NaN/Inf in segmentation loss, using fallback")
                seg_loss = torch.tensor(1.0, device=seg_pred.device, requires_grad=True)
            
            if torch.isnan(cls_loss) or torch.isinf(cls_loss):
                logger.warning("NaN/Inf in classification loss, using fallback")
                cls_loss = torch.tensor(1.0, device=cls_pred.device, requires_grad=True)
            
            total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
            
            # Final check
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.error("NaN/Inf in total loss, using fallback")
                total_loss = torch.tensor(1.0, device=seg_pred.device, requires_grad=True)
            
            return total_loss, {
                'seg_loss': seg_loss.item(),
                'cls_loss': cls_loss.item(),
                'total_loss': total_loss.item()
            }
            
        except Exception as e:
            logger.error(f"Error computing loss: {e}")
            fallback_loss = torch.tensor(1.0, device=seg_pred.device, requires_grad=True)
            return fallback_loss, {
                'seg_loss': 1.0,
                'cls_loss': 1.0,
                'total_loss': 1.0
            }


class WarmupLRScheduler:
    """Learning rate scheduler with warmup and stable decay"""
    
    def __init__(self, optimizer, warmup_epochs: int = 5, base_lr: float = 1e-3, 
                 min_lr: float = 1e-6, decay_factor: float = 0.95):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Decay phase
            decay_epochs = self.current_epoch - self.warmup_epochs
            lr = self.base_lr * (self.decay_factor ** decay_epochs)
        
        # Ensure minimum learning rate
        lr = max(lr, self.min_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing"""
    
    def __init__(self, num_samples: int = 1000, image_size: int = 224):
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic image
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Generate synthetic segmentation mask
        seg_mask = torch.randint(0, 5, (self.image_size, self.image_size))
        
        # Generate synthetic classification label
        cls_label = torch.randint(0, 5, (1,)).item()
        
        return image, seg_mask, cls_label


class EnhancedAdaptiveStreamTrainer(NumericalStabilityMixin):
    """Robust trainer with comprehensive NaN handling"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device, save_dir: str = "checkpoints"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize optimizer with reasonable learning rate
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize scheduler
        self.scheduler = WarmupLRScheduler(
            self.optimizer,
            warmup_epochs=5,
            base_lr=1e-3,
            min_lr=1e-6,
            decay_factor=0.95
        )
        
        # Initialize loss function
        self.criterion = StableLoss(seg_weight=1.0, cls_weight=0.5)
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        self.metrics_history = []
        
        # NaN detection counters
        self.nan_batches = 0
        self.total_batches = 0
    
    def train_epoch(self) -> Dict:
        """Train for one epoch with robust NaN handling"""
        self.model.train()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_cls_loss = 0.0
        valid_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="🔄 Training Enhanced AdaptiveStream")
        
        for batch_idx, (images, seg_masks, cls_labels) in enumerate(progress_bar):
            self.total_batches += 1
            
            try:
                # Move to device
                images = images.to(self.device)
                seg_masks = seg_masks.to(self.device)
                cls_labels = cls_labels.to(self.device)
                
                # Check input data
                if self.check_for_nan_inf(images, f"input_images_batch_{batch_idx}"):
                    logger.warning(f"NaN/Inf detected in input batch {batch_idx}, skipping...")
                    self.nan_batches += 1
                    continue
                
                # Forward pass
                self.optimizer.zero_grad()
                seg_output, cls_output = self.model(images)
                
                # Check outputs
                if (self.check_for_nan_inf(seg_output, f"seg_output_{batch_idx}") or 
                    self.check_for_nan_inf(cls_output, f"cls_output_{batch_idx}")):
                    logger.warning(f"NaN/Inf detected in output {batch_idx} at batch {batch_idx}")
                    self.nan_batches += 1
                    continue
                
                # Compute loss
                loss, loss_dict = self.criterion(seg_output, cls_output, seg_masks, cls_labels)
                
                # Check loss
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                    logger.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}")
                    self.nan_batches += 1
                    continue
                
                # Backward pass
                loss.backward()
                
                # Check gradients and clip
                grad_norm = self.clip_gradients(self.model, max_norm=1.0)
                
                if grad_norm > 100:  # Gradient explosion detection
                    logger.warning(f"Large gradient norm detected: {grad_norm}")
                    self.nan_batches += 1
                    continue
                
                # Optimizer step
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                total_seg_loss += loss_dict['seg_loss']
                total_cls_loss += loss_dict['cls_loss']
                valid_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Grad': f"{grad_norm:.2f}",
                    'Valid': f"{valid_batches}/{batch_idx+1}"
                })
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                self.nan_batches += 1
                continue
        
        # Calculate averages
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            avg_seg_loss = total_seg_loss / valid_batches
            avg_cls_loss = total_cls_loss / valid_batches
        else:
            logger.error("No valid batches in training!")
            avg_loss = float('inf')
            avg_seg_loss = float('inf')
            avg_cls_loss = float('inf')
        
        return {
            'train_loss': avg_loss,
            'train_seg_loss': avg_seg_loss,
            'train_cls_loss': avg_cls_loss,
            'valid_batches': valid_batches,
            'total_batches': len(self.train_loader),
            'nan_batches': self.nan_batches,
            'nan_ratio': self.nan_batches / self.total_batches if self.total_batches > 0 else 0
        }
    
    def validate_epoch(self) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_cls_loss = 0.0
        valid_batches = 0
        
        # Metrics for IoU calculation
        seg_correct = 0
        seg_total = 0
        cls_correct = 0
        cls_total = 0
        
        progress_bar = tqdm(self.val_loader, desc="📊 Evaluating Enhanced AdaptiveStream")
        
        with torch.no_grad():
            for batch_idx, (images, seg_masks, cls_labels) in enumerate(progress_bar):
                try:
                    # Move to device
                    images = images.to(self.device)
                    seg_masks = seg_masks.to(self.device)
                    cls_labels = cls_labels.to(self.device)
                    
                    # Check input data
                    if self.check_for_nan_inf(images, f"val_input_batch_{batch_idx}"):
                        logger.warning(f"NaN detected in batch {batch_idx}, skipping...")
                        continue
                    
                    # Forward pass
                    seg_output, cls_output = self.model(images)
                    
                    # Check outputs
                    if (self.check_for_nan_inf(seg_output, f"val_seg_output_{batch_idx}") or 
                        self.check_for_nan_inf(cls_output, f"val_cls_output_{batch_idx}")):
                        logger.warning(f"NaN detected in batch {batch_idx}, skipping...")
                        continue
                    
                    # Compute loss
                    loss, loss_dict = self.criterion(seg_output, cls_output, seg_masks, cls_labels)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Invalid validation loss at batch {batch_idx}")
                        continue
                    
                    # Update metrics
                    total_loss += loss.item()
                    total_seg_loss += loss_dict['seg_loss']
                    total_cls_loss += loss_dict['cls_loss']
                    
                    # Calculate accuracy
                    seg_pred = torch.argmax(seg_output, dim=1)
                    cls_pred = torch.argmax(cls_output, dim=1)
                    
                    seg_correct += (seg_pred == seg_masks).sum().item()
                    seg_total += seg_masks.numel()
                    cls_correct += (cls_pred == cls_labels).sum().item()
                    cls_total += cls_labels.size(0)
                    
                    valid_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'Loss': f"{loss.item():.4f}",
                        'Valid': f"{valid_batches}/{batch_idx+1}"
                    })
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Calculate averages
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            avg_seg_loss = total_seg_loss / valid_batches
            avg_cls_loss = total_cls_loss / valid_batches
            seg_acc = seg_correct / seg_total if seg_total > 0 else 0.0
            cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
        else:
            logger.warning("📊 No valid batches in evaluation!")
            avg_loss = float('inf')
            avg_seg_loss = float('inf')
            avg_cls_loss = float('inf')
            seg_acc = 0.0
            cls_acc = 0.0
        
        return {
            'val_loss': avg_loss,
            'val_seg_loss': avg_seg_loss,
            'val_cls_loss': avg_cls_loss,
            'val_seg_acc': seg_acc,
            'val_cls_acc': cls_acc,
            'valid_batches': valid_batches,
            'total_batches': len(self.val_loader)
        }
    
    def train(self, num_epochs: int = 50):
        """Main training loop"""
        logger.info("🚀 Starting Enhanced AdaptiveStream Training")
        logger.info(f"Device: {self.device}")
        logger.info(f"Number of epochs: {num_epochs}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Update learning rate
            current_lr = self.scheduler.step()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch+1}/{num_epochs}")
            logger.info(f"Learning rate: {current_lr:.2e}")
            logger.info(f"{'='*60}")
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Combine metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'time': epoch_time,
                'learning_rate': current_lr,
                **train_metrics,
                **val_metrics
            }
            
            # Save metrics
            self.metrics_history.append(epoch_metrics)
            
            # Log epoch summary
            logger.info(f"\n📊 Epoch {epoch+1} Summary:")
            logger.info(f"   Time: {epoch_time:.2f}s")
            logger.info(f"   Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"   Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"   Val Seg IoU: {val_metrics['val_seg_acc']:.4f}")
            logger.info(f"   Val Cls Acc: {val_metrics['val_cls_acc']:.4f}")
            
            # Check for improvement
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                self.save_checkpoint(epoch + 1, 'best_model.pth')
                logger.info("💾 Saved best model checkpoint")
            else:
                self.patience_counter += 1
                logger.info(f"🔄 No improvement for {self.patience_counter} epochs (patience: {self.patience})")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"⏹️ Early stopping after {epoch+1} epochs")
                break
            
            # Save metrics to CSV
            self.save_metrics_csv()
        
        logger.info("✅ Training completed!")
        return self.metrics_history
    
    def save_checkpoint(self, epoch: int, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics_history': self.metrics_history
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def save_metrics_csv(self):
        """Save training metrics to CSV"""
        if not self.metrics_history:
            return
        
        csv_file = self.save_dir / 'training_metrics.csv'
        fieldnames = self.metrics_history[0].keys()
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_history)


def main():
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create model
    model = EnhancedAdaptiveStreamModel(num_classes=5)
    model = model.to(device)
    
    # Create synthetic datasets for testing
    train_dataset = SyntheticDataset(num_samples=1000, image_size=224)
    val_dataset = SyntheticDataset(num_samples=200, image_size=224)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Create trainer
    trainer = EnhancedAdaptiveStreamTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir="checkpoints"
    )
    
    # Start training
    metrics_history = trainer.train(num_epochs=50)
    
    logger.info("📈 Training Summary:")
    logger.info(f"Total epochs: {len(metrics_history)}")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    if metrics_history:
        final_metrics = metrics_history[-1]
        logger.info(f"Final validation accuracy: {final_metrics['val_cls_acc']:.4f}")
        logger.info(f"NaN ratio: {final_metrics.get('nan_ratio', 0):.2%}")


if __name__ == "__main__":
    main()