#!/usr/bin/env python3
"""
Quick test script to demonstrate the fixed training pipeline
"""

import torch
import logging
from train_enhanced_adaptivestream import (
    EnhancedAdaptiveStreamModel, 
    EnhancedAdaptiveStreamTrainer,
    SyntheticDataset
)
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Quick training demo"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = EnhancedAdaptiveStreamModel(num_classes=5)
    model = model.to(device)
    
    # Create small datasets for quick demo
    train_dataset = SyntheticDataset(num_samples=100, image_size=224)
    val_dataset = SyntheticDataset(num_samples=20, image_size=224)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Create trainer
    trainer = EnhancedAdaptiveStreamTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir="quick_test_checkpoints"
    )
    
    # Quick training - only 3 epochs to demonstrate it works
    logger.info("🚀 Starting quick training demo (3 epochs)...")
    metrics_history = trainer.train(num_epochs=3)
    
    # Print results
    logger.info("✅ Training completed successfully!")
    logger.info(f"Total epochs: {len(metrics_history)}")
    
    if metrics_history:
        final_metrics = metrics_history[-1]
        logger.info(f"Final train loss: {final_metrics['train_loss']:.4f}")
        logger.info(f"Final val loss: {final_metrics['val_loss']:.4f}")
        logger.info(f"Final val accuracy: {final_metrics['val_cls_acc']:.4f}")
        logger.info(f"NaN batches ratio: {final_metrics.get('nan_ratio', 0):.2%}")
        
        # Check if training was successful
        if final_metrics['nan_ratio'] < 0.1 and final_metrics['val_loss'] < 10:
            logger.info("🎉 SUCCESS: Training completed without NaN issues!")
        else:
            logger.warning("⚠️ Training had some issues")
    
    return metrics_history

if __name__ == "__main__":
    main()