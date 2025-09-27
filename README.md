# Race Segmentation Training Script

A complete, production-ready training pipeline for race segmentation that addresses common issues with CUDA initialization, JSON serialization, deprecated PyTorch functions, and memory management.

## 🚀 Features

### Fixed Issues
- ✅ **CUDA Error**: Fixed `CUBLAS_STATUS_NOT_INITIALIZED` with proper CUDA initialization
- ✅ **JSON Serialization**: Converts `torch.device` objects to strings before serialization
- ✅ **Deprecated Functions**: Updated to use `torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda')`
- ✅ **Memory Management**: Implemented proper CUDA memory clearing and device management

### Training Pipeline
- 🏗️ **Enhanced Model**: Multi-scale features with ASPP (Atrous Spatial Pyramid Pooling)
- 📊 **Complete Metrics**: Per-class Precision, Recall, F1, and IoU statistics
- 🎯 **Advanced Loss**: Combined Focal Loss + Dice Loss for better segmentation
- 💾 **Checkpointing**: Model saving with best IoU tracking and early stopping
- 🔄 **Mixed Precision**: Efficient training with automatic mixed precision
- 📈 **Professional Logging**: Clean progress bars and detailed training statistics

### Dataset Support
- 📁 **Class-based Structure**: Supports 7 race classes in organized directories
- 🎨 **Data Augmentation**: Optional Albumentations support for better training
- 🏷️ **Race Classes**: white, black, east_asian, indian, latino_hispanic, middle_eastern, southeast_asian

## 📦 Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Core Dependencies
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- OpenCV Python >= 4.7.0
- Pillow >= 9.5.0
- NumPy >= 1.21.0
- tqdm >= 4.64.0

### Optional (for better data augmentation)
```bash
pip install albumentations>=1.3.0
```

## 🗂️ Dataset Structure

The training script expects the following directory structure:

```
dataset/
├── train/
│   ├── images/
│   │   ├── white/
│   │   ├── black/
│   │   ├── east_asian/
│   │   ├── indian/
│   │   ├── latino_hispanic/
│   │   ├── middle_eastern/
│   │   └── southeast_asian/
│   └── masks/
│       ├── white/
│       ├── black/
│       ├── east_asian/
│       ├── indian/
│       ├── latino_hispanic/
│       ├── middle_eastern/
│       └── southeast_asian/
└── val/
    ├── images/
    │   └── (same structure as train)
    └── masks/
        └── (same structure as train)
```

### Create Dataset Structure
```bash
python run_training.py --create-structure --dataset-dir ./my_dataset
```

## 🚀 Usage

### Basic Training
```bash
python race_segmentation_train.py \
    --data_dir ./dataset \
    --checkpoint_dir ./checkpoints \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4
```

### Full Parameter List
```bash
python race_segmentation_train.py \
    --data_dir ./dataset \              # Path to dataset directory
    --checkpoint_dir ./checkpoints \    # Directory to save checkpoints
    --epochs 50 \                       # Number of training epochs
    --batch_size 8 \                    # Batch size
    --learning_rate 1e-4 \              # Learning rate
    --num_workers 4 \                   # Number of data loader workers
    --resume ./checkpoints/best_model.pth  # Resume from checkpoint (optional)
```

### Using the Helper Script
```bash
# Create dataset structure
python run_training.py --create-structure

# Run training (modify paths in script)
python run_training.py
```

## 🧪 Testing

Run the comprehensive test suite to validate all functionality:

```bash
python test_race_segmentation.py
```

The test suite validates:
- CUDA setup and initialization
- Model creation and forward pass
- Loss function implementations
- Metrics calculation
- JSON serialization handling
- PyTorch AMP usage
- Dataset structure handling

## 📊 Training Output

The script produces professional training output with detailed metrics:

```
Epoch 50/50
Learning rate: 0.000001
Train Epoch 50: 100%|██████████| 2441/2441 [35:41<00:00,  1.14it/s, loss=0.8262, acc=0.7789, iou=0.5913]
Val Epoch 50: 100%|██████████| 309/309 [00:58<00:00,  5.28it/s, loss=0.3766, acc=0.7904, iou=0.3529]

============================================================
Epoch 50 Results:
           | Precision  | Recall     | F1         | IoU       
------------------------------------------------------------
Class 0    | 0.6414     | 0.6342     | 0.6116     | 0.4698    
Class 1    | 0.5791     | 0.6149     | 0.5686     | 0.4286    
Class 2    | 0.7602     | 0.7525     | 0.7338     | 0.6135    
Class 3    | 0.6552     | 0.6104     | 0.6111     | 0.4616    
Class 4    | 0.4672     | 0.4379     | 0.4177     | 0.2961    
Class 5    | 0.3931     | 0.4542     | 0.4001     | 0.2725    
Class 6    | 0.5093     | 0.4787     | 0.4638     | 0.3328    
------------------------------------------------------------
Mean IoU: 0.4107
Overall Accuracy: 0.5804
============================================================
```

## 🏗️ Model Architecture

### EnhancedRaceSegmentationModel
- **Backbone**: ResNet50 with pretrained weights
- **Feature Extraction**: Atrous Spatial Pyramid Pooling (ASPP)
- **Multi-scale Features**: Dilated convolutions with rates [6, 12, 18]
- **Global Context**: Adaptive average pooling
- **Decoder**: Progressive upsampling with skip connections

### Loss Function
- **Combined Loss**: Focal Loss + Dice Loss
- **Focal Loss**: Addresses class imbalance (α=1, γ=2)
- **Dice Loss**: Optimizes segmentation overlap
- **Weighting**: Balanced combination for optimal performance

## 🔧 Key Fixes Implemented

### 1. CUDA Issues
```python
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
        
        return device
    except Exception as e:
        logger.error(f"CUDA initialization failed: {e}")
        return torch.device('cpu')
```

### 2. JSON Serialization
```python
# Convert device objects to strings for JSON serialization
state = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'device': str(next(model.parameters()).device),  # Convert device to string
    'detailed_metrics': {str(k): v for k, v in detailed_metrics.items()}  # Convert keys to strings
}
```

### 3. Updated PyTorch Functions
```python
# Updated imports
import torch.amp as amp  # Instead of torch.cuda.amp

# Usage
if device.type == 'cuda':
    scaler = amp.GradScaler('cuda')
    with amp.autocast('cuda'):
        outputs = model(images)
else:
    scaler = amp.GradScaler('cpu')
    with amp.autocast('cpu'):
        outputs = model(images)
```

## 📁 File Structure

```
├── race_segmentation_train.py    # Main training script
├── test_race_segmentation.py     # Comprehensive test suite
├── run_training.py               # Helper script with examples
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── format_race_segmentation_results.py  # Original results formatter
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 4`
   - Use gradient accumulation
   - Enable CPU training: CUDA will automatically fallback

2. **Dataset Not Found**
   - Check dataset structure matches expected format
   - Use `python run_training.py --create-structure` to create template

3. **Import Errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility (>= 3.8)

### Memory Optimization
- Model uses efficient ResNet50 backbone
- Mixed precision training reduces memory usage
- Automatic CUDA memory clearing between epochs
- Gradient accumulation support for large effective batch sizes

## 🎯 Performance Tips

1. **For GPU Training**:
   - Use batch size 8-16 depending on GPU memory
   - Enable mixed precision (default)
   - Use multiple workers for data loading

2. **For CPU Training**:
   - Use smaller batch size (2-4)
   - Reduce number of workers
   - Consider using pretrained backbone

3. **For Better Results**:
   - Use data augmentation (Albumentations)
   - Implement learning rate scheduling (included)
   - Use early stopping based on validation IoU

## 📄 License

This implementation addresses the specific issues mentioned in the problem statement and provides a complete, robust training pipeline for race segmentation tasks.