# Enhanced UTKFace Multi-Model Segmentation Comparison Framework

A comprehensive framework for comparing facial race segmentation models using the UTKFace dataset. This framework provides production-ready training, evaluation, and visualization capabilities for multiple state-of-the-art segmentation architectures.

## 🚀 Features

- **6 State-of-the-Art Models**: UNet, DeepLabV3+, PSPNet, HRNetV2, FaceSeg+, and the new **EnhancedFaceSegmentationWithRaceNet**
- **UTKFace Dataset Support**: Complete processing pipeline for 5 race classes (White, Black, Asian, Indian, Others)
- **Interactive Interface**: User-friendly model selection and configuration
- **Production Features**:
  - Mixed precision training with gradient scaling
  - Early stopping and model checkpointing
  - Multi-GPU support with DataParallel
  - Comprehensive logging and progress tracking
- **Advanced Visualization**: High-quality plots (300 DPI and 600 DPI) with training curves, confusion matrices, and model comparisons
- **Comprehensive Export**: Results in CSV, JSON, and detailed summary reports

## 📋 Requirements

```bash
# Install dependencies
pip install -r requirements.txt
```

### Required Packages
- torch>=1.9.0
- torchvision>=0.10.0
- opencv-python>=4.5.0
- numpy>=1.21.0
- Pillow>=8.3.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=1.0.0
- pandas>=1.3.0
- scipy>=1.7.0
- tqdm>=4.62.0

## 📁 Dataset Structure

Organize your UTKFace dataset as follows:

```
dataset_dir/
├── images/
│   ├── 20_1_0_20170109142408075.jpg  # age_gender_race_date.jpg
│   ├── 25_0_2_20170116174525125.jpg
│   └── ...
└── masks/
    ├── 20_1_0_20170109142408075_mask.png  # corresponding mask files
    ├── 25_0_2_20170116174525125_mask.png
    └── ...
```

### Race Class Mappings
- **0**: White - Color: (255, 250, 250) BGR
- **1**: Black - Color: (139, 69, 19) BGR  
- **2**: Asian - Color: (255, 215, 0) BGR
- **3**: Indian - Color: (255, 153, 51) BGR
- **4**: Others - Color: (152, 251, 152) BGR

## 🎯 Usage

### Interactive Mode
```bash
python enhanced_utkface_segmentation_comparison.py
```

### Command Line Mode
```bash
# Train specific models
python enhanced_utkface_segmentation_comparison.py \
    --dataset-dir /path/to/utkface \
    --models 1,2,6 \
    --batch-size 16 \
    --epochs 50 \
    --lr 4e-5

# Use configuration file
python enhanced_utkface_segmentation_comparison.py \
    --dataset-dir /path/to/utkface \
    --config config_sample.json \
    --models all
```

### Configuration File
Create a JSON configuration file (see `config_sample.json`):

```json
{
  "batch_size": 16,
  "num_epochs": 50,
  "learning_rate": 4e-5,
  "weight_decay": 1e-4,
  "patience": 10,
  "image_size": [256, 256],
  "mixed_precision": true,
  "gradient_clip_value": 1.0
}
```

## 🏗️ Model Architectures

### 1. U-Net
Classic encoder-decoder architecture with skip connections for pixel-level segmentation.

### 2. DeepLabV3+
Advanced segmentation model with atrous convolutions and decoder refinement.

### 3. PSPNet
Pyramid Scene Parsing Network with multi-scale context aggregation.

### 4. HRNetV2
High-resolution network maintaining high-resolution representations throughout.

### 5. FaceSeg+
Specialized face segmentation architecture with attention mechanisms.

### 6. EnhancedFaceSegmentationWithRaceNet ⭐ NEW
Advanced architecture featuring:
- **Spatial Attention Mechanisms**: Enhanced feature focus
- **Adaptive Feature Fusion**: Multi-scale feature integration
- **Boundary Enhancement Processing**: Improved segmentation edges
- **Progressive Refinement Heads**: Multi-stage prediction refinement
- **Pyramid Pooling Module**: Multi-scale context extraction

## 📊 Output and Results

The framework generates comprehensive results in the `results/` directory:

### Visualizations (300 DPI & 600 DPI)
- **Training Curves**: Loss, accuracy, and mIoU progression
- **Confusion Matrices**: Race classification performance
- **Model Comparison Plots**: Side-by-side performance metrics

### Data Export
- **CSV Files**: Detailed metrics for each model
- **JSON Files**: Complete training logs and configurations
- **Summary Reports**: Human-readable performance summaries

### Metrics Tracked
- **Segmentation**: mIoU, per-class IoU, pixel accuracy
- **Race Classification**: Accuracy, precision, recall, F1-score per class
- **Performance**: Training time, inference speed, model parameters
- **Hardware**: GPU utilization, memory usage

## 🔧 Advanced Features

### Multi-GPU Training
Automatically detected and enabled when multiple GPUs are available:
```python
# Automatically enables DataParallel for multi-GPU setups
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### Mixed Precision Training
Accelerated training with automatic mixed precision:
```python
# Enabled via configuration
"mixed_precision": true
```

### Early Stopping
Prevents overfitting with configurable patience:
```python
# Stops training when validation loss stops improving
"patience": 10,
"min_delta": 1e-4
```

### Model Checkpointing
Automatic saving of best models and regular checkpoints:
- `{ModelName}_best_model.pth`: Best validation performance
- `{ModelName}_epoch_{N}.pth`: Regular epoch checkpoints

## 🎨 Customization

### Adding New Models
1. Implement the model class inheriting from `nn.Module`
2. Ensure the forward method returns the expected dictionary format
3. Add to the model registry in `UTKFaceSegmentationComparison`

### Custom Loss Functions
Modify the trainer's loss functions for specific requirements:
```python
# In SegmentationTrainer.__init__()
self.seg_criterion = nn.CrossEntropyLoss(ignore_index=-1)
self.race_criterion = nn.CrossEntropyLoss()
```

### Data Augmentation
Customize transforms in `create_utkface_transforms()`:
```python
# Add custom augmentations for training
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # Add your custom transforms here
])
```

## 📈 Performance Tips

### Optimizing Training Speed
- Use mixed precision training (`mixed_precision: true`)
- Increase batch size based on GPU memory
- Use multiple workers for data loading (`num_workers: 4-8`)
- Enable pin memory (`pin_memory: true`)

### Memory Optimization
- Reduce batch size if encountering OOM errors
- Use gradient checkpointing for very deep models
- Clear cache between model trainings: `torch.cuda.empty_cache()`

### Improving Results
- Experiment with different learning rates (1e-5 to 1e-4 typical range)
- Adjust patience and early stopping criteria
- Use learning rate scheduling (already included)
- Fine-tune data augmentation parameters

## 🐛 Troubleshooting

### Common Issues

**"Dataset directory not found"**
- Ensure the dataset path is correct
- Verify the directory contains `images/` and `masks/` subdirectories

**"CUDA out of memory"**
- Reduce batch size (`--batch-size 8` or smaller)
- Reduce image size in configuration
- Close other GPU applications

**"No module named 'torch'"**
- Install required dependencies: `pip install -r requirements.txt`
- Ensure you're using the correct Python environment

**Poor model performance**
- Check dataset quality and mask accuracy
- Verify race class color mappings are correct
- Increase training epochs or adjust learning rate
- Ensure proper train/validation split

## 📝 License

This framework is provided for research and educational purposes. Please ensure compliance with UTKFace dataset licensing terms when using this code.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional model architectures
- Enhanced data augmentation strategies
- Advanced evaluation metrics
- Performance optimizations
- Documentation improvements

## 📚 References

1. UTKFace Dataset: [Large Scale Face Dataset](https://susanqq.github.io/UTKFace/)
2. U-Net: [Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
3. DeepLabV3+: [Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611)
4. PSPNet: [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
5. HRNet: [Deep High-Resolution Representation Learning](https://arxiv.org/abs/1908.07919)

---

**Framework Version**: 1.0.0  
**Last Updated**: December 2024  
**Maintainer**: Enhanced UTKFace Segmentation Team