#!/usr/bin/env python3
"""
Example Usage Script for Enhanced UTKFace Multi-Model Segmentation Framework

This script demonstrates how to use the framework in various scenarios.
"""

import os
import json
import torch
from enhanced_utkface_segmentation_comparison import (
    UTKFaceSegmentationComparison, 
    EnhancedFaceSegmentationWithRaceNet,
    DEFAULT_CONFIG
)

def example_interactive_usage():
    """Example 1: Interactive model selection and training."""
    print("="*60)
    print("EXAMPLE 1: Interactive Usage")
    print("="*60)
    
    # Initialize framework with default configuration
    app = UTKFaceSegmentationComparison(DEFAULT_CONFIG)
    
    # This would normally show interactive interface
    print("In interactive mode, you would see:")
    print("- Model selection menu")
    print("- Dataset directory prompt")  
    print("- Training progress bars")
    print("- Automatic visualization generation")

def example_programmatic_usage():
    """Example 2: Programmatic usage without interaction."""
    print("="*60)
    print("EXAMPLE 2: Programmatic Usage")
    print("="*60)
    
    # Custom configuration
    custom_config = DEFAULT_CONFIG.copy()
    custom_config.update({
        'batch_size': 8,
        'num_epochs': 20,
        'learning_rate': 2e-5,
        'mixed_precision': True
    })
    
    print("Custom configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")
    
    # Initialize framework
    app = UTKFaceSegmentationComparison(custom_config)
    
    print("\nAvailable models:")
    from enhanced_utkface_segmentation_comparison import MODEL_CONFIGS
    for model_id, config in MODEL_CONFIGS.items():
        print(f"  {model_id}. {config['display_name']}")

def example_single_model_testing():
    """Example 3: Testing a single model."""
    print("="*60)
    print("EXAMPLE 3: Single Model Testing")
    print("="*60)
    
    # Create the enhanced model
    model = EnhancedFaceSegmentationWithRaceNet(
        num_segmentation_classes=2,
        num_race_classes=5
    )
    
    print(f"Model: EnhancedFaceSegmentationWithRaceNet")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(2, 3, 256, 256)  # Batch of 2 images
    
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"\nModel outputs:")
    for key, tensor in outputs.items():
        print(f"  {key}: {tuple(tensor.shape)}")

def example_dataset_validation():
    """Example 4: Dataset validation workflow."""
    print("="*60)
    print("EXAMPLE 4: Dataset Validation")
    print("="*60)
    
    print("For dataset validation, you would typically:")
    print("1. Run dataset structure validation:")
    print("   python dataset_utils.py /path/to/utkface --validate")
    print()
    print("2. Generate comprehensive report:")
    print("   python dataset_utils.py /path/to/utkface --generate-report")
    print()
    print("3. Clean masks if needed:")
    print("   python dataset_utils.py /path/to/utkface --clean-masks")

def example_configuration_management():
    """Example 5: Configuration file management."""
    print("="*60)
    print("EXAMPLE 5: Configuration Management")
    print("="*60)
    
    # Create custom configuration
    config = {
        "batch_size": 16,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 5e-4,
        "patience": 15,
        "image_size": [320, 320],
        "mixed_precision": True,
        "gradient_clip_value": 2.0
    }
    
    # Save to file
    config_file = "custom_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created configuration file: {config_file}")
    print("Usage:")
    print(f"  python enhanced_utkface_segmentation_comparison.py --config {config_file}")
    
    # Clean up
    if os.path.exists(config_file):
        os.remove(config_file)

def example_command_line_usage():
    """Example 6: Command line usage patterns."""
    print("="*60)
    print("EXAMPLE 6: Command Line Usage Examples")
    print("="*60)
    
    examples = [
        {
            "description": "Train all models with default settings",
            "command": "python enhanced_utkface_segmentation_comparison.py --dataset-dir /path/to/utkface --models all"
        },
        {
            "description": "Train only the enhanced model with custom settings",
            "command": "python enhanced_utkface_segmentation_comparison.py --dataset-dir /path/to/utkface --models 6 --batch-size 8 --epochs 30"
        },
        {
            "description": "Compare UNet vs Enhanced model",
            "command": "python enhanced_utkface_segmentation_comparison.py --dataset-dir /path/to/utkface --models 1,6 --lr 2e-5"
        },
        {
            "description": "Use configuration file",
            "command": "python enhanced_utkface_segmentation_comparison.py --dataset-dir /path/to/utkface --config config_sample.json"
        },
        {
            "description": "Validate dataset before training",
            "command": "python dataset_utils.py /path/to/utkface --validate --generate-report"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}:")
        print(f"   {example['command']}")
        print()

def main():
    """Run all examples."""
    print("Enhanced UTKFace Multi-Model Segmentation Framework")
    print("Usage Examples and Demonstrations")
    print()
    
    examples = [
        example_interactive_usage,
        example_programmatic_usage,
        example_single_model_testing,
        example_dataset_validation,
        example_configuration_management,
        example_command_line_usage
    ]
    
    for example_func in examples:
        example_func()
        print()
    
    print("="*60)
    print("QUICK START GUIDE")
    print("="*60)
    print("1. Organize your UTKFace dataset:")
    print("   dataset/")
    print("   ├── images/")
    print("   │   └── age_gender_race_*.jpg")
    print("   └── masks/")
    print("       └── age_gender_race_*_mask.png")
    print()
    print("2. Validate your dataset:")
    print("   python dataset_utils.py dataset/ --validate")
    print()
    print("3. Start training:")
    print("   python enhanced_utkface_segmentation_comparison.py")
    print()
    print("4. Check results in the 'results/' directory")
    print()
    print("For detailed documentation, see README.md")

if __name__ == "__main__":
    main()