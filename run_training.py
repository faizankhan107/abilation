#!/usr/bin/env python3
"""
Sample script to run race segmentation training

This script demonstrates how to use the race_segmentation_train.py script
with proper arguments and dataset structure.
"""

import os
import sys
import subprocess
from pathlib import Path


def create_sample_dataset_structure(base_dir):
    """Create sample dataset directory structure"""
    base_path = Path(base_dir)
    
    # Race classes
    classes = [
        'white', 'black', 'east_asian', 'indian', 
        'latino_hispanic', 'middle_eastern', 'southeast_asian'
    ]
    
    # Create directory structure
    for split in ['train', 'val']:
        for data_type in ['images', 'masks']:
            for class_name in classes:
                (base_path / split / data_type / class_name).mkdir(parents=True, exist_ok=True)
    
    print(f"Created dataset directory structure at: {base_path}")
    print("\nExpected structure:")
    print("dataset/")
    print("├── train/")
    print("│   ├── images/")
    print("│   │   ├── white/")
    print("│   │   ├── black/")
    print("│   │   ├── east_asian/")
    print("│   │   ├── indian/")
    print("│   │   ├── latino_hispanic/")
    print("│   │   ├── middle_eastern/")
    print("│   │   └── southeast_asian/")
    print("│   └── masks/")
    print("│       ├── white/")
    print("│       ├── black/")
    print("│       ├── east_asian/")
    print("│       ├── indian/")
    print("│       ├── latino_hispanic/")
    print("│       ├── middle_eastern/")
    print("│       └── southeast_asian/")
    print("└── val/")
    print("    ├── images/")
    print("    │   └── (same class structure)")
    print("    └── masks/")
    print("        └── (same class structure)")


def run_training_example():
    """Run training with example parameters"""
    
    # Example dataset path (you need to replace this with your actual dataset path)
    dataset_path = "./dataset"  # Replace with your actual dataset path
    checkpoint_dir = "./checkpoints"
    
    # Training parameters
    params = {
        "data_dir": dataset_path,
        "checkpoint_dir": checkpoint_dir,
        "epochs": 50,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "num_workers": 4
    }
    
    # Build command
    cmd = [
        sys.executable, "race_segmentation_train.py",
        "--data_dir", params["data_dir"],
        "--checkpoint_dir", params["checkpoint_dir"],
        "--epochs", str(params["epochs"]),
        "--batch_size", str(params["batch_size"]),
        "--learning_rate", str(params["learning_rate"]),
        "--num_workers", str(params["num_workers"])
    ]
    
    print("Training command:")
    print(" ".join(cmd))
    print("\nTo run training manually, use:")
    print(f"python race_segmentation_train.py \\")
    print(f"    --data_dir {params['data_dir']} \\")
    print(f"    --checkpoint_dir {params['checkpoint_dir']} \\")
    print(f"    --epochs {params['epochs']} \\")
    print(f"    --batch_size {params['batch_size']} \\")
    print(f"    --learning_rate {params['learning_rate']} \\")
    print(f"    --num_workers {params['num_workers']}")
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"\n⚠️  Dataset directory not found: {dataset_path}")
        print("Please prepare your dataset or run with --create-structure to create sample structure")
        return False
    
    # Check if required files exist
    train_dir = Path(dataset_path) / "train"
    val_dir = Path(dataset_path) / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"\n⚠️  Training or validation directory not found")
        print(f"Please ensure {train_dir} and {val_dir} exist with proper structure")
        return False
    
    print(f"\n✓ Dataset structure looks good!")
    print(f"✓ Ready to start training...")
    
    # Uncomment the line below to actually run training
    # subprocess.run(cmd)
    
    return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run race segmentation training')
    parser.add_argument('--create-structure', action='store_true', 
                       help='Create sample dataset directory structure')
    parser.add_argument('--dataset-dir', type=str, default='./dataset',
                       help='Dataset directory path')
    
    args = parser.parse_args()
    
    if args.create_structure:
        create_sample_dataset_structure(args.dataset_dir)
        print(f"\n📁 Sample dataset structure created at: {args.dataset_dir}")
        print("Now you need to:")
        print("1. Add your actual images to the images/ subdirectories")
        print("2. Add corresponding segmentation masks to the masks/ subdirectories")
        print("3. Run training with: python run_training.py")
    else:
        success = run_training_example()
        if not success:
            print("\n💡 Tip: Run with --create-structure to create sample dataset structure")


if __name__ == '__main__':
    main()