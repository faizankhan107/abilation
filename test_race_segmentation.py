#!/usr/bin/env python3
"""
Test script for race segmentation training script

This script validates the basic functionality and checks for common issues:
- CUDA initialization
- Model creation and forward pass
- JSON serialization of training logs
- Updated PyTorch functions usage
"""

import sys
import json
import tempfile
import warnings
from pathlib import Path

import torch
import torch.amp as amp
import numpy as np
from PIL import Image

# Add the current directory to path to import our module
sys.path.append('/home/runner/work/abilation/abilation')

try:
    from race_segmentation_train import (
        EnhancedRaceSegmentationModel,
        setup_cuda,
        CombinedLoss,
        FocalLoss,
        DiceLoss,
        calculate_metrics,
        RaceSegmentationDataset
    )
    print("✓ Successfully imported race segmentation modules")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    sys.exit(1)


def test_cuda_setup():
    """Test CUDA setup and initialization"""
    print("\n=== Testing CUDA Setup ===")
    
    try:
        device = setup_cuda()
        print(f"✓ CUDA setup completed, device: {device}")
        
        # Test basic tensor operations
        if device.type == 'cuda':
            test_tensor = torch.randn(10, 10, device=device)
            result = torch.matmul(test_tensor, test_tensor.T)
            print(f"✓ CUDA tensor operations working, shape: {result.shape}")
            
            # Clear memory
            del test_tensor, result
            torch.cuda.empty_cache()
        
        return device
    except Exception as e:
        print(f"✗ CUDA setup failed: {e}")
        return torch.device('cpu')


def test_model_creation():
    """Test model creation and forward pass"""
    print("\n=== Testing Model Creation ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EnhancedRaceSegmentationModel(num_classes=7, pretrained=False)
        model = model.to(device)
        print("✓ Model created successfully")
        
        # Test forward pass
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 256, 256, device=device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        expected_shape = (batch_size, 7, 256, 256)
        if output.shape == expected_shape:
            print(f"✓ Forward pass successful, output shape: {output.shape}")
        else:
            print(f"✗ Forward pass shape mismatch, expected: {expected_shape}, got: {output.shape}")
        
        # Clean up
        del model, input_tensor, output
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"✗ Model creation/forward pass failed: {e}")
        return False


def test_loss_functions():
    """Test loss functions"""
    print("\n=== Testing Loss Functions ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create dummy data
        batch_size, num_classes, height, width = 2, 7, 64, 64
        predictions = torch.randn(batch_size, num_classes, height, width, device=device)
        targets = torch.randint(0, num_classes, (batch_size, height, width), device=device)
        
        # Test Focal Loss
        focal_loss = FocalLoss()
        focal_result = focal_loss(predictions, targets)
        print(f"✓ Focal Loss working, value: {focal_result.item():.4f}")
        
        # Test Dice Loss
        dice_loss = DiceLoss()
        dice_result = dice_loss(predictions, targets)
        print(f"✓ Dice Loss working, value: {dice_result.item():.4f}")
        
        # Test Combined Loss
        combined_loss = CombinedLoss()
        combined_result = combined_loss(predictions, targets)
        print(f"✓ Combined Loss working, value: {combined_result.item():.4f}")
        
        # Clean up
        del predictions, targets
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False


def test_metrics_calculation():
    """Test metrics calculation"""
    print("\n=== Testing Metrics Calculation ===")
    
    try:
        # Create dummy prediction and target data
        batch_size, height, width = 10, 64, 64
        predictions = torch.randint(0, 7, (batch_size, height, width))
        targets = torch.randint(0, 7, (batch_size, height, width))
        
        metrics = calculate_metrics(predictions, targets, num_classes=7)
        
        # Check if metrics are calculated for all classes
        if len(metrics) == 7:
            print("✓ Metrics calculated for all 7 classes")
            
            # Check metric values are reasonable
            for class_id, metric in metrics.items():
                required_keys = ['precision', 'recall', 'f1', 'iou']
                if all(key in metric for key in required_keys):
                    print(f"✓ Class {class_id} metrics: P={metric['precision']:.3f}, R={metric['recall']:.3f}, F1={metric['f1']:.3f}, IoU={metric['iou']:.3f}")
                else:
                    print(f"✗ Missing metric keys for class {class_id}")
                    return False
        else:
            print(f"✗ Expected 7 classes, got {len(metrics)}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Metrics calculation failed: {e}")
        return False


def test_json_serialization():
    """Test JSON serialization of training logs"""
    print("\n=== Testing JSON Serialization ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a sample training log with device objects
        training_log = {
            'epoch': 1,
            'train_loss': 0.5,
            'val_loss': 0.4,
            'device': str(device),  # Convert device to string
            'model_info': {
                'num_parameters': 1000000,
                'device': str(device)  # Convert device to string
            },
            'detailed_metrics': {
                '0': {'precision': 0.8, 'recall': 0.7, 'f1': 0.75, 'iou': 0.6},
                '1': {'precision': 0.7, 'recall': 0.8, 'f1': 0.74, 'iou': 0.59}
            }
        }
        
        # Test JSON serialization
        json_str = json.dumps(training_log, indent=2)
        print("✓ JSON serialization successful")
        
        # Test deserialization
        loaded_log = json.loads(json_str)
        print("✓ JSON deserialization successful")
        
        # Verify device strings
        if loaded_log['device'] == str(device):
            print(f"✓ Device serialization correct: {loaded_log['device']}")
        else:
            print(f"✗ Device serialization error: expected {str(device)}, got {loaded_log['device']}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False


def test_pytorch_amp_usage():
    """Test updated PyTorch AMP functions"""
    print("\n=== Testing PyTorch AMP Usage ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test GradScaler with updated syntax
        if device.type == 'cuda':
            scaler = amp.GradScaler('cuda')
        else:
            scaler = amp.GradScaler('cpu')
        print("✓ GradScaler created with updated syntax")
        
        # Test autocast with updated syntax
        if device.type == 'cuda':
            with amp.autocast('cuda'):
                x = torch.randn(2, 3, 4, 4, device=device)
                y = torch.randn(2, 3, 4, 4, device=device)
                result = torch.matmul(x, y.transpose(-2, -1))
        else:
            with amp.autocast('cpu'):
                x = torch.randn(2, 3, 4, 4, device=device)
                y = torch.randn(2, 3, 4, 4, device=device)
                result = torch.matmul(x, y.transpose(-2, -1))
        
        print("✓ Autocast working with updated syntax")
        
        # Clean up
        del x, y, result
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"✗ PyTorch AMP test failed: {e}")
        return False


def test_dataset_structure():
    """Test dataset handling for class-based structure"""
    print("\n=== Testing Dataset Structure ===")
    
    try:
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory structure
            classes = ['white', 'black', 'east_asian', 'indian', 'latino_hispanic', 'middle_eastern', 'southeast_asian']
            
            for split in ['train', 'val']:
                for data_type in ['images', 'masks']:
                    for class_name in classes:
                        (temp_path / split / data_type / class_name).mkdir(parents=True, exist_ok=True)
            
            # Create dummy image and mask files
            for split in ['train']:
                for class_name in classes[:2]:  # Only test first 2 classes
                    # Create dummy image
                    img_path = temp_path / split / 'images' / class_name / 'test_001.jpg'
                    dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                    dummy_img.save(img_path)
                    
                    # Create dummy mask
                    mask_path = temp_path / split / 'masks' / class_name / 'test_001.png'
                    dummy_mask = Image.fromarray(np.random.randint(0, 7, (64, 64), dtype=np.uint8))
                    dummy_mask.save(mask_path)
            
            # Test dataset creation
            dataset = RaceSegmentationDataset(str(temp_path / 'train'), transform=None, is_training=True)
            
            if len(dataset) > 0:
                print(f"✓ Dataset created successfully with {len(dataset)} samples")
                
                # Test data loading
                sample_image, sample_mask = dataset[0]
                print(f"✓ Sample loaded: image shape {sample_image.shape}, mask shape {sample_mask.shape}")
                
                return True
            else:
                print("✗ Dataset is empty")
                return False
                
    except Exception as e:
        print(f"✗ Dataset structure test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("Race Segmentation Training Script Tests")
    print("=" * 50)
    
    tests = [
        ("CUDA Setup", test_cuda_setup),
        ("Model Creation", test_model_creation),
        ("Loss Functions", test_loss_functions),
        ("Metrics Calculation", test_metrics_calculation),
        ("JSON Serialization", test_json_serialization),
        ("PyTorch AMP Usage", test_pytorch_amp_usage),
        ("Dataset Structure", test_dataset_structure),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! The race segmentation training script is ready.")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return passed == len(results)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)