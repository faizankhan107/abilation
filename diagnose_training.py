#!/usr/bin/env python3
"""
Diagnostic script to identify and fix NaN issues in training.
This script runs various tests to pinpoint the exact cause of NaN problems.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from train_enhanced_adaptivestream import (
    EnhancedAdaptiveStreamModel, 
    StableLoss, 
    SyntheticDataset,
    NumericalStabilityMixin
)
from config import TrainingConfig, DebugConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingDiagnostics(NumericalStabilityMixin):
    """Comprehensive diagnostics for training issues"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
    
    def test_model_initialization(self, model: nn.Module) -> dict:
        """Test if model weights are properly initialized"""
        logger.info("🔍 Testing model initialization...")
        
        results = {
            'nan_weights': False,
            'inf_weights': False,
            'zero_weights': False,
            'large_weights': False,
            'weight_stats': {}
        }
        
        for name, param in model.named_parameters():
            if param.data is not None:
                # Check for NaN/Inf
                if torch.isnan(param.data).any():
                    results['nan_weights'] = True
                    logger.error(f"❌ NaN weights found in {name}")
                
                if torch.isinf(param.data).any():
                    results['inf_weights'] = True
                    logger.error(f"❌ Inf weights found in {name}")
                
                # Check for all zeros
                if (param.data == 0).all():
                    results['zero_weights'] = True
                    logger.warning(f"⚠️ All zero weights in {name}")
                
                # Check for extremely large weights
                max_weight = param.data.abs().max().item()
                if max_weight > 100:
                    results['large_weights'] = True
                    logger.warning(f"⚠️ Large weights in {name}: max={max_weight:.2e}")
                
                # Collect statistics
                results['weight_stats'][name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item()
                }
        
        if not any([results['nan_weights'], results['inf_weights'], results['zero_weights']]):
            logger.info("✅ Model initialization looks good")
        
        return results
    
    def test_forward_pass(self, model: nn.Module, batch_size: int = 4) -> dict:
        """Test forward pass with synthetic data"""
        logger.info("🔍 Testing forward pass...")
        
        model.eval()
        results = {
            'forward_pass_success': False,
            'output_nan': False,
            'output_inf': False,
            'output_stats': {}
        }
        
        try:
            # Create synthetic input
            input_tensor = torch.randn(batch_size, 3, 224, 224, device=self.device)
            
            with torch.no_grad():
                seg_output, cls_output = model(input_tensor)
                
                # Check outputs
                if torch.isnan(seg_output).any():
                    results['output_nan'] = True
                    logger.error("❌ NaN in segmentation output")
                
                if torch.isnan(cls_output).any():
                    results['output_nan'] = True
                    logger.error("❌ NaN in classification output")
                
                if torch.isinf(seg_output).any():
                    results['output_inf'] = True
                    logger.error("❌ Inf in segmentation output")
                
                if torch.isinf(cls_output).any():
                    results['output_inf'] = True
                    logger.error("❌ Inf in classification output")
                
                # Collect output statistics
                results['output_stats']['seg'] = {
                    'shape': list(seg_output.shape),
                    'mean': seg_output.mean().item(),
                    'std': seg_output.std().item(),
                    'min': seg_output.min().item(),
                    'max': seg_output.max().item()
                }
                
                results['output_stats']['cls'] = {
                    'shape': list(cls_output.shape),
                    'mean': cls_output.mean().item(),
                    'std': cls_output.std().item(),
                    'min': cls_output.min().item(),
                    'max': cls_output.max().item()
                }
                
                results['forward_pass_success'] = True
                logger.info("✅ Forward pass successful")
                
        except Exception as e:
            logger.error(f"❌ Forward pass failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def test_loss_computation(self, model: nn.Module, batch_size: int = 4) -> dict:
        """Test loss computation"""
        logger.info("🔍 Testing loss computation...")
        
        model.eval()
        loss_fn = StableLoss()
        results = {
            'loss_computation_success': False,
            'loss_nan': False,
            'loss_inf': False,
            'loss_stats': {}
        }
        
        try:
            # Create synthetic data
            input_tensor = torch.randn(batch_size, 3, 224, 224, device=self.device)
            seg_target = torch.randint(0, 5, (batch_size, 224, 224), device=self.device)
            cls_target = torch.randint(0, 5, (batch_size,), device=self.device)
            
            with torch.no_grad():
                seg_output, cls_output = model(input_tensor)
                loss, loss_dict = loss_fn(seg_output, cls_output, seg_target, cls_target)
                
                # Check loss values
                if torch.isnan(loss):
                    results['loss_nan'] = True
                    logger.error("❌ NaN in total loss")
                
                if torch.isinf(loss):
                    results['loss_inf'] = True
                    logger.error("❌ Inf in total loss")
                
                # Collect loss statistics
                results['loss_stats'] = loss_dict
                results['loss_computation_success'] = True
                logger.info(f"✅ Loss computation successful: {loss.item():.4f}")
                
        except Exception as e:
            logger.error(f"❌ Loss computation failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def test_backward_pass(self, model: nn.Module, optimizer: optim.Optimizer, batch_size: int = 4) -> dict:
        """Test backward pass and gradient computation"""
        logger.info("🔍 Testing backward pass...")
        
        model.train()
        loss_fn = StableLoss()
        results = {
            'backward_pass_success': False,
            'grad_nan': False,
            'grad_inf': False,
            'grad_stats': {},
            'grad_norms': {}
        }
        
        try:
            # Create synthetic data
            input_tensor = torch.randn(batch_size, 3, 224, 224, device=self.device)
            seg_target = torch.randint(0, 5, (batch_size, 224, 224), device=self.device)
            cls_target = torch.randint(0, 5, (batch_size,), device=self.device)
            
            # Forward pass
            optimizer.zero_grad()
            seg_output, cls_output = model(input_tensor)
            loss, loss_dict = loss_fn(seg_output, cls_output, seg_target, cls_target)
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        results['grad_nan'] = True
                        logger.error(f"❌ NaN gradients in {name}")
                    
                    if torch.isinf(param.grad).any():
                        results['grad_inf'] = True
                        logger.error(f"❌ Inf gradients in {name}")
                    
                    # Calculate gradient norm
                    grad_norm = param.grad.norm().item()
                    results['grad_norms'][name] = grad_norm
                    
                    if grad_norm > 1000:
                        logger.warning(f"⚠️ Large gradient norm in {name}: {grad_norm:.2e}")
            
            # Total gradient norm
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            results['total_grad_norm'] = total_grad_norm
            
            results['backward_pass_success'] = True
            logger.info(f"✅ Backward pass successful, total grad norm: {total_grad_norm:.2e}")
            
        except Exception as e:
            logger.error(f"❌ Backward pass failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def test_optimizer_step(self, model: nn.Module, optimizer: optim.Optimizer, batch_size: int = 4) -> dict:
        """Test optimizer step"""
        logger.info("🔍 Testing optimizer step...")
        
        model.train()
        loss_fn = StableLoss()
        results = {
            'optimizer_step_success': False,
            'weight_change': False,
            'weight_stats_before': {},
            'weight_stats_after': {}
        }
        
        try:
            # Save initial weights
            initial_weights = {}
            for name, param in model.named_parameters():
                initial_weights[name] = param.data.clone()
                results['weight_stats_before'][name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item()
                }
            
            # Training step
            input_tensor = torch.randn(batch_size, 3, 224, 224, device=self.device)
            seg_target = torch.randint(0, 5, (batch_size, 224, 224), device=self.device)
            cls_target = torch.randint(0, 5, (batch_size,), device=self.device)
            
            optimizer.zero_grad()
            seg_output, cls_output = model(input_tensor)
            loss, loss_dict = loss_fn(seg_output, cls_output, seg_target, cls_target)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Check weight changes
            for name, param in model.named_parameters():
                weight_diff = (param.data - initial_weights[name]).abs().max().item()
                if weight_diff > 1e-10:
                    results['weight_change'] = True
                
                results['weight_stats_after'][name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'max_change': weight_diff
                }
            
            results['optimizer_step_success'] = True
            
            if results['weight_change']:
                logger.info("✅ Optimizer step successful, weights updated")
            else:
                logger.warning("⚠️ Optimizer step completed but no weight changes detected")
            
        except Exception as e:
            logger.error(f"❌ Optimizer step failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def test_learning_rate_values(self, optimizer: optim.Optimizer) -> dict:
        """Test if learning rate values are reasonable"""
        logger.info("🔍 Testing learning rate values...")
        
        results = {
            'lr_values': [],
            'lr_too_small': False,
            'lr_too_large': False,
            'lr_reasonable': True
        }
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            results['lr_values'].append(lr)
            
            if lr < 1e-8:
                results['lr_too_small'] = True
                results['lr_reasonable'] = False
                logger.warning(f"⚠️ Learning rate too small: {lr:.2e}")
            
            if lr > 1e-1:
                results['lr_too_large'] = True
                results['lr_reasonable'] = False
                logger.warning(f"⚠️ Learning rate too large: {lr:.2e}")
        
        if results['lr_reasonable']:
            logger.info(f"✅ Learning rates look reasonable: {results['lr_values']}")
        
        return results
    
    def test_data_quality(self, dataset, batch_size: int = 4) -> dict:
        """Test data quality"""
        logger.info("🔍 Testing data quality...")
        
        results = {
            'data_quality_good': True,
            'nan_in_data': False,
            'inf_in_data': False,
            'data_stats': {}
        }
        
        try:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            for i, (images, seg_masks, cls_labels) in enumerate(dataloader):
                if i >= 5:  # Test first 5 batches
                    break
                
                # Check images
                if torch.isnan(images).any():
                    results['nan_in_data'] = True
                    results['data_quality_good'] = False
                    logger.error(f"❌ NaN in images, batch {i}")
                
                if torch.isinf(images).any():
                    results['inf_in_data'] = True
                    results['data_quality_good'] = False
                    logger.error(f"❌ Inf in images, batch {i}")
                
                # Collect data statistics
                results['data_stats'][f'batch_{i}'] = {
                    'image_mean': images.mean().item(),
                    'image_std': images.std().item(),
                    'image_min': images.min().item(),
                    'image_max': images.max().item(),
                    'seg_mask_unique': torch.unique(seg_masks).tolist(),
                    'cls_labels_unique': torch.unique(cls_labels).tolist()
                }
            
            if results['data_quality_good']:
                logger.info("✅ Data quality looks good")
        
        except Exception as e:
            logger.error(f"❌ Data quality test failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_full_diagnostics(self) -> dict:
        """Run complete diagnostic suite"""
        logger.info("🚀 Starting full training diagnostics...")
        
        # Initialize model and optimizer
        model = EnhancedAdaptiveStreamModel(num_classes=5).to(self.device)
        optimizer = optim.AdamW(model.parameters(), **TrainingConfig.get_optimizer_config())
        dataset = SyntheticDataset(num_samples=100)
        
        # Run all tests
        self.results['model_init'] = self.test_model_initialization(model)
        self.results['forward_pass'] = self.test_forward_pass(model)
        self.results['loss_computation'] = self.test_loss_computation(model)
        self.results['backward_pass'] = self.test_backward_pass(model, optimizer)
        self.results['optimizer_step'] = self.test_optimizer_step(model, optimizer)
        self.results['learning_rate'] = self.test_learning_rate_values(optimizer)
        self.results['data_quality'] = self.test_data_quality(dataset)
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate diagnostic report"""
        report = []
        report.append("=" * 60)
        report.append("TRAINING DIAGNOSTICS REPORT")
        report.append("=" * 60)
        
        if not self.results:
            report.append("No diagnostics run yet. Call run_full_diagnostics() first.")
            return "\n".join(report)
        
        # Summarize issues
        issues = []
        fixes = []
        
        # Model initialization issues
        if self.results.get('model_init', {}).get('nan_weights'):
            issues.append("NaN weights in model initialization")
            fixes.append("Re-initialize model with proper weight initialization")
        
        # Forward pass issues
        if self.results.get('forward_pass', {}).get('output_nan'):
            issues.append("NaN in forward pass outputs")
            fixes.append("Check model architecture and add numerical stability")
        
        # Loss computation issues
        if self.results.get('loss_computation', {}).get('loss_nan'):
            issues.append("NaN in loss computation")
            fixes.append("Use stable loss function with proper epsilon values")
        
        # Gradient issues
        if self.results.get('backward_pass', {}).get('grad_nan'):
            issues.append("NaN in gradients")
            fixes.append("Add gradient clipping and check loss function")
        
        # Learning rate issues
        lr_results = self.results.get('learning_rate', {})
        if lr_results.get('lr_too_small'):
            issues.append("Learning rate too small")
            fixes.append("Increase learning rate or reset scheduler")
        if lr_results.get('lr_too_large'):
            issues.append("Learning rate too large")
            fixes.append("Decrease learning rate")
        
        # Data quality issues
        if self.results.get('data_quality', {}).get('nan_in_data'):
            issues.append("NaN in input data")
            fixes.append("Clean input data and add proper preprocessing")
        
        # Generate report
        if issues:
            report.append(f"❌ ISSUES FOUND ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                report.append(f"  {i}. {issue}")
            
            report.append("\n🔧 SUGGESTED FIXES:")
            for i, fix in enumerate(fixes, 1):
                report.append(f"  {i}. {fix}")
        else:
            report.append("✅ NO MAJOR ISSUES FOUND")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "diagnostic_report.txt"):
        """Save diagnostic report to file"""
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            f.write(report)
            f.write("\n\nDETAILED RESULTS:\n")
            f.write("=" * 40 + "\n")
            
            for test_name, test_results in self.results.items():
                f.write(f"\n{test_name.upper()}:\n")
                f.write(str(test_results))
                f.write("\n")
        
        logger.info(f"📝 Diagnostic report saved to {filename}")


def main():
    """Run diagnostics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print configurations
    print("\n" + "=" * 60)
    print("RUNNING TRAINING DIAGNOSTICS")
    print("=" * 60)
    
    TrainingConfig.print_config()
    print()
    DebugConfig.print_debug_config()
    
    # Run diagnostics
    diagnostics = TrainingDiagnostics(device)
    results = diagnostics.run_full_diagnostics()
    
    # Generate and print report
    report = diagnostics.generate_report()
    print("\n" + report)
    
    # Save report
    diagnostics.save_report("training_diagnostics_report.txt")
    
    return results


if __name__ == "__main__":
    main()