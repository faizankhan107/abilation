#!/usr/bin/env python3
"""
Script to extract per-class metrics (IoU, precision, recall, F1) from model training CSV files.
Generates formatted output showing epoch-wise per-class performance for each model.
"""

import csv
import os
from pathlib import Path

def load_csv_files():
    """Load all CSV files in the directory."""
    csv_files = {}
    current_dir = Path("/home/runner/work/abilation/abilation")
    
    for file_path in current_dir.glob("*_COMPLETE_race_matrices.csv"):
        model_name = file_path.stem.replace("_COMPLETE_race_matrices", "")
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)
                csv_files[model_name] = data
                print(f"Loaded {model_name}: {len(data)} epochs")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return csv_files

def handle_null_values(value):
    """Handle null/zero values appropriately."""
    if value is None or value == '' or float(value) == 0.0:
        return "N/A"
    return f"{float(value):.4f}"

def extract_per_class_metrics(data, model_name):
    """Extract per-class metrics for validation data."""
    results = []
    
    # Define class mappings for race classification 
    race_classes = ['white', 'black', 'asian', 'indian', 'others']
    
    # Define class mappings for segmentation
    seg_classes = ['background', 'face']
    
    for row in data:
        epoch = int(float(row['epoch']))
        
        # Get overall metrics
        train_loss = float(row.get('train_segmentation_loss', 0))
        train_acc = float(row.get('train_segmentation_accuracy', 0))
        train_miou = float(row.get('train_segmentation_miou', 0))
        val_loss = float(row.get('val_segmentation_loss', 0))
        val_acc = float(row.get('val_segmentation_accuracy', 0))
        val_miou = float(row.get('val_segmentation_miou', 0))
        
        epoch_data = {
            'epoch': epoch,
            'model': model_name,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_miou': train_miou,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_miou': val_miou,
            'classes': []
        }
        
        # Extract race classification metrics (validation)
        for i, class_name in enumerate(race_classes):
            precision = float(row.get(f'val_{class_name}_precision', 0))
            recall = float(row.get(f'val_{class_name}_recall', 0))
            f1 = float(row.get(f'val_{class_name}_f1', 0))
            iou = float(row.get(f'val_{class_name}_iou', 0))
            
            class_data = {
                'class_id': i,
                'class_name': class_name,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'iou': iou
            }
            epoch_data['classes'].append(class_data)
        
        # Add segmentation classes 
        for i, class_name in enumerate(seg_classes):
            precision = float(row.get(f'val_{class_name}_precision', 0))
            recall = float(row.get(f'val_{class_name}_recall', 0))
            f1 = float(row.get(f'val_{class_name}_f1', 0))
            iou = float(row.get(f'val_{class_name}_iou', 0))
            
            class_data = {
                'class_id': i + len(race_classes),
                'class_name': f"seg_{class_name}",
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'iou': iou
            }
            epoch_data['classes'].append(class_data)
        
        results.append(epoch_data)
    
    return results

def format_epoch_results(epoch_data):
    """Format epoch results in the requested style."""
    output = []
    output.append("=" * 60)
    output.append(f"Epoch {epoch_data['epoch']} Results ({epoch_data['model']}):")
    output.append(f"           | Precision  | Recall     | F1         | IoU       ")
    output.append("-" * 60)
    
    # Calculate mean IoU for non-zero values
    valid_ious = [cls['iou'] for cls in epoch_data['classes'] if cls['iou'] > 0]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0
    
    # Format each class
    for cls in epoch_data['classes']:
        class_name = cls['class_name'].replace('seg_', '').title()
        if cls['class_name'].startswith('seg_'):
            class_name = f"Seg-{class_name}"
        
        output.append(f"{class_name:<8} | {handle_null_values(cls['precision']):<10} | "
                     f"{handle_null_values(cls['recall']):<10} | {handle_null_values(cls['f1']):<10} | "
                     f"{handle_null_values(cls['iou']):<10}")
    
    output.append("-" * 60)
    output.append(f"Mean IoU: {mean_iou:.4f}")
    output.append(f"Overall Accuracy: {epoch_data['val_acc']:.4f}")
    output.append("=" * 60)
    output.append("")
    output.append("Epoch Summary:")
    output.append(f"Train - Loss: {epoch_data['train_loss']:.4f}, Acc: {epoch_data['train_acc']:.4f}, IoU: {epoch_data['train_miou']:.4f}")
    output.append(f"Val   - Loss: {epoch_data['val_loss']:.4f}, Acc: {epoch_data['val_acc']:.4f}, IoU: {epoch_data['val_miou']:.4f}")
    
    # Check if this was the best epoch
    if epoch_data['val_loss'] < 0.5:  # Simple heuristic
        output.append(f"✓ Good model performance - Loss: {epoch_data['val_loss']:.4f}")
    
    output.append("")
    
    return "\n".join(output)

def analyze_model_performance(csv_files, target_miou=0.4039):
    """Analyze which models have mIoU below or around the target."""
    print(f"\n=== MODEL PERFORMANCE ANALYSIS ===")
    print(f"Target mIoU: {target_miou}")
    print("-" * 50)
    
    for model_name, data in csv_files.items():
        val_mious = []
        for row in data:
            if 'val_segmentation_miou' in row and row['val_segmentation_miou']:
                val_mious.append(float(row['val_segmentation_miou']))
        
        if val_mious:
            max_miou = max(val_mious)
            mean_miou = sum(val_mious) / len(val_mious)
            final_miou = val_mious[-1]
            
            status = "✓ ABOVE TARGET" if max_miou > target_miou else "✗ BELOW TARGET"
            
            print(f"{model_name:15} | Max: {max_miou:.4f} | Mean: {mean_miou:.4f} | Final: {final_miou:.4f} | {status}")
        else:
            print(f"{model_name:15} | No mIoU data available")

def main():
    print("=== EXTRACTING PER-CLASS METRICS FROM MODEL TRAINING DATA ===\n")
    
    # Load CSV files
    csv_files = load_csv_files()
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    # Analyze overall model performance
    analyze_model_performance(csv_files)
    
    # Extract and display metrics for each model
    for model_name, data in csv_files.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING MODEL: {model_name.upper()}")
        print(f"{'='*80}")
        
        results = extract_per_class_metrics(data, model_name)
        
        # Show results for first few epochs and last few epochs
        epochs_to_show = []
        if len(results) > 10:
            epochs_to_show = results[:3] + ["..."] + results[-3:]
        else:
            epochs_to_show = results
        
        for epoch_data in epochs_to_show:
            if epoch_data == "...":
                print("... (intermediate epochs omitted) ...\n")
                continue
            print(format_epoch_results(epoch_data))
        
        # Save detailed results to file
        output_file = f"/home/runner/work/abilation/abilation/{model_name}_detailed_metrics.txt"
        with open(output_file, 'w') as f:
            f.write(f"DETAILED METRICS FOR {model_name.upper()}\n")
            f.write("="*80 + "\n\n")
            for epoch_data in results:
                f.write(format_epoch_results(epoch_data) + "\n")
        
        print(f"✓ Detailed metrics saved to: {output_file}")

if __name__ == "__main__":
    main()