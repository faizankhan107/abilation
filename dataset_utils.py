#!/usr/bin/env python3
"""
UTKFace Dataset Utilities

This module provides utilities for preparing and validating the UTKFace dataset
for use with the enhanced segmentation comparison framework.

Features:
- Dataset structure validation
- Color mask validation and conversion
- Statistics and quality checks
- Morphological cleaning operations
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import argparse

# Import race classes and color mappings
RACE_CLASSES = ['White', 'Black', 'Asian', 'Indian', 'Others']
RACE_COLOR_MAP = {
    0: (255, 250, 250),  # White
    1: (139, 69, 19),    # Black  
    2: (255, 215, 0),    # Asian
    3: (255, 153, 51),   # Indian
    4: (152, 251, 152)   # Others
}

def validate_dataset_structure(dataset_dir: str) -> Dict[str, any]:
    """
    Validate UTKFace dataset structure and return statistics.
    
    Args:
        dataset_dir: Path to dataset directory
        
    Returns:
        Dictionary with validation results and statistics
    """
    dataset_path = Path(dataset_dir)
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {
            'total_images': 0,
            'total_masks': 0,
            'matched_pairs': 0,
            'race_distribution': defaultdict(int),
            'age_distribution': defaultdict(int),
            'gender_distribution': defaultdict(int)
        }
    }
    
    # Check main directory
    if not dataset_path.exists():
        results['valid'] = False
        results['errors'].append(f"Dataset directory not found: {dataset_dir}")
        return results
    
    # Check subdirectories
    images_dir = dataset_path / 'images'
    masks_dir = dataset_path / 'masks'
    
    if not images_dir.exists():
        results['valid'] = False
        results['errors'].append("Images directory not found: images/")
    
    if not masks_dir.exists():
        results['valid'] = False
        results['errors'].append("Masks directory not found: masks/")
    
    if not results['valid']:
        return results
    
    # Collect image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'*{ext.upper()}'))
    
    results['statistics']['total_images'] = len(image_files)
    
    # Collect mask files
    mask_files = list(masks_dir.glob('*_mask.png'))
    results['statistics']['total_masks'] = len(mask_files)
    
    # Validate filename format and collect statistics
    valid_pairs = 0
    
    for image_path in image_files:
        filename = image_path.stem
        
        try:
            # Parse UTKFace filename: age_gender_race_date&time.jpg
            parts = filename.split('_')
            
            if len(parts) >= 3:
                age = int(parts[0])
                gender = int(parts[1])  # 0: male, 1: female
                race = int(parts[2])    # 0-4: race classes
                
                # Check if corresponding mask exists
                mask_path = masks_dir / f"{filename}_mask.png"
                
                if mask_path.exists():
                    valid_pairs += 1
                    
                    # Update statistics
                    results['statistics']['race_distribution'][race] += 1
                    results['statistics']['age_distribution'][age // 10 * 10] += 1  # Age groups
                    results['statistics']['gender_distribution'][gender] += 1
                else:
                    results['warnings'].append(f"Missing mask for image: {filename}")
                    
                # Validate race class
                if race < 0 or race >= len(RACE_CLASSES):
                    results['warnings'].append(f"Invalid race class {race} in file: {filename}")
                    
            else:
                results['warnings'].append(f"Invalid filename format: {filename}")
                
        except (ValueError, IndexError) as e:
            results['warnings'].append(f"Error parsing filename {filename}: {e}")
    
    results['statistics']['matched_pairs'] = valid_pairs
    
    # Check if we have sufficient data
    if valid_pairs == 0:
        results['valid'] = False
        results['errors'].append("No valid image-mask pairs found")
    elif valid_pairs < 100:
        results['warnings'].append(f"Very few valid pairs found: {valid_pairs}")
    
    return results

def validate_color_masks(dataset_dir: str, sample_size: int = 100) -> Dict[str, any]:
    """
    Validate color masks and check color mapping consistency.
    
    Args:
        dataset_dir: Path to dataset directory
        sample_size: Number of masks to sample for validation
        
    Returns:
        Dictionary with validation results
    """
    masks_dir = Path(dataset_dir) / 'masks'
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'color_stats': {
            'total_masks_checked': 0,
            'valid_color_masks': 0,
            'color_distribution': defaultdict(int),
            'unexpected_colors': set()
        }
    }
    
    if not masks_dir.exists():
        results['valid'] = False
        results['errors'].append("Masks directory not found")
        return results
    
    # Get mask files
    mask_files = list(masks_dir.glob('*_mask.png'))
    
    if not mask_files:
        results['valid'] = False
        results['errors'].append("No mask files found")
        return results
    
    # Sample masks for checking
    import random
    sample_masks = random.sample(mask_files, min(sample_size, len(mask_files)))
    
    expected_colors = set(RACE_COLOR_MAP.values())
    black_color = (0, 0, 0)  # Background
    
    for mask_path in sample_masks:
        try:
            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            if mask is None:
                results['errors'].append(f"Could not load mask: {mask_path.name}")
                continue
            
            results['color_stats']['total_masks_checked'] += 1
            
            # Get unique colors in the mask
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            unique_colors = set()
            
            h, w, c = mask_rgb.shape
            for i in range(h):
                for j in range(w):
                    color = tuple(mask_rgb[i, j])
                    unique_colors.add(color)
            
            # Check if colors are valid
            valid_mask = True
            for color in unique_colors:
                # Convert RGB back to BGR for comparison
                bgr_color = (color[2], color[1], color[0])
                
                if bgr_color == black_color:
                    continue  # Background is allowed
                elif bgr_color in expected_colors:
                    # Find which race class this color belongs to
                    for race_id, race_color in RACE_COLOR_MAP.items():
                        if race_color == bgr_color:
                            results['color_stats']['color_distribution'][race_id] += 1
                            break
                else:
                    results['color_stats']['unexpected_colors'].add(color)
                    valid_mask = False
            
            if valid_mask:
                results['color_stats']['valid_color_masks'] += 1
            else:
                results['warnings'].append(f"Unexpected colors in mask: {mask_path.name}")
                
        except Exception as e:
            results['errors'].append(f"Error processing mask {mask_path.name}: {e}")
    
    # Convert set to list for JSON serialization
    results['color_stats']['unexpected_colors'] = list(results['color_stats']['unexpected_colors'])
    
    return results

def clean_masks_morphological(dataset_dir: str, 
                             kernel_size: int = 3,
                             iterations: int = 1,
                             backup: bool = True) -> int:
    """
    Apply morphological operations to clean up mask files.
    
    Args:
        dataset_dir: Path to dataset directory
        kernel_size: Size of morphological kernel
        iterations: Number of iterations for operations
        backup: Whether to create backup of original masks
        
    Returns:
        Number of masks processed
    """
    masks_dir = Path(dataset_dir) / 'masks'
    
    if not masks_dir.exists():
        raise FileNotFoundError("Masks directory not found")
    
    # Create backup directory if requested
    if backup:
        backup_dir = masks_dir.parent / 'masks_backup'
        backup_dir.mkdir(exist_ok=True)
    
    mask_files = list(masks_dir.glob('*_mask.png'))
    processed_count = 0
    
    # Morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    for mask_path in mask_files:
        try:
            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            if mask is None:
                continue
            
            # Backup original if requested
            if backup:
                backup_path = backup_dir / mask_path.name
                cv2.imwrite(str(backup_path), mask)
            
            # Apply morphological operations
            # Opening: erosion followed by dilation (removes noise)
            mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
            
            # Closing: dilation followed by erosion (fills holes)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            
            # Save cleaned mask
            cv2.imwrite(str(mask_path), mask_cleaned)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {mask_path.name}: {e}")
    
    return processed_count

def generate_dataset_report(dataset_dir: str, output_file: str = 'dataset_report.html'):
    """
    Generate a comprehensive HTML report of dataset statistics and validation.
    
    Args:
        dataset_dir: Path to dataset directory
        output_file: Output HTML file path
    """
    # Validate dataset
    structure_results = validate_dataset_structure(dataset_dir)
    color_results = validate_color_masks(dataset_dir)
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>UTKFace Dataset Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .error {{ color: red; }}
            .warning {{ color: orange; }}
            .success {{ color: green; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>UTKFace Dataset Validation Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Dataset Directory: {dataset_dir}</p>
        </div>
        
        <div class="section">
            <h2>Dataset Structure Validation</h2>
            <p class="{'success' if structure_results['valid'] else 'error'}">
                Status: {'✓ Valid' if structure_results['valid'] else '✗ Invalid'}
            </p>
            
            {_format_errors_warnings(structure_results)}
            
            <h3>Statistics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Images</td><td>{structure_results['statistics']['total_images']}</td></tr>
                <tr><td>Total Masks</td><td>{structure_results['statistics']['total_masks']}</td></tr>
                <tr><td>Valid Pairs</td><td>{structure_results['statistics']['matched_pairs']}</td></tr>
            </table>
            
            <h3>Race Distribution</h3>
            <table>
                <tr><th>Race Class</th><th>Count</th><th>Percentage</th></tr>
                {_format_race_distribution(structure_results['statistics']['race_distribution'])}
            </table>
        </div>
        
        <div class="section">
            <h2>Color Mask Validation</h2>
            <p class="{'success' if color_results['valid'] else 'error'}">
                Status: {'✓ Valid' if color_results['valid'] else '✗ Invalid'}
            </p>
            
            {_format_errors_warnings(color_results)}
            
            <h3>Color Statistics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Masks Checked</td><td>{color_results['color_stats']['total_masks_checked']}</td></tr>
                <tr><td>Valid Color Masks</td><td>{color_results['color_stats']['valid_color_masks']}</td></tr>
                <tr><td>Unexpected Colors Found</td><td>{len(color_results['color_stats']['unexpected_colors'])}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            {_generate_recommendations(structure_results, color_results)}
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Dataset report generated: {output_file}")

def _format_errors_warnings(results: Dict) -> str:
    """Format errors and warnings for HTML report."""
    html = ""
    
    if results['errors']:
        html += "<h4 class='error'>Errors:</h4><ul>"
        for error in results['errors']:
            html += f"<li class='error'>{error}</li>"
        html += "</ul>"
    
    if results['warnings']:
        html += "<h4 class='warning'>Warnings:</h4><ul>"
        for warning in results['warnings']:
            html += f"<li class='warning'>{warning}</li>"
        html += "</ul>"
    
    return html

def _format_race_distribution(race_dist: Dict) -> str:
    """Format race distribution for HTML table."""
    total = sum(race_dist.values())
    html = ""
    
    for race_id in range(len(RACE_CLASSES)):
        count = race_dist.get(race_id, 0)
        percentage = (count / total * 100) if total > 0 else 0
        html += f"<tr><td>{RACE_CLASSES[race_id]}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
    
    return html

def _generate_recommendations(structure_results: Dict, color_results: Dict) -> str:
    """Generate recommendations based on validation results."""
    recommendations = []
    
    # Check for common issues
    stats = structure_results['statistics']
    
    if stats['matched_pairs'] < 1000:
        recommendations.append("Consider obtaining more data - current dataset is quite small for deep learning.")
    
    if len(structure_results['warnings']) > stats['matched_pairs'] * 0.1:
        recommendations.append("High number of warnings detected - review filename formats and file integrity.")
    
    if color_results['color_stats']['valid_color_masks'] < color_results['color_stats']['total_masks_checked'] * 0.9:
        recommendations.append("Many masks contain unexpected colors - consider running mask cleaning operations.")
    
    # Check race balance
    race_counts = [stats['race_distribution'].get(i, 0) for i in range(len(RACE_CLASSES))]
    if max(race_counts) > min(race_counts) * 5:  # Imbalance threshold
        recommendations.append("Race classes are highly imbalanced - consider data augmentation or class weighting.")
    
    if not recommendations:
        recommendations.append("Dataset appears to be in good condition for training!")
    
    html = "<ul>"
    for rec in recommendations:
        html += f"<li>{rec}</li>"
    html += "</ul>"
    
    return html

def main():
    """Command line interface for dataset utilities."""
    parser = argparse.ArgumentParser(description='UTKFace Dataset Utilities')
    parser.add_argument('dataset_dir', help='Path to UTKFace dataset directory')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate dataset structure and masks')
    parser.add_argument('--clean-masks', action='store_true',
                       help='Apply morphological cleaning to masks')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comprehensive HTML report')
    parser.add_argument('--kernel-size', type=int, default=3,
                       help='Kernel size for morphological operations')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of iterations for morphological operations')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup when cleaning masks')
    
    args = parser.parse_args()
    
    if args.validate:
        print("Validating dataset structure...")
        structure_results = validate_dataset_structure(args.dataset_dir)
        
        print(f"Dataset valid: {structure_results['valid']}")
        print(f"Total images: {structure_results['statistics']['total_images']}")
        print(f"Valid pairs: {structure_results['statistics']['matched_pairs']}")
        
        if structure_results['errors']:
            print("\nErrors:")
            for error in structure_results['errors']:
                print(f"  - {error}")
        
        if structure_results['warnings']:
            print("\nWarnings:")
            for warning in structure_results['warnings']:
                print(f"  - {warning}")
        
        print("\nValidating color masks...")
        color_results = validate_color_masks(args.dataset_dir)
        
        print(f"Valid color masks: {color_results['color_stats']['valid_color_masks']}"
              f"/{color_results['color_stats']['total_masks_checked']}")
        
        if color_results['color_stats']['unexpected_colors']:
            print(f"Unexpected colors found: {len(color_results['color_stats']['unexpected_colors'])}")
    
    if args.clean_masks:
        print("Cleaning masks with morphological operations...")
        processed = clean_masks_morphological(
            args.dataset_dir,
            kernel_size=args.kernel_size,
            iterations=args.iterations,
            backup=not args.no_backup
        )
        print(f"Processed {processed} mask files")
    
    if args.generate_report:
        print("Generating dataset report...")
        generate_dataset_report(args.dataset_dir)

if __name__ == "__main__":
    main()