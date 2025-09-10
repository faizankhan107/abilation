import json
import os

# Load the JSON file
json_file_path = r'D:\NPU\artical\Race classification\results folder\UTKFace results with ablation\Race segmentation UTK metrics\race_segmentation_bg_minus1_training_log.json'

# Class mapping (example mapping, adjust as necessary)
class_mapping = {
    0: 'Class A',
    1: 'Class B',
    2: 'Class C',
    3: 'Class D',
}

# Function to format metrics
def format_metrics(metrics):
    header = 'Class   Precision   Recall   F1   IoU'
    formatted_output = [header]
    for class_id, metric in metrics.items():
        line = f'{class_mapping[class_id]:<7} {metric[