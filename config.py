#!/usr/bin/env python3
"""
Configuration file for Enhanced AdaptiveStream training parameters.
This file contains all the hyperparameters that can be tuned to fix training issues.
"""

import torch

class TrainingConfig:
    """Training configuration class with robust default parameters"""
    
    # Model parameters
    NUM_CLASSES = 5
    INPUT_CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 8  # Reduced batch size to prevent memory issues
    NUM_EPOCHS = 50
    
    # Learning rate parameters (CRITICAL for fixing NaN issues)
    BASE_LR = 1e-3          # Base learning rate (reduced from potentially higher values)
    MIN_LR = 1e-6           # Minimum learning rate to prevent underflow
    WARMUP_EPOCHS = 5       # Number of warmup epochs
    DECAY_FACTOR = 0.95     # Learning rate decay factor
    
    # Optimizer parameters
    WEIGHT_DECAY = 1e-4     # L2 regularization
    BETAS = (0.9, 0.999)    # Adam betas
    EPS = 1e-8              # Adam epsilon for numerical stability
    
    # Gradient clipping (CRITICAL for preventing NaN)
    MAX_GRAD_NORM = 1.0     # Maximum gradient norm
    
    # Loss function weights
    SEG_WEIGHT = 1.0        # Segmentation loss weight
    CLS_WEIGHT = 0.5        # Classification loss weight
    
    # Early stopping
    PATIENCE = 10           # Early stopping patience
    
    # Numerical stability parameters
    CLAMP_MIN = -10         # Minimum value for clamping logits
    CLAMP_MAX = 10          # Maximum value for clamping logits
    EPS_DIVISION = 1e-8     # Epsilon for safe division
    
    # Data loading
    NUM_WORKERS = 2         # Number of data loading workers
    PIN_MEMORY = True       # Pin memory for faster GPU transfer
    
    # Model initialization
    INIT_GAIN = 0.02        # Initialization gain for better stability
    
    # Logging and checkpointing
    LOG_INTERVAL = 100      # Log every N batches
    SAVE_INTERVAL = 5       # Save checkpoint every N epochs
    
    # Dataset parameters (for synthetic data)
    TRAIN_SAMPLES = 1000    # Number of training samples
    VAL_SAMPLES = 200       # Number of validation samples
    IMAGE_SIZE = 224        # Input image size
    
    # Device configuration
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
    
    # Mixed precision training (can help with stability)
    USE_AMP = False         # Automatic Mixed Precision
    
    # Dropout rates
    DROPOUT_RATE = 0.5      # Dropout rate in classifier
    
    # Batch normalization
    BN_MOMENTUM = 0.1       # BatchNorm momentum
    BN_EPS = 1e-5          # BatchNorm epsilon
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("Enhanced AdaptiveStream Training Configuration:")
        print("=" * 50)
        
        attrs = [attr for attr in dir(cls) if not attr.startswith('_') and not callable(getattr(cls, attr))]
        
        for attr in sorted(attrs):
            if attr != 'print_config':
                value = getattr(cls, attr)
                print(f"{attr:20}: {value}")
        print("=" * 50)
    
    @classmethod
    def get_optimizer_config(cls):
        """Get optimizer configuration dictionary"""
        return {
            'lr': cls.BASE_LR,
            'weight_decay': cls.WEIGHT_DECAY,
            'betas': cls.BETAS,
            'eps': cls.EPS
        }
    
    @classmethod
    def get_scheduler_config(cls):
        """Get scheduler configuration dictionary"""
        return {
            'warmup_epochs': cls.WARMUP_EPOCHS,
            'base_lr': cls.BASE_LR,
            'min_lr': cls.MIN_LR,
            'decay_factor': cls.DECAY_FACTOR
        }
    
    @classmethod
    def get_loss_config(cls):
        """Get loss function configuration dictionary"""
        return {
            'seg_weight': cls.SEG_WEIGHT,
            'cls_weight': cls.CLS_WEIGHT
        }


class DebugConfig:
    """Configuration for debugging NaN issues"""
    
    # Enable various debugging features
    ENABLE_NAN_DETECTION = True
    ENABLE_GRADIENT_MONITORING = True
    ENABLE_WEIGHT_MONITORING = True
    ENABLE_LOSS_MONITORING = True
    
    # Debugging thresholds
    LARGE_GRAD_THRESHOLD = 100.0    # Threshold for large gradient detection
    LARGE_LOSS_THRESHOLD = 1e6      # Threshold for large loss detection
    
    # Fallback values for NaN replacement
    NAN_REPLACEMENT = 0.0
    INF_REPLACEMENT_POS = 1e6
    INF_REPLACEMENT_NEG = -1e6
    
    # Monitoring intervals
    MONITOR_EVERY_N_BATCHES = 50    # Monitor weights/gradients every N batches
    
    @classmethod
    def print_debug_config(cls):
        """Print debugging configuration"""
        print("Debug Configuration:")
        print("=" * 30)
        
        attrs = [attr for attr in dir(cls) if not attr.startswith('_') and not callable(getattr(cls, attr))]
        
        for attr in sorted(attrs):
            if attr != 'print_debug_config':
                value = getattr(cls, attr)
                print(f"{attr:25}: {value}")
        print("=" * 30)


class ExperimentalConfig:
    """Experimental configurations to try if standard approach fails"""
    
    # Alternative learning rate schedules
    ALT_LR_SCHEDULES = {
        'cosine': {
            'T_max': 50,
            'eta_min': 1e-6
        },
        'step': {
            'step_size': 10,
            'gamma': 0.7
        },
        'plateau': {
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-6
        }
    }
    
    # Alternative optimizers
    ALT_OPTIMIZERS = {
        'sgd': {
            'lr': 1e-2,
            'momentum': 0.9,
            'weight_decay': 1e-4
        },
        'rmsprop': {
            'lr': 1e-3,
            'alpha': 0.99,
            'eps': 1e-8,
            'weight_decay': 1e-4
        }
    }
    
    # Different initialization schemes
    INIT_SCHEMES = ['kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform']
    
    # Different loss functions
    ALT_LOSS_FUNCTIONS = {
        'focal': {
            'alpha': 0.25,
            'gamma': 2.0
        },
        'label_smoothing': {
            'smoothing': 0.1
        }
    }
    
    @classmethod
    def print_experimental_config(cls):
        """Print experimental configuration options"""
        print("Experimental Configuration Options:")
        print("=" * 40)
        
        print("Alternative LR Schedules:")
        for name, config in cls.ALT_LR_SCHEDULES.items():
            print(f"  {name}: {config}")
        
        print("\nAlternative Optimizers:")
        for name, config in cls.ALT_OPTIMIZERS.items():
            print(f"  {name}: {config}")
        
        print(f"\nInitialization Schemes: {cls.INIT_SCHEMES}")
        
        print("\nAlternative Loss Functions:")
        for name, config in cls.ALT_LOSS_FUNCTIONS.items():
            print(f"  {name}: {config}")
        
        print("=" * 40)


if __name__ == "__main__":
    # Print all configurations
    TrainingConfig.print_config()
    print()
    DebugConfig.print_debug_config()
    print()
    ExperimentalConfig.print_experimental_config()