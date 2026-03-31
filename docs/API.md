# Internal API Documentation

This document describes the key classes and functions within the `src/` directory.

## Models (`src.classification.model`)

### `CropDiseaseClassifier`
The primary model for Stage 1 classification.
- `__init__(model_name, num_classes, env_feature_dim)`: Initializes the Swin Transformer and MLP fusion head.
- `forward(pixel_values, env_features)`: Multi-modal forward pass. Returns logits.

## Loss Functions (`src.classification.loss`)

### `FocalLoss`
Custom loss for imbalanced disease datasets.
- `__init__(alpha, gamma, reduction)`: Configures the focusing parameters.
- `forward(inputs, targets)`: Calculates weighted cross-entropy loss.

## Utilities (`src.utils`)

### `Config` (`config.py`)
Centralized configuration management.
- `load(path)`: Loads configuration from YAML with environment variable overrides.
- `save(path)`: Persists current configuration to file.

### `Logger` (`logger.py`)
Professional logging system.
- `setup_logger(name, level)`: Returns a logger with console and daily rotating file handlers.

### `Preprocess` (`preprocess.py`)
- `get_train_transforms(size)`: Returns advanced training augmentations.
- `get_val_transforms(size)`: Returns standard validation transforms.
- `scale_environmental_data(env_dict)`: Normalizes environmental context into tensors.

### `Metrics` (`metrics.py`)
- `calculate_classification_metrics(y_true, y_pred)`: Returns Acc, F1, Precision, Recall.
- `calculate_iou(box1, box2)`: Calculates IoU for two bounding boxes.
- `calculate_segmentation_iou(pred_mask, target_mask)`: Calculates mIoU for segmentations.
