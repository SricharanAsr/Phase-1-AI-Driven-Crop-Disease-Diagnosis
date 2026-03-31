import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from typing import Dict, List, Union

def calculate_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor], 
    y_pred: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """Calculates standard classification metrics for disease diagnosis.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class indices.

    Returns:
        Dict[str, float]: Dictionary containing Accuracy, F1, Precision, and Recall.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0)
    }

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculates Intersection over Union (IoU) for two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def calculate_segmentation_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    """Calculates Mean IoU for semantic segmentation masks."""
    dims = (1, 2)
    intersection = (pred_mask & target_mask).float().sum(dims)
    union = (pred_mask | target_mask).float().sum(dims)
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()
