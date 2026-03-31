import torch
import numpy as np
from src.utils.preprocess import scale_environmental_data
from src.utils.metrics import calculate_iou, calculate_classification_metrics
from src.utils.config import Config

def test_environmental_scaling():
    """Tests normalization of environmental data."""
    test_env = {"temperature": 25.0, "humidity": 50.0, "n": 75, "p": 75, "k": 75}
    # Ranges: T(50), H(100), NPK(150)
    # Expected: 25/50=0.5, 50/100=0.5, 75/150=0.5
    
    tensor = scale_environmental_data(test_env)
    
    expected = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    assert torch.allclose(tensor, expected)

def test_iou_calculation():
    """Tests Intersection over Union for bounding boxes."""
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([5, 5, 15, 15])
    
    iou = calculate_iou(box1, box2)
    # intersection: [5,5,10,10] area=25
    # union: 100 + 100 - 25 = 175
    # expected: 25/175 = 0.1428
    
    assert abs(iou - 0.1428) < 1e-4

def test_classification_metrics():
    """Tests the precision/recall/f1 calculations."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1]) # 75% accuracy
    
    metrics = calculate_classification_metrics(y_true, y_pred)
    
    assert metrics["accuracy"] == 0.75
    assert "f1_macro" in metrics
    assert "precision_macro" in metrics

def test_config_defaults():
    """Verifies default configuration values."""
    cfg = Config()
    assert cfg.model.num_classes == 115
    assert cfg.training.batch_size == 32
    assert isinstance(cfg.data_path, str)
