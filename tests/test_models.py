import torch
import pytest
from src.classification.model import CropDiseaseClassifier
from src.classification.loss import FocalLoss

def test_model_initialization():
    """Tests that the model initializes with default parameters."""
    model = CropDiseaseClassifier(num_classes=115, env_feature_dim=5)
    assert isinstance(model, torch.nn.Module)
    assert model.hidden_size == 768  # For Swin-Tiny

def test_model_forward_pass():
    """Tests the model forward pass with dummy data."""
    batch_size = 2
    num_classes = 10
    env_dim = 5
    
    model = CropDiseaseClassifier(num_classes=num_classes, env_feature_dim=env_dim)
    model.eval()

    dummy_pixels = torch.randn(batch_size, 3, 224, 224)
    dummy_env = torch.randn(batch_size, env_dim)

    with torch.no_grad():
        logits = model(dummy_pixels, dummy_env)

    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()

def test_focal_loss():
    """Tests the Focal Loss calculation."""
    batch_size = 4
    num_classes = 5
    
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    
    inputs = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    loss = criterion(inputs, targets)
    
    assert loss.dim() == 0  # Scaler
    assert loss >= 0
    
    loss.backward()
    assert inputs.grad is not None
