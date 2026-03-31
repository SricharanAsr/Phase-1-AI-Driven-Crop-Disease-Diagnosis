import torch
import torch.nn as nn
from transformers import SwinModel, SwinConfig
from src.utils.logger import logger

from typing import Optional, Dict

class CropDiseaseClassifier(nn.Module):
    """Stage 1: Classification model using Swin Transformer with environmental fusion.

    This model integrates hierarchical vision features from a Swin Transformer
    with environmental context (temperature, humidity, soil metrics) to provide
    robust crop disease diagnosis in varied field conditions.

    Attributes:
        swin (SwinModel): Pre-trained Swin Transformer backbone.
        hidden_size (int): Hidden dimension size of the transformer.
        env_fc (nn.Sequential): Processing layers for environmental features.
        classifier (nn.Sequential): Fusion and classification head.
    """
    def __init__(
        self, 
        model_name: str = "microsoft/swin-tiny-patch4-window7-224", 
        num_classes: int = 115, 
        env_feature_dim: int = 5
    ):
        """Initializes the CropDiseaseClassifier.

        Args:
            model_name (str): HuggingFace model identifier for Swin Transformer.
            num_classes (int): Number of diagnostic disease classes.
            env_feature_dim (int): Dimension of the environmental feature vector.
        """
        super(CropDiseaseClassifier, self).__init__()
        
        logger.info(f"Initializing Swin Transformer backbone: {model_name}")
        self.swin = SwinModel.from_pretrained(model_name)
        
        # Swin hidden size (tiny: 768, small: 768, base: 1024)
        self.hidden_size = self.swin.config.hidden_size
        
        # Environmental feature processing (Contextual MLP)
        self.env_fc = nn.Sequential(
            nn.Linear(env_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion and Classification Head
        # Combines visual features (hidden_size) with environmental features (64)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values: torch.Tensor, env_features: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass with multi-modal data fusion.

        Args:
            pixel_values (torch.Tensor): Preprocessed image tensors of shape (batch, 3, H, W).
            env_features (torch.Tensor): Scaled environmental data of shape (batch, env_dim).

        Returns:
            torch.Tensor: Logits for each disease class of shape (batch, num_classes).
        """
        outputs = self.swin(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        
        env_processed = self.env_fc(env_features)  # (batch_size, 64)
        
        # Fusion: Late concatenation of visual and environmental embeddings
        combined = torch.cat((pooled_output, env_processed), dim=1)
        
        logits = self.classifier(combined)
        return logits

if __name__ == "__main__":
    # Test initialization with default parameters
    try:
        model = CropDiseaseClassifier()
        logger.info("Model template successfully initialized for Stage 1.")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
