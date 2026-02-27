import torch
import torch.nn as nn
from transformers import SwinModel, SwinConfig
from src.utils.logger import logger

class CropDiseaseClassifier(nn.Module):
    """
    Stage 1: Classification model using Swin Transformer.
    Integrates environmental features for robust crop disease diagnosis.
    """
    def __init__(self, model_name: str = "microsoft/swin-tiny-patch4-window7-224", 
                 num_classes: int = 115, 
                 env_feature_dim: int = 5):
        super(CropDiseaseClassifier, self).__init__()
        
        logger.info(f"Initializing Swin Transformer: {model_name}")
        self.swin = SwinModel.from_pretrained(model_name)
        
        # Swin hidden size (tiny is 768)
        self.hidden_size = self.swin.config.hidden_size
        
        # Environmental feature processing
        self.env_fc = nn.Sequential(
            nn.Linear(env_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion and Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values: torch.Tensor, env_features: torch.Tensor):
        """
        Forward pass with image and environmental data fusion.
        """
        outputs = self.swin(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output # (batch_size, hidden_size)
        
        env_processed = self.env_fc(env_features) # (batch_size, 64)
        
        # Fusion: Concatenation
        combined = torch.cat((pooled_output, env_processed), dim=1)
        
        logits = self.classifier(combined)
        return logits

if __name__ == "__main__":
    # Test initialization
    model = CropDiseaseClassifier()
    logger.info("Model template successfully initialized.")
