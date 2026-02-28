import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrConfig
from src.utils.logger import logger

class InfectionDetector(nn.Module):
    """
    Stage 2: Infection localization using DETR (Detection Transformer).
    Identifies bounding boxes for lesions and diseased areas on the crop.
    """
    def __init__(self, model_name: str = "facebook/detr-resnet-50", 
                 num_labels: int = 1):
        super(InfectionDetector, self).__init__()
        
        logger.info(f"Initializing DETR Detector: {model_name}")
        # Initialize with specific number of labels (default 1 for 'infection')
        self.detr = DetrForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor = None):
        """
        Forward pass for object detection.
        Returns bounding boxes and class logits.
        """
        outputs = self.detr(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs.logits, outputs.pred_boxes

if __name__ == "__main__":
    # Test initialization
    model = InfectionDetector()
    logger.info("DETR Model template successfully initialized.")
