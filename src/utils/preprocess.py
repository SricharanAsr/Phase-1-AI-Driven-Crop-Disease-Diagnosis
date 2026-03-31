import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Tuple, Union, Dict
from src.utils.logger import logger

def get_train_transforms(size: Tuple[int, int] = (224, 224)) -> T.Compose:
    """Advanced augmentations for robust training in field conditions."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(size: Tuple[int, int] = (224, 224)) -> T.Compose:
    """Standard normalization for validation and inference."""
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def scale_environmental_data(env_dict: Dict[str, float]) -> torch.Tensor:
    """Standardizes environmental context using agriculture-specific ranges.

    Metrics handle:
    - Temperature (0-50 C)
    - Humidity (0-100 %)
    - Nitrogen, Phosphorus, Potassium (0-150 mg/kg)
    """
    # Agriculture-specific Max Ranges for Normalization
    ranges = {
        "temperature": 50.0,
        "humidity": 100.0,
        "n": 150.0,
        "p": 150.0,
        "k": 150.0
    }
    
    data = [
        env_dict.get("temperature", 25.0) / ranges["temperature"],
        env_dict.get("humidity", 60.0) / ranges["humidity"],
        env_dict.get("n", 50.0) / ranges["n"],
        env_dict.get("p", 50.0) / ranges["p"],
        env_dict.get("k", 50.0) / ranges["k"]
    ]
    
    # Clamp values between 0 and 1 to handle outliers
    data_np = np.clip(np.array(data, dtype=np.float32), 0.0, 1.0)
    return torch.tensor(data_np).unsqueeze(0)

if __name__ == "__main__":
    # Test tensor generation with edge cases
    test_env = {"temperature": 32.5, "humidity": 85.0, "n": 120, "p": 20, "k": 45}
    env_tensor = scale_environmental_data(test_env)
    logger.info(f"Normalized Environmental Context: {env_tensor}")
