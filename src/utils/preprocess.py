import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Tuple, Union

def get_image_transforms(size: Tuple[int, int] = (224, 224)) -> T.Compose:
    """
    Standard transforms for crop disease images.
    """
    return T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path: str, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Loads and preprocesses an image for model input.
    """
    img = Image.open(image_path).convert("RGB")
    transforms = get_image_transforms(size)
    return transforms(img).unsqueeze(0) # Add batch dimension

def scale_environmental_data(env_dict: dict) -> torch.Tensor:
    """
    Scales environmental data (Temp, Humidity, N, P, K) to a normalized tensor.
    Placeholder for actual scaling logic.
    """
    # Assuming input order: [Temp, Humidity, N, P, K]
    data = [
        env_dict.get("temperature", 25.0),
        env_dict.get("humidity", 60.0),
        env_dict.get("n", 50.0),
        env_dict.get("p", 50.0),
        env_dict.get("k", 50.0)
    ]
    
    # Simple normalization placeholder (min-max or Z-score would be better)
    data_np = np.array(data, dtype=np.float32) / 100.0
    return torch.tensor(data_np).unsqueeze(0)

if __name__ == "__main__":
    # Test tensor generation
    test_env = {"temperature": 28.5, "humidity": 75.0, "n": 40, "p": 30, "k": 45}
    env_tensor = scale_environmental_data(test_env)
    print(f"Environmental Tensor: {env_tensor}")
