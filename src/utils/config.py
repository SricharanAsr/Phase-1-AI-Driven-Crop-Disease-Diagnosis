import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from src.utils.logger import logger

import os

@dataclass
class ModelConfig:
    name: str = "microsoft/swin-tiny-patch4-window7-224"
    num_classes: int = 115
    env_feature_dim: int = 5

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 2e-5
    epochs: int = 50
    weight_decay: float = 0.01

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_path: str = "data/plantvillage dataset"
    output_dir: str = "outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]):
        """Recursively create Config from dictionary."""
        model_data = data.get("model", {})
        training_data = data.get("training", {})
        
        return cls(
            model=ModelConfig(**model_data) if model_data else ModelConfig(),
            training=TrainingConfig(**training_data) if training_data else TrainingConfig(),
            data_path=data.get("data_path", "data/plantvillage dataset"),
            output_dir=data.get("output_dir", "outputs"),
            device=data.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

    @classmethod
    def load(cls, path: Optional[str] = "config.yaml"):
        """Loads configuration from a YAML file with environment variable overrides."""
        config_instance = cls()
        
        if path and Path(path).exists():
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                if data:
                    config_instance = cls._from_dict(data)
                logger.info(f"Configuration loaded from {path}")
            except Exception as e:
                logger.error(f"Failed to load config from {path}: {e}. Using defaults.")
        else:
            if path != "config.yaml":
                logger.warning(f"Config file {path} not found. Using defaults.")

        # Environment Variable Overrides (Prefix: CROP_)
        config_instance.data_path = os.getenv("CROP_DATA_PATH", config_instance.data_path)
        config_instance.output_dir = os.getenv("CROP_OUTPUT_DIR", config_instance.output_dir)
        
        return config_instance

    def save(self, path: str = "config.yaml"):
        """Saves current configuration to a YAML file."""
        try:
            Path(path).parent.mkdir(exist_ok=True, parents=True)
            with open(path, "w") as f:
                yaml.dump(asdict(self), f, default_flow_style=False)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save config to {path}: {e}")

# Global config instance initialized from default path
cfg = Config.load()
