import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from src.utils.logger import logger

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

    @classmethod
    def load(cls, path: Optional[str] = None):
        """Loads configuration from a YAML file."""
        if path and Path(path).exists():
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            
            # Simple recursive update for dataclasses
            model_data = data.get("model", {})
            train_data = data.get("training", {})
            
            return cls(
                model=ModelConfig(**model_data),
                training=TrainingConfig(**train_data),
                data_path=data.get("data_path", "data/plantvillage dataset"),
                output_dir=data.get("output_dir", "outputs")
            )
        
        logger.warning("Config file not found or path not provided. Using defaults.")
        return cls()

    def save(self, path: str):
        """Saves current configuration to a YAML file."""
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
        logger.info(f"Configuration saved to {path}")

# Default config instance
cfg = Config.load()
