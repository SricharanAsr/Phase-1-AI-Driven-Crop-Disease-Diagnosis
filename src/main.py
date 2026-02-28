import torch
from src.utils.logger import logger
from src.utils.config import cfg
from src.classification.model import CropDiseaseClassifier
from src.detection.model import InfectionDetector
from src.segmentation.model import SAM2UNetSegmenter
from src.utils.preprocess import scale_environmental_data

def run_pipeline():
    """
    Demonstrates the initialization of the 3-stage diagnostic pipeline.
    """
    logger.info("Starting AI-Driven Crop Disease Diagnosis Pipeline")
    logger.info(f"Using Configuration: {cfg.model.name}, Batch Size: {cfg.training.batch_size}")

    # 1. Initialize Stage 1: Classification
    logger.info("Loading Stage 1: Classification Model...")
    classifier = CropDiseaseClassifier(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        env_feature_dim=cfg.model.env_feature_dim
    )

    # 2. Initialize Stage 2: Detection
    logger.info("Loading Stage 2: Infection Detection Model...")
    detector = InfectionDetector()

    # 3. Initialize Stage 3: Segmentation
    logger.info("Loading Stage 3: Severity Segmentation Model...")
    segmenter = SAM2UNetSegmenter()

    # Simple Demonstration with Dummy Data
    logger.info("Pipeline initialized. Ready for inference.")
    
    # Dummy Environmental Data
    dummy_env = {"temperature": 32.0, "humidity": 80.0, "n": 45, "p": 40, "k": 50}
    env_tensor = scale_environmental_data(dummy_env)
    logger.info(f"Processed Environmental Context: {env_tensor.shape}")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.critical(f"Pipeline failure: {e}")
