# Data Directory

This directory contains the datasets used for training and evaluating the AI-driven crop disease diagnosis pipeline.

## Structure

- `plantvillage-dataset.zip`: Original compressed dataset from PlantVillage.
- `plantvillage dataset/`: Extracted images structured by crop and disease class.
- `Crop_recommendation.csv`: Environmental data for the Gradient Boosted Model (GBM) fusion.

## Planned Datasets

1.  **PlantVillage (Primary)**: 87,000+ images across 38 classes.
2.  **PlantSeg**: 11,000+ pixel-labeled images for segmentation.
3.  **Mendeley Groundnut**: Specialized dataset for groundnut diseases.
4.  **Indian Crop Datasets**: Real-world field images to mitigate field domain shift.

## Data Pipelining

Raw data should be preprocessed and placed in a `processed/` subdirectory (ignored by Git) for model training.
