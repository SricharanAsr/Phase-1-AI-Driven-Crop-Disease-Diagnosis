# Project Architecture: AI-Driven Crop Disease Diagnosis

This document visualizes the multi-modal, three-stage intelligent diagnostic pipeline designed for robust field-condition crop disease assessment.

## High-Level System Architecture

![Architecture Diagram](Architecture_Diagram.jpeg)

*Proposed Hybrid CNN-Transformer Architecture for Crop Disease Detection*

## Logical Workflow Diagram (Mermaid)

```mermaid
graph TD
    subgraph "Input Layer"
        IMG["RGB Image (Crop Leaf)"]
        ENV["Environmental Data<br/>(Temp, Humidity, Rainfall, Soil NPK/pH)"]
    end

    subgraph "Visual Diagnostic Pipeline"
        direction TB
        S1["<b>Stage 1: Classification</b><br/>Swin Transformer<br/>(115 Disease Classes)"]
        S2["<b>Stage 2: Detection</b><br/>DETR (Detection Transformer)<br/>(Lesion Localization)"]
        S3["<b>Stage 3: Segmentation</b><br/>SAM2-UNet<br/>(Pixel-level Severity)"]
        
        S1 --> S2
        S2 --> S3
    end

    subgraph "Environmental Analysis Pipeline"
        GBM["Contextual Embedding Layer<br/>(Normalizes Environmental Metrics)"]
    end

    subgraph "Intelligent Fusion Layer"
        FUSION{"Multi-Modal<br/>Feature Fusion"}
        
        S1 --> FUSION
        GBM --> FUSION
    end

    subgraph "Output & Decision Support"
        DIAG["Final Diagnosis"]
        MAP["Severity Heatmap (%)"]
        ADVISOR["Diagnostic Report &<br/>Treatment Recommendations"]
        
        FUSION --> DIAG
        S3 --> MAP
        DIAG --> ADVISOR
    end

    IMG --> S1
    ENV --> GBM

    style S1 fill:#f9f,stroke:#333,stroke-width:2px
    style S2 fill:#bbf,stroke:#333,stroke-width:2px
    style S3 fill:#bfb,stroke:#333,stroke-width:2px
    style GBM fill:#fdb,stroke:#333,stroke-width:2px
    style FUSION fill:#fff,stroke:#333,stroke-width:4px
```

## Technical Component Breakdown

### 1. Stage 1: Hierarchical Classification
- **Backbone**: `microsoft/swin-tiny-patch4-window7-224`
- **Mechanism**: Utilizes Shifted-Window Attention to capture multi-scale features, identifying 115 disease classes across 8 crop species.
- **Environmental Fusion**: Integrates scaled environmental tensors (T, H, N, P, K) to resolve visual ambiguities between nutrient deficiencies and early-stage infections.

### 2. Stage 2: Transformer-based Detection (DETR)
- **Model**: Detection Transformer (DETR).
- **Function**: Performs set-based object detection for lesion localization. This stage identifies multiple infection points on a single plant organ, providing spatial context for the severity analysis.

### 3. Stage 3: Severity Segmentation (SAM2-UNet)
- **Model**: Segment Anything Model 2 (SAM2) integrated with UNet skip connections.
- **Function**: Generates high-resolution binary masks of infected tissues.
- **Metric**: Calculates the **Disease Severity Index (DSI)** as the ratio of infected pixels to total leaf pixels.

### 4. Decision Support System
- **Output**: Generates a comprehensive PDF report including disease identification, localized severity map, and treatment advisor.
- **Language Support**: Designed for multilingual accessibility to support diverse agricultural communities in India.
