# Phase 1 Report: AI-Driven Crop Disease Diagnosis

**Project Team:** Deemant, Rahul C V, Vishal Karthikeyan S S, Sricharan A, Sundar
**Guide:** Dr. Bagyammal T
**Target Year:** 2025–26 (B.Tech CSE)

---

## 1. Study and Motivation
Agriculture sustains the majority of the Indian population, yet crop diseases cause an estimated **$220 Billion** in annual global losses. Over **70 million smallholder farmers** lack access to expert plant pathology diagnostics. Traditional methods of manual inspection are slow, error-prone, and expert-dependent. This project aims to democratize expert-level plant disease diagnosis by providing an AI-driven tool robust enough for real-world field conditions.

### Objective
To build a multi-stage, multi-modal diagnostic pipeline that goes beyond simple classification to provide localization (where the disease is) and segmentation (how severe it is), while incorporating environmental context for higher accuracy.

---

## 2. Research Gaps & Domain Identification
Through an extensive review of existing literature, the following gaps were identified:
1.  **Single-Modality Dependence**: Most systems only use RGB images, missing critical thermal or biochemical cues.
2.  **Limited Crop Scope**: Existing models typically cover only 1–4 crops, while real-world farming often involves diverse species.
3.  **Environmental Neglect**: Temperature, humidity, and soil health (NPK) are major drivers of disease spread but are rarely integrated into AI predictions.
4.  **Field Domain Shift**: Models trained in lab settings (like PlantVillage) experience a **22.4% accuracy drop** in real fields.
5.  **Annotation Bottleneck**: Creating pixel-level masks for segmentation is extremely time-consuming (38.7 min/image).

---

## 3. Literature Survey & Benchmark Review
We surveyed 12 foundational and recent papers (2016–2025).

| Paper | Model | Multi-Crop | Env. Data | Segmentation | Field Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Mohanty (2016) | CNN (PlantVillage) | ❌ | ❌ | ❌ | ~77% |
| Lu (2017) | Deep CNN (Rice) | ❌ | ❌ | ❌ | 94.7% (Lab) |
| Iqbal (2025) | PlantHealthNet | ✅ | ❌ | Partial | 95.8% |
| **This Work** | **Swin+DETR+SAM2** | ✅ | ✅ | ✅ | **Target: ≥96%** |

---

## 4. Proposed Methodology & Technical Depth
The system employs a **3-Stage Intelligent Pipeline**:

### Stage 1: Classification (Swin Transformer)
Uses **Shifted-Window Attention** to capture both local textures (mildew, rust) and global leaf patterns. 
- *Novelty*: Injection of environmental features into the final representation.
- *Loss*: Large-scale multi-class Cross-Entropy for 115 disease classes.

### Stage 2: Object Detection (DETR - Detection Transformer)
Uses an end-to-end transformer decoder with object queries to localize infection sites without needing manual anchor tuning (NMS-free).
- *Novelty*: Handles overlapping lesions effectively, reducing annotation error by **41%**.

### Stage 3: Segmentation (SAM2-UNet)
Integrates Segment Anything Model 2 with UNet skip connections for zero-shot guided segmentation.
- *Novelty*: Semi-supervised learning reduces the need for pixel-labeled samples by **60%**, generating accurate severity maps.

### Environmental Fusion
A **Gradient Boosted Model (GBM)** processes Temperature, Humidity, and Soil N/P/K/pH. The final prediction is a weighted ensemble:
$$Y_{final} = \alpha Y_{CNN} + \beta Y_{GBM}$$

---

## 5. Technical Knowledge & Stack
- **Frameworks**: PyTorch, HuggingFace, Meta AI's SAM2.
- **Datasets**: PlantVillage (87K), PlantSeg (11K), Mendeley Groundnut (10K), and Indian Crop datasets.
- **Evaluation**: NIST framework benchmarks, Dice Coefficient (Segmentation), and Top-5 Accuracy (Classification).

---

## 6. Current Status & Setup
- **Version Control**: Git repository initialized with 3-stage directory structure.
- **Task Management**: Project roadmap defined with scheduled sprints for Data Collection, Model Training, and Deployment.
- **Baseline**: Literature survey and mathematical framework completed.

---

## 7. Expected Outcomes
- High-precision diagnosis across **8 crop species** and **115+ classes**.
- **Real-time severity assessment** (Percentage of leaf area infected).
- **Multilingual advisor** (Hindi, Regional) providing treatment recommendations.
- **Edge Deployment** for low-connectivity rural environments.
