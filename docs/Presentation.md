# AI-Driven Crop Disease Diagnosis: Helping Indian Farmers
**Panel Review 1 | B.Tech CSE | 2025–26**

---

## Project Team
- **Deemant** (CB.SC.U4CSE23715)
- **Rahul C V** (CB.SC.U4CSE23740)
- **Vishal Karthikeyan S S** (CB.SC.U4CSE23756)
- **Sricharan A** (CB.SC.U4CSE23764)
- **Sundar** (CB.SC.U4CSE23348)

**Guide:** Dr. Bagyammal T, Asst. Prof., Dept. of CSE, Amrita Vishwa Vidyapeetham

---

## 1. Problem Statement & Motivation
### The Global Crisis
- **$220 Billion** annual global crop losses from plant diseases.
- **70 Million+** smallholder farmers lack expert diagnostics.
- Pathenogens cover **115+ disease classes** across fungal, bacterial, and viral types.
- Manual inspection is slow, expert-dependent, and prone to error.

### The Research Gap
Most existing AI systems are **single-modal** (RGB only), ignore environmental context, and suffer a **22.4% accuracy drop** in real-world field conditions compared to lab environments.

---

## 2. Literature Survey (2016–2025)
### Foundational & Recent Advances
- **Mohanty (2016)**: CNN on PlantVillage. Achieved 99.35% lab accuracy but dropped to 77% in the field.
- **Lu (2017)**: Deep CNN for rice; limited to a single crop.
- **Iqbal (2025)**: PlantHealthNet. Transformer-enhanced hybrid; lacks pixel-level segmentation.
- **This Work**: Unifies **Swin Transformer + DETR + SAM2** with environmental fusion for 8 crop species.

### Comparative Analysis
| Paper | Model | Multi-Crop | Env. Data | Segmentation | Field Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Mohanty 2016 | CNN | ❌ | ❌ | ❌ | ~77% |
| Lu 2017 | Deep CNN | ❌ | ❌ | ❌ | 94.7% (Lab) |
| Iqbal 2025 | PlantHealthNet | ✅ | ❌ | Partial | 95.8% |
| **Ours** | **Swin+DETR+SAM2** | **✅** | **✅** | **✅** | **Target: ≥96%** |

---

## 3. Core Research Gaps Addressed
1. **Single-Modality**: Moving beyond RGB-only to include environmental biochemical cues.
2. **Limited Crop Diversity**: Covering 8 major crops simultaneously (Corn, Rice, wheat, Tomato, etc.).
3. **No Environmental Context**: Integrating Temperature, Humidity, and Soil NPK/pH directly into the model.
4. **Domain Shift**: Explicitly training for robustness in varied field lighting and conditions.
5. **Annotation Bottleneck**: Using SAM2 to reduce pixel-level labeling effort by 60%.

---

## 4. Proposed Architecture: Three-Stage Pipeline

### Stage 1: Classification (Swin Transformer)
- Captures local textures and global leaf patterns via Shifted-Window Attention.
- **Novelty**: Injection of environmental features into the classifier head.

### Stage 2: Object Detection (DETR)
- End-to-end transformer for simultaneous multi-lesion detection.
- **Benefit**: 41% reduction in annotation error vs. traditional CNN detectors.

### Stage 3: Segmentation (SAM2-UNet)
- Pixel-level severity mapping for precise dosage recommendations.
- **Benefit**: Semi-supervised learning significantly saves expert labeling time.

### Environmental Fusion Layer
A **Gradient Boosted Model (GBM)** processes:
- Ambient Temperature & Humidity
- Cumulative Rainfall
- Soil Nitrogen, Phosphorus, Potassium (NPK) & pH

---

## 5. Technical Stack
- **Frameworks**: PyTorch, HuggingFace, SAM2 (Meta AI).
- **Core Models**: Swin Transformer (backbone), DETR (detector), UNet (decoder).
- **Tooling**: Google Colab Pro+ (A100 GPUs), GitHub (Version Control), Jira (Task Tracking).
- **Deployment**: TensorFlow Lite for edge/mobile, Streamlit for the web dashboard.

---

## 6. Project Roadmap
### Phase 1 (Completed)
- Literature survey and gap analysis.
- Algorithm selection and mathematical architecture.
- Baseline repository and task board setup.

### Phase 2 (Upcoming)
- Large-scale dataset curation and semi-supervised annotation.
- Baseline training for Stage 1 and Stage 2.

### Phase 3 & 4
- Environmental fusion layer integration.
- Mobile deployment and field validation in Indian conditions.

---

## 7. Expected Outcomes
- **High Accuracy**: Targets >96% diagnostic accuracy across 115 disease classes.
- **Severity Mapping**: Real-time assessment of infection percentage.
- **Multilingual Support**: Advisor providing treatment in Hindi and regional languages.
- **Economic Impact**: Enabling early intervention for smallholder farmers.
