# Jira Sprint Plan — AI-Driven Crop Disease Diagnosis

**Project Key:** CROP  
**Board Type:** Scrum  

---

## Sprint 1: Research & Foundation (Phase 1) — ✅ COMPLETED

> **Duration:** Week 1–4 | **Status:** Done

### Epic: Literature Survey & Gap Analysis
| Task | Type | Priority | Status |
|------|------|----------|--------|
| Review foundational CNN papers (Mohanty 2016, Lu 2017) | Task | High | ✅ Done |
| Survey transformer-based approaches (Swin, ViT, DETR) | Task | High | ✅ Done |
| Review segmentation methods (UNet, SAM, SAM2) | Task | High | ✅ Done |
| Review environmental fusion methods in agri-AI | Task | Medium | ✅ Done |
| Compile comparative analysis table (12 papers) | Task | High | ✅ Done |
| Document 5 research gaps | Task | High | ✅ Done |

### Epic: Algorithm Selection & Architecture Design
| Task | Type | Priority | Status |
|------|------|----------|--------|
| Evaluate backbone classifiers (ResNet vs Swin vs ViT) | Task | High | ✅ Done |
| Select Swin Transformer for Stage 1 (Classification) | Decision | High | ✅ Done |
| Select DETR for Stage 2 (Detection) | Decision | High | ✅ Done |
| Design SAM2-UNet hybrid for Stage 3 (Segmentation) | Task | High | ✅ Done |
| Design Environmental Fusion layer (GBM) | Task | Medium | ✅ Done |
| Create system architecture diagram | Task | Medium | ✅ Done |

### Epic: Dataset Curation
| Task | Type | Priority | Status |
|------|------|----------|--------|
| Download PlantVillage dataset (~87K images) | Task | High | ✅ Done |
| Identify 8 target crop species | Task | High | ✅ Done |
| Source PlantSeg segmentation dataset (11K) | Task | Medium | ✅ Done |
| Collect Mendeley Groundnut dataset (10K) | Task | Medium | ✅ Done |
| Source crop recommendation CSV (soil/weather features) | Task | Medium | ✅ Done |

### Epic: Mathematical Framework
| Task | Type | Priority | Status |
|------|------|----------|--------|
| Define multi-class Cross-Entropy loss for classification | Task | High | ✅ Done |
| Define weighted ensemble formula (αY_CNN + βY_GBM) | Task | High | ✅ Done |
| Select evaluation metrics (Dice, Top-5 Accuracy) | Task | Medium | ✅ Done |

### Epic: Project Setup
| Task | Type | Priority | Status |
|------|------|----------|--------|
| Initialize GitHub repository | Task | High | ✅ Done |
| Create directory structure (docs/, data/, models/, src/) | Task | Medium | ✅ Done |
| Write project README | Task | Medium | ✅ Done |
| Configure Jira board with sprints | Task | Medium | ✅ Done |
| Prepare Phase 1 Report | Task | High | ✅ Done |
| Create presentation slides | Task | High | ✅ Done |

---

## Sprint 2: Data & Baseline Training (Phase 2) — 🔜 UPCOMING

> **Duration:** Week 5–10 | **Status:** To Do

### Epic: Data Preprocessing & Augmentation
| Task | Type | Priority | Status |
|------|------|----------|--------|
| Clean and normalize PlantVillage images | Task | High | To Do |
| Apply augmentation (rotation, flip, color jitter) | Task | High | To Do |
| Create train/val/test splits (70/15/15) | Task | High | To Do |
| Generate semi-supervised segmentation masks via SAM2 | Task | Medium | To Do |

### Epic: Stage 1 — Classification Model Training
| Task | Type | Priority | Status |
|------|------|----------|--------|
| Fine-tune Swin Transformer on PlantVillage | Task | High | To Do |
| Implement environmental feature injection | Task | High | To Do |
| Achieve baseline accuracy ≥ 95% on validation set | Task | High | To Do |

### Epic: Stage 2 — Detection Model Training
| Task | Type | Priority | Status |
|------|------|----------|--------|
| Train DETR on annotated lesion bounding boxes | Task | High | To Do |
| Evaluate mAP on detection test set | Task | High | To Do |

---

## Sprint 3: Fusion, Deployment & Validation (Phase 3) — 📋 PLANNED

> **Duration:** Week 11–16 | **Status:** To Do

### Epic: Stage 3 — Segmentation
| Task | Type | Priority | Status |
|------|------|----------|--------|
| Train SAM2-UNet on segmentation masks | Task | High | To Do |
| Calculate Dice coefficient on test set | Task | High | To Do |

### Epic: Environmental Fusion & Ensemble
| Task | Type | Priority | Status |
|------|------|----------|--------|
| Train GBM on crop recommendation features | Task | High | To Do |
| Implement weighted ensemble (αY_CNN + βY_GBM) | Task | High | To Do |
| Tune α and β hyperparameters | Task | Medium | To Do |

### Epic: Deployment & Field Testing
| Task | Type | Priority | Status |
|------|------|----------|--------|
| Convert models to TensorFlow Lite | Task | High | To Do |
| Build Streamlit web dashboard | Task | High | To Do |
| Implement multilingual advisor (Hindi/Regional) | Task | Medium | To Do |
| Field validation with real crop images | Task | High | To Do |
| Write final project report | Task | High | To Do |

---

## How to Set Up This Board in Jira

### Step 1: Create a Free Jira Account
1. Go to [https://www.atlassian.com/software/jira/free](https://www.atlassian.com/software/jira/free)
2. Click **"Get it free"**
3. Sign up with your Amrita email

### Step 2: Create the Project
1. Click **"Create Project"**
2. Choose **Scrum** board template
3. Set project name: **AI-Driven Crop Disease Diagnosis**
4. Set project key: **CROP**

### Step 3: Create Epics
Create these 10 Epics (use the Roadmap/Timeline view):
1. Literature Survey & Gap Analysis
2. Algorithm Selection & Architecture Design
3. Dataset Curation
4. Mathematical Framework
5. Project Setup
6. Data Preprocessing & Augmentation
7. Stage 1 — Classification Training
8. Stage 2 — Detection Training
9. Stage 3 — Segmentation
10. Deployment & Field Testing

### Step 4: Create Tasks Under Each Epic
For each row in the tables above, create a **Task** (or **Story**) and:
- Link it to its parent **Epic**
- Set **Priority** (High / Medium)
- Set **Status** (Done for Sprint 1 tasks, To Do for others)

### Step 5: Create Sprints
1. Go to **Backlog** view
2. Click **"Create Sprint"** three times to create Sprint 1, Sprint 2, Sprint 3
3. Drag tasks into their respective sprints
4. For **Sprint 1** — click **"Complete Sprint"** to mark it as finished

### Step 6: Take a Screenshot
Once your board looks clean, take a screenshot showing:
- The **Board view** with Sprint 1 completed
- The **Roadmap/Timeline** view showing all 3 sprints
