# KNEEMRI ACL INJURY CLASSIFICATION (KneeMRI-AAN)
## FINAL PRODUCT REQUIREMENTS DOCUMENT
**Notebook-First, Explainable Deep-Learning Research Prototype with Attention-Based Slice Aggregation**  
*Version 3.0 • Updated October 2026*

---

## Document Control

| Field | Decision / Specification |
| :--- | :--- |
| **Project Title** | KneeMRI-AAN: Attention-Based Slice Aggregation Network for Knee MRI ACL Injury Classification |
| **Primary Deliverable** | Clean, self-contained, reproducible IPython/Jupyter Notebook (`notebooks/KneeMRI_ACL_Classification.ipynb`) |
| **Dataset** | KneeMRI — Official Zenodo dataset (917 volumes, 12-bit grayscale MRI, proton-density fat-suppressed) |
| **Clinical Task** | 3-Class ACL Injury Classification (0: Healthy, 1: Partially Injured, 2: Completely Ruptured) |
| **Research Question** | *“Does learned slice weighting improve volume-level ACL classification compared with treating all selected MRI slices equally?”* |
| **Baseline Architecture** | ImageNet-pretrained ResNet-18 with uniform Mean Slice Pooling (E0) |
| **Proposed Architecture** | ImageNet-pretrained ResNet-18 with Lightweight Attention-Based Slice Pooling (E1) |
| **Input Strategy** | $K$ ROI-aware informative 2D slices extracted from each 3D MRI volume ($K$ configured as 7) |
| **Primary Model Selection Metric** | **Validation Macro-F1** (Mandated due to severe 690 / 172 / 55 class imbalance) |
| **Explainability** | Grad-CAM feature attribution + Learned Slice Attention Weight Visualizations |
| **Compute Target** | NVIDIA GeForce RTX 4050 Laptop GPU (CUDA enabled); core experiment suite target $\\le 5$ hours |
| **Training Control** | Mandatory full-state atomic checkpointing with seamless pause/resume capability |
| **Optional Interface** | Local Streamlit demonstration app (`app/streamlit_app.py`) after notebook verification |
| **Project Nature** | Academic/Research prototype; rigorously structured, simple & explainable code for vivas |

---

## 1. Executive Summary

This Product Requirements Document (PRD) establishes the architecture, experimental framework, and engineering protocols for **KneeMRI-AAN**, an end-to-end, reproducible deep learning system for automated Anterior Cruciate Ligament (ACL) tear assessment from volumetric magnetic resonance imaging (MRI).

Rather than deploying an opaque black-box or computationally prohibitive 3D volumetric model, this project addresses a fundamental clinical question: **Which MRI slices actually show the tear, and can a neural network learn to prioritize those slices automatically?**

The pipeline extracts $K$ informative slices per MRI volume, extracts deep visual features using a shared pretrained **ResNet-18** backbone, learns slice relevance via a **lightweight slice-attention mechanism**, and aggregates features into a 3-class volume-level prediction. The baseline model employs standard uniform mean pooling across slices, providing an exact controlled reference to evaluate whether learned slice weighting delivers statistically superior diagnostic performance.

The project strictly abides by an RTX 4050 Laptop GPU budget ($\le 5$ hours), enforces strict patient-level split isolation to eliminate data leakage, employs class-weighted objectives to address the 690/172/55 class imbalance, and provides Grad-CAM visual explanations and full-state pause/resume checkpointing.

---

## 2. Problem Statement & Research Foundation

Given a 3D knee MRI scan comprising 32 to 36 cross-sectional slices, determine whether the ACL is **Healthy (Class 0)**, **Partially Injured (Class 1)**, or **Completely Ruptured (Class 2)**.

### 2.1 Primary Research Question
> **“Does learned slice weighting improve volume-level ACL classification compared with treating all selected MRI slices equally?”**

### 2.2 Literature & Academic Justification
1. **2D CNN vs. 3D CNN for MRI**: Direct 3D convolutional networks require massive voxel parameters and suffer severe overfitting when trained on small medical cohorts (e.g., 917 volumes). Furthermore, ImageNet pretraining cannot be directly transferred to 3D kernels without heuristic weight inflation. 2D CNN backbones leverage rich pretrained natural image representations, converge rapidly, and require orders of magnitude less GPU memory.
2. **ResNet-18 as a Diagnostic Backbone**: Published benchmarks in musculoskeletal imaging (notably the Stanford MRNet study by Bien et al., 2018) established ResNet-18 as the premier trade-off between representational capacity, generalization on small medical datasets, and training stability.
3. **Learned Inter-Slice Weighting (Attention-based MIL)**: The diagnostic signature of an ACL rupture is localized to a small subset of slices intersecting the intercondylar notch, while adjacent slices capture distal bone or soft tissue. In classical Multi-Instance Learning (MIL) (Ilse et al., 2018), uniform mean pooling dilutes pathological signals with uninformative slices, while max-pooling is susceptible to outlier artifacts. A lightweight attention mechanism learns a convex combination of slice representations, providing both improved classification accuracy and intrinsic slice-level clinical interpretability.

### 2.3 Non-Clinical Positioning
This system is an academic research prototype. It is explicitly not designed, certified, or intended for clinical diagnosis, patient triage, or direct medical treatment decisions.

---

## 3. Dataset Specification

| Attribute | Specification |
| :--- | :--- |
| **Dataset Source** | Official KneeMRI Zenodo repository (Stajduhar et al.) |
| **Total Volumes** | 917 knee MRI examinations |
| **Unique Patients/Exams** | 909 unique exam IDs (8 duplicate exams containing multiple series) |
| **Image Representation** | 12-bit grayscale DICOM-derived arrays stored as `.pck` (pickle) files |
| **Volume Dimensions** | Typically $(32, 320, 320)$ voxels per examination |
| **Acquisition Protocol** | Siemens Avanto 1.5T scanner; sagittal proton-density (PD) weighted with fat suppression |
| **Class Distribution** | **Class 0 (Healthy):** 690 volumes (75.2%)<br>**Class 1 (Partial Tear):** 172 volumes (18.8%)<br>**Class 2 (Complete Tear):** 55 volumes (6.0%) |
| **Bounding Box Metadata** | `roiX`, `roiY`, `roiZ`, `roiHeight`, `roiWidth`, `roiDepth` providing 3D ground-truth radiologist localization |
| **Storage Structure** | `metadata.csv` mapped to pickled volume files in `volumetric_data/volumetric_data/` |
| **Dataset License** | Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0) |

---

## 4. Architecture Selection Rationale

To ensure academic defensibility during technical review and viva examinations, the selection of **ResNet-18** as the core feature extractor is justified against candidate architectures across six criteria:

| Architecture | Parameters | ImageNet Top-1 | GPU Memory (Batch 16) | Interpretability (Grad-CAM) | Syllabus Alignment | Feasibility on RTX 4050 (≤5h) | Selection Decision & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AlexNet** | 61.1M | 56.5% | Moderate | Coarse | Foundational | Yes | **Rejected**: Oversized FC layers (4096), low representational power, prone to severe overfitting on medical images. |
| **VGG-16 / VGG-19** | 138.4M | 71.5% | Very High (>5 GB) | Good | Classical deep CNN | Marginal (Slow) | **Rejected**: Excessive parameter volume, slow backward pass, high VRAM usage limits slice batching. |
| **GoogLeNet / Inception-v3** | 23.8M | 77.3% | High | Complex (Multi-branch) | Advanced CNN | Yes | **Rejected**: Multi-branch concatenated receptive fields make layer attribution and Grad-CAM hooks difficult to explain cleanly. |
| **DenseNet-121** | 7.98M | 75.0% | High (Feature maps) | Moderate | Dense connections | Yes | **Rejected**: Memory footprint expands rapidly during training due to concatenated dense feature buffers. |
| **EfficientNet-B0** | 5.3M | 77.1% | Low | Moderate | Compound scaling | Yes | **Rejected**: Inverted residual blocks with depthwise convolutions train slower on small GPU batch regimes and are more difficult to teach/explain in viva than standard residual blocks. |
| **ResNet-18 (Selected)** | **11.2M** | **69.8%** | **Low (<1.5 GB)** | **Excellent (Layer4)** | **Core Syllabus** | **Ideal (~1.5–2h)** | **SELECTED**: Optimal depth, identity skip connections resolve vanishing gradients, highly parameter-efficient, fast GPU throughput, transparent Grad-CAM localization, matches syllabus. |
| **ResNet-50** | 25.6M | 76.1% | Moderate | Excellent | Bottleneck blocks | Feasible (~3.5h) | **Rejected for Baseline**: 3-layer bottleneck blocks overparameterize on 917 samples, increasing overfitting risk on minority classes. |
| **3D CNN (ResNet3D)** | 33.2M | N/A (MedicalNet) | Extreme (>6 GB) | Difficult | Volumetric CNN | No (Out of budget) | **Rejected**: Lacks robust ImageNet pretraining, requires heavy spatial-temporal convolutions, risks memory OOM, training takes >8 hours. |
| **Vision Transformer (ViT)**| 86.6M | 81.8% | High | Attention maps | Transformers | No | **Rejected**: Demands hundreds of thousands of samples to learn spatial inductive biases; completely overfits on 917 medical cases without massive pretraining. |

---

## 5. System Architecture & Proposed Pipeline

### 5.1 Proposed Pipeline (KneeMRI-AAN)
```
Raw Knee MRI Volume (32×320×320)
       │
       ▼
ROI-Aware Preprocessing & Normalization (12-bit uint16 → [0, 1] float32)
       │
       ▼
Informative Slice Selection (K=7 slices centered around anatomical ROI)
       │
       ▼
Shared Pretrained ResNet-18 Backbone (Weights frozen/partially tuned)
       │
       ▼
Slice Feature Representations: H = [h_1, h_2, ..., h_K] where h_k ∈ R^512
       │
       ▼
Lightweight Inter-Slice Attention Mechanism:
  Score:   s_k = w^T tanh(V h_k)    (w ∈ R^64, V ∈ R^(64×512))
  Weights: a_k = exp(s_k) / ∑_j exp(s_j)
       │
       ▼
Volume-Level Feature Aggregation:
  h_vol = ∑_{k=1}^K a_k h_k   (Convex combination in R^512)
       │
       ▼
Fully Connected Classification Head (Linear(512, 128) → ReLU → Dropout(0.3) → Linear(128, 3))
       │
       ▼
3-Class Softmax Probabilities: [P(Healthy), P(Partial), P(Complete)]
       │
       ▼
Grad-CAM Saliency Maps & Attention Slice Ranking (Explainability)
```

### 5.2 Baseline Architecture (E0 Reference)
In the baseline model, slice features are aggregated uniformly without attention:
$$h_{vol}^{\text{baseline}} = \frac{1}{K} \sum_{k=1}^K h_k$$
This enables a direct scientific comparison: any performance delta between E0 and E1 is strictly attributable to **learned inter-slice attention**.

### 5.3 Optional 2.5D Extension (E6)
Rather than passing single 2D slices repeated across 3 channels, the optional 2.5D variant stacks three adjacent anatomical slices $(z-1, z, z+1)$ into the RGB channels of the ResNet-18 input, providing local 3D spatial continuity while preserving 2D convolutional efficiency.

---

## 6. Data Leakage Prevention — Critical Protocol

Patient-level data leakage produces falsely inflated validation metrics and invalidates medical machine learning models. The following safeguards are mandatory:

1. **Patient/Exam Grouping**: The dataset contains 917 volumes across 909 unique examinations. All volumes sharing an `examId` **must reside in the exact same split**.
2. **Split Creation Before Preprocessing**: Splitting occurs strictly at the exam level on `metadata.csv` prior to any slice extraction, image cropping, or data augmentation.
3. **Stratification**: The split is stratified according to the volume-level `aclDiagnosis` label:
   * **Train Set:** ~70% (~641 volumes)
   * **Validation Set:** ~15% (~138 volumes)
   * **Test Set:** ~15% (~138 volumes)
4. **Persistent Split File**: The split assignments are permanently stored in `splits/volume_split.csv` to ensure 100% reproducibility across all notebook restarts.
5. **Augmentation Isolation**: Data augmentations (random affine transformations, mild intensity shifts) are applied **exclusively to the training split during training**. Validation and test volumes undergo strictly deterministic center cropping and normalization.

---

## 7. Preprocessing & Slice Selection Pipeline

1. **Volume Loading**: Safely load `.pck` files from disk, verifying uint16 intensity range and dimensions.
2. **Min-Max & Percentile Normalization**: Robustly normalize 12-bit MRI signals:
   $$I_{\text{norm}} = \text{clip}\left(\frac{I - P_1}{P_{99} - P_1}, 0, 1\right)$$
   where $P_1$ and $P_{99}$ are the 1st and 99th intensity percentiles.
3. **ROI Extraction**: Utilize `roiX, roiY, roiWidth, roiHeight` to crop the joint space containing the cruciate ligament, adding an anatomical context margin (30% padding) to avoid edge clipping.
4. **Informative Slice Selection ($K=7$)**: Slices are extracted symmetrically around the anatomical center:
   $$z_{\text{center}} = \text{roiZ} + \left\lfloor \frac{\text{roiDepth}}{2} \right\rfloor$$
   Selecting slices in the range $[z_{\text{center}} - 3, z_{\text{center}} + 3]$ captures the full course of the ligament from the femoral attachment to the tibial plateau.
5. **Standardized Input Resizing**: Each slice is resized to $(224, 224)$ pixels and duplicated across 3 channels to interface directly with ImageNet-pretrained ResNet-18 weights.

---

## 8. Loss Formulation & Class Imbalance Strategy

The dataset exhibits severe class imbalance:
$$\text{Healthy (690)} : \text{Partial (172)} : \text{Complete (55)} \approx 12.5 : 3.1 : 1.0$$
A naive model predicting all samples as "Healthy" would achieve 75.2% accuracy while failing completely on injured patients.

### 8.1 Balanced Class Weights
Class weights are computed strictly from the **training split**:
$$W_c = \frac{N_{\text{train}}}{C \cdot N_c}$$
where $N_{\text{train}}$ is the total number of training volumes, $C = 3$ is the number of classes, and $N_c$ is the frequency of class $c$ in the training split.

### 8.2 Loss Function
Weighted Cross-Entropy Loss:
$$\mathcal{L} = -\sum_{c=0}^{2} W_c \cdot y_c \log(\hat{y}_c)$$

---

## 9. Model Selection & Evaluation Protocol

1. **Primary Metric: Validation Macro-F1**:
   $$\text{Macro-F1} = \frac{\text{F1}_{\text{Healthy}} + \text{F1}_{\text{Partial}} + \text{F1}_{\text{Complete}}}{3}$$
   Macro-F1 weights each clinical category equally, penalizing models that ignore minority tear cases.
2. **Secondary Metrics**:
   * Per-class Recall / Sensitivity (critical for detecting complete tears)
   * Per-class Precision / Positive Predictive Value
   * Overall Multi-class Accuracy
   * Multi-class Confusion Matrix
   * One-vs-Rest Multiclass ROC-AUC
3. **Locking Protocol**: The test set is evaluated **only once** on the locked best model checkpoint to prevent test-set overfitting.

---

## 10. Mandatory Full-State Checkpoint & Resume Engine

To ensure robustness against training interruptions, power outages, and laptop thermal throttling, a production-grade checkpointing system is implemented:

### 10.1 Checkpoint Artifacts
1. `checkpoints/latest_checkpoint.pth`: Saved at the completion of **every epoch**.
2. `checkpoints/best_model.pth`: Saved whenever **Validation Macro-F1** achieves a new best value.

### 10.2 Saved State Dictionary
Each checkpoint file contains:
* `epoch`: Integer index of the completed epoch.
* `model_state_dict`: Full neural network weights.
* `optimizer_state_dict`: Optimizer momentum/gradient buffers.
* `scheduler_state_dict`: Learning rate scheduler state.
* `best_macro_f1`: Highest recorded validation macro-F1.
* `history`: Complete dictionary of train/val loss, accuracy, and macro-F1 curves.
* `config`: Complete hyperparameter dictionary.
* `rng_states`: PyTorch, CUDA, NumPy, and Python random generator states.
* `split_id`: Hash or identifier of `volume_split.csv`.

### 10.3 Automatic Resume Behavior
When `RESUME = True`, the training loop checks for `latest_checkpoint.pth`. If found:
1. All network parameters, optimizer states, and histories are restored.
2. Training immediately resumes at `completed_epoch + 1`.
3. Completed epochs are **never retrained**.
4. If no checkpoint is found, training initializes gracefully from Epoch 0.
5. All file writes use an atomic temporary swap pattern (`checkpoint.tmp` $\to$ `checkpoint.pth`) to prevent file corruption during premature termination.

---

## 11. Explicit Experimental Plan (E0 – E7)

The experimental suite systematically investigates optimization, modeling, and slice aggregation:

| ID | Experiment Name | Controlled Variable / Modification | Research Purpose |
| :--- | :--- | :--- | :--- |
| **E0** | **Baseline ResNet-18** | ResNet-18 + Uniform Mean Slice Pooling + Adam | Reference benchmark (treats all slices equally) |
| **E1** | **Proposed AAN Model** | ResNet-18 + Lightweight Attention Slice Pooling | Directly answers primary research question: does learned slice weighting improve classification? |
| **E2** | **Class Weighting Ablation** | Standard Cross-Entropy vs. Balanced Class-Weighted Cross-Entropy | Quantifies performance gain on minority classes (Partial & Complete tears) |
| **E3** | **Optimizer Comparison** | Adam ($\beta_1=0.9, \beta_2=0.999$) vs. SGD with Momentum ($\mu=0.9$) | Syllabus-aligned comparison of adaptive vs. classical gradient optimization |
| **E4** | **Augmentation & Regularization** | Minimal Preprocessing vs. Data Augmentation + Dropout ($0.3$) + Weight Decay ($10^{-4}$) | Evaluates regularization against validation overfitting |
| **E5** | **Fine-Tuning Strategy** | Frozen Backbone (Linear Probe) vs. Gradual Unfreezing of ResNet Layer4 | Investigates domain adaptation from ImageNet natural features to knee MRI features |
| **E6** | **Optional 2.5D Extension** | Single 2D Slice vs. 3-Slice Stacking $(z-1, z, z+1)$ | Assesses whether local through-plane spatial context improves tear detection |
| **E7** | **Grad-CAM & Error Analysis** | Layer4 Feature Activation Heatmaps & Attention Weight Attribution | Qualitative and clinical explainability; failure case diagnostics |

---

## 12. Explainability & Clinical Visualization

The model must not act as an uninterpretable black box. Explainability is delivered through two complementary mechanisms:

1. **Slice-Level Attention Weights ($a_k$)**: Visual bar chart ranking the $K$ slices from most to least attended by the network, cross-referenced against the radiologist's ground-truth bounding box slices (`roiZ` to `roiZ + roiDepth`).
2. **Intra-Slice Grad-CAM Saliency**: Saliency heatmaps generated from the final convolutional block (`layer4`) of ResNet-18:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
   Heatmaps are overlaid on the grayscale MRI slice to visually confirm whether the model focuses on the intercondylar notch and tibial plateau rather than spurious background artifacts.

---

## 13. Notebook Structure & Implementation Standards

The primary deliverable is `notebooks/KneeMRI_ACL_Classification.ipynb`. It is organized into 16 sequential, clearly documented cells:

```
01. System Environment & Hardware Diagnostic (GPU / CUDA / PyTorch check)
02. Dataset Integrity Audit & File Verification (917 volumes checked)
03. Exploratory Data Analysis & Class Imbalance Visualization
04. Volumetric MRI Inspection & ROI Slice Overlay
05. Patient-Level Stratified Train/Val/Test Split Creation
06. Dataset Class & Informative Slice Preprocessing Pipeline
07. PyTorch DataLoader Construction & Batch Inspection
08. Model Architectures (ResNet-18 Mean Pooling Baseline & Attention Pooling Proposed)
09. Training Engine & Safe Checkpoint/Resume Handlers
10. Experiment E0: Baseline Model Training (ResNet-18 + Mean Pooling)
11. Experiment E1: Proposed Model Training (ResNet-18 + Attention Pooling)
12. Controlled Experiments (E2: Class Weights, E3: Optimizer, E4: Augmentation, E5: Fine-tuning)
13. Comprehensive Test Evaluation & Comparative Metric Tables
14. Explainability Suite: Inter-Slice Attention Weights & Grad-CAM Visualizations
15. Clinical Error Analysis: Failure Case Diagnostics & Minority Class Breakdown
16. Final Model Checkpoint Export & Summary
```

### Code Style Guidelines
* **Simple, Clean, and Modular**: Code is written in straightforward Python and PyTorch without complex abstract class hierarchies or obscure third-party libraries, ensuring full transparency during academic vivas.
* **Inline Explanations**: Every training loop, attention module, and preprocessing step includes comprehensive markdown cells and inline comments.

---

## 14. Hardware & Time Budget

| Parameter | Specification / Constraint |
| :--- | :--- |
| **Target GPU** | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) |
| **Precision** | PyTorch Automatic Mixed Precision (`torch.amp.autocast`) enabled |
| **Batch Size** | 16 volumes ($16 \times 7 = 112$ 2D slice forward passes per batch) |
| **Baseline Epoch Time** | ~15–20 seconds per epoch |
| **Total 30-Epoch Run** | ~8–10 minutes per experiment |
| **Full Experimental Suite** | $\le 2.5$ hours total compute time (well within the 5-hour constraint) |

---

## 15. Repository Structure

```
KneeMRI-AAN/
├── KneeMRI_ACL_Final_PRD.docx      # Updated official PRD document
├── KneeMRI_ACL_Final_PRD.md        # Markdown version of official PRD
├── metadata.csv                    # Dataset clinical metadata (917 records)
├── volumetric_data/                # Raw MRI volumes directory
│   └── volumetric_data/*.pck       # 917 pickled MRI volume files
├── notebooks/
│   └── KneeMRI_ACL_Classification.ipynb  # Primary deliverable notebook
├── src/
│   ├── dataset.py                  # Modular dataset & preprocessing helper
│   ├── models.py                   # ResNet-18 baseline & Attention pooling architectures
│   ├── training.py                 # Training loop, metrics & checkpoint/resume logic
│   └── gradcam.py                  # PyTorch Grad-CAM & visualization utilities
├── checkpoints/
│   ├── baseline_best_model.pth     # E0 checkpoint
│   ├── proposed_best_model.pth     # E1 checkpoint
│   └── latest_checkpoint.pth       # Active pause/resume checkpoint
├── splits/
│   └── volume_split.csv            # Leakage-safe stratified exam split
├── reports/
│   ├── figures/                    # Generated plots, confusion matrices, Grad-CAM overlays
│   └── metrics_summary.csv         # Comparative metrics table across E0-E7
├── app/
│   └── streamlit_app.py            # Optional local interactive web demonstrator
├── requirements.txt                # Dependency list
└── README.md                       # Comprehensive academic project overview
```

---

## 16. Acceptance Criteria

1. **Dataset Integrity**: All 917 volumes in `volumetric_data/volumetric_data/` verified against `metadata.csv` with 0 missing files.
2. **Leakage Elimination**: Zero patient/exam overlap between Train, Validation, and Test partitions; verified by asserting unique `examId` intersections.
3. **Primary Research Answered**: Quantitative comparison between E0 (Mean Pooling) and E1 (Attention Pooling) conducted with statistical and metric reporting.
4. **Primary Metric Prioritization**: Model selection, early stopping, and hyperparameter tuning strictly driven by **Validation Macro-F1**.
5. **Fault-Tolerant Checkpoint/Resume**: Verified interruption and successful continuation without loss of epoch count, metrics, or optimizer momentum.
6. **Interpretability**: Generation of both inter-slice attention bar charts and intra-slice Grad-CAM heatmaps for correct and misclassified cases.
7. **Budget Adherence**: Complete training and evaluation run finished well within the $\le 5$-hour RTX 4050 compute window.
8. **Simple & Defensible Codebase**: Modular, readable PyTorch implementation designed for direct comprehension and defense in academic vivas.
