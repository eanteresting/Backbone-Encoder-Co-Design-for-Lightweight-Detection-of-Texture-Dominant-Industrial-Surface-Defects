# Backbone-Encoder-Co-Design-for-Lightweight-Detection-of-Texture-Dominant-Industrial-Surface-Defects
April 16th, 2026

Due to the resubmission timeline suggested by the editor (within two weeks) and we are currently preparing the initial draft of a related patent application. The patent draft is expected to be completed within this period (approximately ten days). Following the completion of the initial filing, we will promptly release the source code（It is expected to take about two weeks）.

----------------------------------------------------------------------------------------------------------------------------------
## Abstract

This repository contains the implementation of **STD-UniDETR**, a lightweight DETR-based framework designed for defect detection under **texture-dominant conditions**.

In industrial visual inspection, subtle defects are often entangled with structured background textures across multiple scales, leading to representation ambiguity and degraded detection performance. To address this challenge, we propose a **unified backbone–encoder co-design framework**, which jointly improves:

- **Defect-oriented representation formation (before fusion)**
- **Adaptive post-fusion feature exploitation (after fusion)**

The key contributions include:

1. **Backbone–Encoder Co-Design Framework**  
   A unified design that addresses texture-induced representation ambiguity by coupling pre-fusion representation learning with post-fusion feature refinement.

2. **STD-HGNetV2 (Backbone Redesign)**  
   A stage-aligned backbone that progressively enhances defect representations via:
   - Adaptive multi-scale perception (AMS-Stem)
   - Scale–frequency representation enhancement (PWConv)
   - Selective discriminative refinement (BSA)

3. **STD-HybridEncoderV2 (Encoder Redesign)**  
   A lightweight post-fusion refinement module (C3K2-MSAM) that performs:
   - Partial-channel adaptive refinement  
   - Multi-scale receptive field modeling  
   - Parameter-free spatial interaction

4. **Strong Accuracy–Efficiency Trade-off**  
   Achieves **29.7% AP50–95 with only 7.6 GFLOPs and 5.5M parameters**, outperforming existing lightweight detectors under comparable computational budgets.


## Environment Configuration

### Hardware Requirements

- NVIDIA GPU (Recommended: RTX 30/40 series)
- CUDA 12.x compatible GPU

### Software Dependencies

- Python 3.10
- PyTorch 2.3+
- CUDA 12.1

### Installation

# Create environment
conda create -n [your project name] python=3.10
conda activate [your project name]

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
1. Obtain and deploy DINOv3 from the official source.
2. Install the required dependencies during runtime.

## Data Preparation

### Datasets

* **PAO Severstal Steel Defect Dataset**
* **NEU-DET Dataset**

### Notes
* In PAO Dataset, background-only samples are partially filtered to reduce imbalance

---

## Project Structure
```
/project_root
├── Baseline/                # Baseline (UniDETR)
├── modules/               # Core modules (AMS-Stem, PWConv, BSA, MSAM)
├── configs/               # Training configuration files
```

---

## 🔬 Core Algorithms

### 1. Adaptive Multi-Scale Stem (AMS-Stem)

* Stabilizes early feature extraction under scale ambiguity and texture interference
* Uses **multi-branch receptive fields + prior-guided fusion**

**Key Features:**

* Multi-scale candidate generation
* Learnable prior-guided weighting
* Channel–spatial refinement

---

### 2. PWConv (Scale–Frequency Representation Enhancement)

* Combines **spatial convolution + wavelet-based multi-depth reconstruction**
* Preserves hierarchical frequency cues for subtle defect modeling

**Mathematical Formulation:**

'Y = w0 · Y_base + Σ wk · Y_wave(k)'


**Key Features:**

* Wavelet pyramid decomposition (multi-level DWT/IDWT)
* Multi-depth reconstruction aggregation
* Global adaptive fusion

---

### 3. BSA (Block Shuffle Attention)

* Enhances final-stage discriminative ability
* Introduces **cross-group interaction + adaptive refinement**

**Key Features:**

* Channel shuffle for cross-group interaction
* Adaptive coefficient α controls refinement strength
* Conditional computation for efficiency

---

### 4. C3K2-MSAM (Post-Fusion Feature Exploitation)

* Performs adaptive refinement after feature fusion

**Key Features:**

* Partial-channel processing (efficient)
* Dual-branch modeling:

  * Dynamic local convolution
  * Large-kernel global modeling
* Shift-based spatial mixing (parameter-free)

---

## 🚀 Training

python train.py --config configs/std_unidetr.yaml


## 📊 Main Results

| Model           | Params | FLOPs | AP50–95  | AP50     |
| --------------- | ------ | ----- | -------- | -------- |
| UniDETR         | 3.6M   | 6.8G  | 26.2     | 56.0     |
| **STD-UniDETR** | 5.5M   | 7.6G  | **29.7** | **60.9** |

## 📊 Generalization on NEU-DET

To further validate the generalization ability of STD-UniDETR, we conduct experiments on the **NEU-DET dataset**, a widely used benchmark for surface defect classification and detection.

### Results on NEU-DET

| Model            | Params | FLOPs | AP50–95 | AP50 |
|------------------|--------|-------|--------|------|
| UniDETR          | 3.6M   | 6.8G  | 45.2   | 76.6 |
| **STD-UniDETR**  | 5.5M   | 7.6G  | **50.9** | **82.2** |



## 📌 Notes

* This repository corresponds to the manuscript submitted to *The Visual Computer*.
* The repository includes the **core implementation and training pipeline required for reproducing the reported results**.
* Additional documentation and improvements will be released soon.
