# Weed vs Crop Discrimination Using Deep Learning and Computer Vision
> **Department of Computer Science and Engineering, Indian Institute of Information Technology Kottayam**  
> **Project Authors:** Soumallya Sarkar (2023BCS0172) · N Kedar Vaishnav (2023BCS0163) · Veeramalla Balaji Bhargav (2023BCS0162)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Non--Commercial-lightgrey.svg)](#8-license--citations)
[![Tests](https://img.shields.io/badge/Tests-12%20Passed-brightgreen.svg)](#3-automated-test-suite-12-tests-in-8s)

Field-image weed localization: detecting and separating crop plants from weeds in real agricultural field photographs using deep learning and object detection.

---

## 1. Project Overview

Manual weed scouting across vast crop fields is labor-intensive and subjective, while blanket chemical herbicide application wastes agrochemicals, harms soil biology, and risks crop damage. Precision agriculture and site-specific weed management (SSWM) require high-accuracy spatial localization to answer: **"Where are the weeds, where are the crops, and what is the weed pressure in this frame?"**

Whole-image classification cannot solve this problem because field images contain **both** crops and weeds simultaneously. This project formulates the problem as **object detection**, predicting a bounding box, category label (`crop` vs `weed`), and confidence score for every plant instance.

### Key Highlights
- **Architecture**: Faster R-CNN with `MobileNetV3-Large FPN` backbone (COCO-pretrained weights) fine-tuned for foreground classes: `0: background`, `1: crop`, `2: weed`.
- **Dataset**: Official **CropAndWeed** dataset (WACV 2023) mapped to the `CropOrWeed2` binary variant.
- **Strict Leakage Prevention**: Session-grouped data splitting (`prefix-session`) ensuring **strict zero session overlap** between Train, Validation, and Test sets.
- **Pre-packaged Model & Samples**: The trained model weights (`models/best.pt`, 72.5 MB) and 20 real agricultural test photographs (`sample_images/`) are included directly in the repository for instant evaluation without downloading the full multi-GB dataset.
- **Full Stack Deliverables**: EDA pipeline, CNN patch baseline, detector training pipeline, test-set evaluator, CLI inference engine, Streamlit interactive web application, and a 12-test automated pytest suite.

---

## 2. Quickstart & How to Run

> [!IMPORTANT]
> **Academic Project Evaluation Guide:**  
> **[Click here to open the Complete Setup & Evaluation Guide (`docs/SETUP_AND_EVALUATION_GUIDE.md`)](docs/SETUP_AND_EVALUATION_GUIDE.md)** for detailed walkthroughs, hardware benchmarks, evaluation rubrics, and dynamic training instructions.

### 2.1 Clone and Environment Setup

```bash
# Clone the repository
git clone https://github.com/soumallyasarkar/WEED_VS_CROP_DL_PROJECT_SPECIFICATION.git
cd WEED_VS_CROP_DL_PROJECT_SPECIFICATION

# Create virtual environment
python -m venv .venv

# Activate virtual environment:
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (CMD):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2.2 Option A: Instant Evaluation (No Dataset Download Required)

The default trained model (`models/best.pt`) and 20 real field images ([`sample_images/`](sample_images/)) are pre-bundled in the repository:

#### 1. Interactive Streamlit Web Application (Recommended)
```bash
streamlit run app/streamlit_app.py
```
- Opens automatically in your browser at `http://localhost:8501`.
- Drag and drop any image from [`sample_images/`](sample_images/) (`ave-0000-0001.jpg`, `ave-0035-0001.jpg`, etc.).
- Adjust the confidence threshold slider (0.10–0.95), switch between CPU and CUDA GPU, and inspect real-time KPIs (**Crop Count**, **Weed Count**, and **Weed Pressure Index %**).

#### 2. Single-Image Command-Line Inference
```bash
python -m src.inference.predict --image sample_images/ave-0000-0001.jpg --checkpoint models/best.pt --score-threshold 0.50 --out outputs/predictions/
```
Generates an annotated image (`ave-0000-0001_annotated.jpg`) and machine-readable detections (`ave-0000-0001_detections.json`).

#### 3. Automated Test Suite (12 Tests in ~8s)
```bash
pytest -v
```
Validates zero data leakage, dataset loaders, label mapping, model forward pass, checkpoint integrity, and inference functions (12/12 passing).

---

### 2.3 Option B: Training from Scratch & Dynamic Epochs

If you wish to download the raw official CropAndWeed dataset (~3.5 GB) and train models yourself:

#### 1. Download Dataset & Generate Splits
```bash
# Download official annotations and images
python scripts/download_dataset.py --parts annotations images1

# Generate session-grouped splits (zero leakage) and EDA plots
python scripts/prepare_dataset.py --config configs/default.yaml
```

#### 2. Train with Dynamic Epochs According to Available Time
You can dynamically specify the number of epochs via `--epochs <N>` without editing configuration files. Real-time terminal progress is tracked via `tqdm`:

| Available Time | PowerShell / Bash Command | Samples | Epochs | GPU Time (RTX 2050) | CPU Time (Standard Laptop) | Evaluation Objective |
|---|---|:---:|:---:|:---:|:---:|---|
| **~20 Seconds** | `python src/training/train.py --samples 40 --epochs 1` | 40 imgs | 1 | **~20 sec** | **~45 sec** | **Live Viva Smoke Test**: Verify pipeline, loss backpropagation & GPU execution in 20s. |
| **~1 Minute** | `python src/training/train.py --epochs 1` | 300 imgs | 1 | **~1.0 min** | **~3 min** | **1-Epoch Verification**: Complete epoch pass, loss reduction & checkpoint saving. |
| **~3 Minutes** | `python src/training/train.py --epochs 3` | 300 imgs | 3 | **~3.2 min** | **~8 min** | **3-Epoch Quick Demo**: Watch training loss decrease and validation mAP climb dynamically. |
| **~5 Minutes** | `python src/training/train.py --epochs 5` | 300 imgs | 5 | **~4.9 min** | **~15 min** | **Standard Baseline**: Full baseline convergence (~44.18% validation mAP@0.5). |
| **~28 Minutes** | `python src/training/train.py --config configs/high_accuracy.yaml` | 750 imgs | 10 | **~28 min** | **~50 min** | **High-Accuracy Run**: 750 images, achieving 54.86% test mAP@0.5. |

> [!TIP]
> **No GPU available? Force CPU mode:** Add `--device cpu` to any command (e.g., `python src/training/train.py --samples 40 --epochs 1 --device cpu`).

#### 3. Test-Set Evaluation & Metric Reporting
```bash
python -m src.evaluation.evaluate --config configs/default.yaml --checkpoint models/best.pt
```
Evaluates the detector on 284 held-out test images, generating `outputs/reports/metrics.json`, `eval_report.md`, PR curves, and confusion matrices.

---

## 3. Directory Structure

```text
weed-vs-crop-discrimination/
├── configs/                      # Experiment configurations
│   ├── default.yaml              # Full dataset configuration (10 epochs, 640x1024)
│   ├── laptop.yaml               # Reduced image size & subset configuration
│   ├── smoke.yaml                # Ultra-fast pipeline validation (~20 images, 1 epoch)
│   ├── high_accuracy.yaml        # 750-sample high accuracy training configuration
│   └── full_dataset.yaml         # Complete 100% dataset (1,381 images) configuration
├── docs/                         # Comprehensive documentation
│   ├── SETUP_AND_EVALUATION_GUIDE.md # Evaluator & reviewer setup guide with time breakdown
│   ├── FIGURES_AND_SCREENSHOTS_GALLERY.md # Complete visual figures & detection gallery
│   ├── WEED_VS_CROP_DL_PROJECT_SPECIFICATION.md # Full 10-phase project specification
│   ├── dataset_notes.md          # Official dataset verification, layout & terms
│   └── progress_log.md           # Phase-by-phase execution tracking & milestones
├── data/
│   ├── raw/                      # Downloaded dataset (CropAndWeed bboxes & images)
│   └── processed/                # splits.csv, stats.json
├── models/                       # Checkpoints & class mappings
│   ├── class_mapping.json        # {"background": 0, "crop": 1, "weed": 2}
│   ├── best.pt                   # Trained model weights checkpoint (72.5 MB, mAP@0.5: 0.4091)
│   └── last.pt                   # Final epoch checkpoint
├── sample_images/                # 20 curated real field photographs for instant first-run testing
├── outputs/
│   ├── plots/                    # EDA, confusion matrices, PR curves, learning curves, samples
│   │   ├── samples.png           # Ground-truth sample visualizer
│   │   ├── class_distribution.png# Crop vs weed count distribution
│   │   ├── box_size_hist.png     # Instance bounding box size breakdown
│   │   ├── pr_curves.png         # Precision-Recall curves
│   │   ├── detection_confusion_matrix.png # Detection IoU match matrix
│   │   ├── baseline_confusion_matrix.png  # Patch classifier confusion matrix
│   │   └── learning_curves.png   # Training & validation loss trajectories
│   ├── reports/                  # Evaluation metrics & training logs
│   │   ├── metrics.json          # Complete numerical evaluation metrics
│   │   ├── eval_report.md        # Comprehensive evaluation & error analysis report
│   │   ├── baseline_metrics.json # Patch baseline upper-bound classifier metrics
│   │   ├── train.log             # Formatted execution log
│   │   └── train_log.csv         # Per-epoch loss tracking
│   └── predictions/              # Sample annotated images & inference JSON exports
├── src/                          # Modular source code
│   ├── config.py                 # Configuration loader and validator
│   ├── data/
│   │   ├── parse_annotations.py  # Bbox CSV parsing, boundary clipping, label mapping
│   │   ├── splits.py             # Session-grouped split creation & verification
│   │   └── dataset.py            # PyTorch Dataset, augmentations, collate_fn
│   ├── models/
│   │   ├── baseline.py           # Patch CNN baseline (upper-bound classifier)
│   │   └── detector.py           # Faster R-CNN MobileNetV3-FPN builder & checkpointer
│   ├── training/
│   │   └── train.py              # Training loop, scheduler, checkpointing, early stopping
│   ├── evaluation/
│   │   ├── metrics.py            # mAP, IoU greedy matching, detection confusion matrix
│   │   └── evaluate.py           # Test-set evaluation, threshold sweeps, reports
│   ├── inference/
│   │   └── predict.py            # Standalone and shared inference functions
│   └── utils/
│       ├── seed.py               # Seed management for reproducibility
│       ├── logging.py            # Structured logging
│       └── viz.py                # Bounding box drawing and plotting utilities
├── app/
│   └── streamlit_app.py          # Interactive Streamlit web application
├── scripts/
│   ├── check_env.py              # Environment and hardware verification
│   ├── download_dataset.py       # Official dataset downloader and setup
│   ├── download_dataset.sh       # Bash wrapper for download
│   ├── prepare_dataset.py        # Validation, EDA stats, plots, and splits
│   └── make_sample_predictions.py# Batch sample visualizer
├── tests/                        # 12 automated unit and integration tests
├── requirements.txt              # Frozen project dependencies (PyTorch 2.5.1 + CUDA 12.1)
└── pytest.ini                    # Test suite configuration
```

---

## 4. Hardware & CUDA Acceleration

* **GPU Execution:** When running on an NVIDIA GPU with CUDA 12.1, PyTorch utilizes cuDNN accelerated convolution and FPN operations.
* **VRAM Requirements:**
  * Inference (Streamlit / CLI): **< 2.0 GB VRAM**
  * Smoke / Laptop Training: **~3.5 – 4.5 GB VRAM**
  * Full Resolution Training (`configs/default.yaml`): **~6.0 – 8.0 GB VRAM**
* **Inference Latency Comparison:**
  * **CUDA (GPU):** ~15 – 35 ms per image.
  * **CPU:** ~250 – 600 ms per image.
* **Diagnostics:** Run `python scripts/check_env.py` to inspect hardware and CUDA detection.

---

## 5. Observed Dataset & Benchmark Performance

From the 2,003 validated field photographs (`1920x1088` resolution, 0 corrupt images):
- **Crop Instances**: 5,128 annotations
- **Weed Instances**: 14,657 annotations (74.1% weed prevalence)
- **Bounding Box Scale Breakdown**:
  - `Tiny (< 32x32 px)`: 4,314 boxes (21.8%)
  - `Small (32 - 96 px)`: 9,150 boxes (46.2%)
  - `Medium (96 - 256 px)`: 4,702 boxes (23.8%)
  - `Large (> 256 px)`: 1,619 boxes (8.2%)
- Over **68% of all instances are small or tiny**, verifying the necessity of Feature Pyramid Networks (FPN) for multi-scale feature resolution.

### 5.1 Baseline Patch Classifier (Phase 5a)
Evaluated on 1,000 test patches with ground-truth boxes:
- **Accuracy**: 75.60% (Majority baseline: 80.70%)
- **Weed Precision**: 91.09% | **Weed Recall**: 77.32% | **Weed F1**: 83.65%

### 5.2 Detector Performance on Untouched Test Split (Phase 7)
Evaluated on 284 held-out test field images with Faster R-CNN MobileNetV3-Large FPN (`models/best.pt`):

| Metric | Value | Description |
|---|---|---|
| **mAP@0.5** | **0.4091** (40.91%) | Detection mAP at standard IoU 0.50 |
| **mAP@[0.5:0.95]** | **0.1957** (19.57%) | Strict COCO primary challenge metric |
| **Crop AP** | **0.3247** (32.47%) | Average Precision for Crop class |
| **Weed AP** | **0.0667** (6.67%) | Average Precision for Weed class |
| **Crop Precision** | **0.6360** (63.60%) | At score threshold 0.50, IoU 0.50 |
| **Crop Recall** | **0.6011** (60.11%) | At score threshold 0.50, IoU 0.50 |
| **Crop F1** | **0.6180** (61.80%) | Harmonic mean of crop precision/recall |
| **Weed Precision** | **0.4709** (47.09%) | At score threshold 0.50, IoU 0.50 |
| **Weed Recall** | **0.2086** (20.86%) | At score threshold 0.50, IoU 0.50 |
| **Weed F1** | **0.2891** (28.91%) | Harmonic mean of weed precision/recall |

*Confidence Threshold Sweep (Weed Class at IoU 0.5)*:
- At threshold 0.30: Weed Precision = 26.09%, Weed Recall = 29.54% (F1 = 27.71%)
- At threshold 0.50: Weed Precision = 47.09%, Weed Recall = 20.86% (F1 = 28.91%)
- At threshold 0.70: Weed Precision = 67.17%, Weed Recall = 12.25% (F1 = 20.72%)
- At threshold 0.90: Weed Precision = 78.65%, Weed Recall = 2.78% (F1 = 5.36%)

---

## 6. Limitations & Agricultural Risk Analysis

- **Site & Lighting Specificity**: The model was trained on ground-level handheld agricultural photos in Central Europe. Generalization to aerial drone imagery, tractor-mounted boom cameras, different crop cultivars, or arid soils cannot be assumed without domain adaptation.
- **False Positive vs False Negative Costs**:
  - *False Positive Weed* (Crop flagged as weed): In targeted herbicide spot spraying, this results in crop chemical injury and yield loss (high cost).
  - *False Negative Weed* (Weed missed): The weed escapes treatment, competes with crops for water/nutrients, and sheds seeds (moderate to high cost).
  - Operators should adjust the **Confidence Score Threshold** (e.g. 0.60–0.70 for crop safety vs 0.35–0.45 for aggressive weed eradication) based on specific field priorities.
- **Non-Commercial License**: The dataset is released by AIT under a non-commercial academic license. This software is an educational research prototype and is not certified for autonomous herbicide spraying without human supervisor oversight.

---

## 7. Authors & Project Team

This project was developed at the **Indian Institute of Information Technology Kottayam (IIIT Kottayam)**, Department of Computer Science and Engineering:

| Author | Roll Number | Department | Institution | Contact |
|---|:---:|---|---|---|
| **Soumallya Sarkar** | `2023BCS0172` | Computer Science and Engineering | Indian Institute of Information Technology Kottayam | [soumallya23bcs172@iiitkottayam.ac.in](mailto:soumallya23bcs172@iiitkottayam.ac.in) |
| **N Kedar Vaishnav** | `2023BCS0163` | Computer Science and Engineering | Indian Institute of Information Technology Kottayam | [kedar23bcs163@iiitkottayam.ac.in](mailto:kedar23bcs163@iiitkottayam.ac.in) |
| **Veeramalla Balaji Bhargav** | `2023BCS0162` | Computer Science and Engineering | Indian Institute of Information Technology Kottayam | [balaji23bcs162@iiitkottayam.ac.in](mailto:balaji23bcs162@iiitkottayam.ac.in) |

---

## 8. License & Citations

The CropAndWeed dataset is made available by AIT Austrian Institute of Technology GmbH under a custom non-commercial license for academic, research, and educational purposes.

If citing this work, please use:
```bibtex
@article{sarkar2026weeddetection,
  title     = {Weed Detection in Crop Fields Using Faster R-CNN with a MobileNetV3-Large Feature Pyramid Network},
  author    = {Sarkar, Soumallya and Vaishnav, N Kedar and Bhargav, Veeramalla Balaji},
  journal   = {Department of Computer Science and Engineering, Indian Institute of Information Technology Kottayam},
  year      = {2026}
}

@inproceedings{Steininger_2023_WACV,
  author    = {Steininger, Daniel and Trondl, Andreas and Croonen, Gerardus and Simon, Julia and Widhalm, Verena},
  title     = {The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  month     = {January},
  year      = {2023},
  pages     = {3729-3738}
}
```
