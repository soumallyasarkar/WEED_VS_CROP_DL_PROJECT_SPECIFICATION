# Weed vs Crop Discrimination Using Deep Learning and Computer Vision

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Non--Commercial-lightgrey.svg)](#license--citations)
[![Tests](https://img.shields.io/badge/Tests-12%20Passed-brightgreen.svg)](#testing)

Field-image weed localization: detecting and separating crop plants from weeds in real agricultural field photographs using deep learning and object detection.

---

## 1. Project Overview

Manual weed scouting across vast crop fields is labor-intensive and subjective, while blanket chemical herbicide application wastes agrochemicals, harms soil biology, and risks crop damage. Precision agriculture and site-specific weed management (SSWM) require high-accuracy spatial localization to answer: **"Where are the weeds, where are the crops, and what is the weed pressure in this frame?"**

Whole-image classification cannot solve this problem because field images contain **both** crops and weeds simultaneously. This project formulates the problem as **object detection**, predicting a bounding box, category label (`crop` vs `weed`), and confidence score for every plant instance.

### Key Highlights
- **Architecture**: Faster R-CNN with `MobileNetV3-Large FPN` backbone (COCO-pretrained weights) fine-tuned for foreground classes: `0: background`, `1: crop`, `2: weed`.
- **Dataset**: Official **CropAndWeed** dataset (WACV 2023) mapped to the `CropOrWeed2` binary variant.
- **Strict Leakage Prevention**: Session-grouped data splitting (`prefix-session`) ensuring **strict zero session overlap** between Train, Validation, and Test sets.
- **Full Stack Deliverables**: EDA pipeline, CNN patch baseline, detector training pipeline, test-set evaluator, CLI inference engine, Streamlit interactive web application, and pytest test suite.

---

## 2. Directory Structure

```text
weed-vs-crop-discrimination/
├── configs/                      # Experiment configurations
│   ├── default.yaml              # Full dataset configuration
│   ├── laptop.yaml               # Reduced image size & subset configuration
│   └── smoke.yaml                # Ultra-fast pipeline validation (~20 images)
├── docs/                         # Detailed documentation
│   ├── dataset_notes.md          # Official dataset verification, layout & terms
│   └── progress_log.md           # Phase-by-phase execution tracking
├── data/
│   ├── raw/                      # Downloaded dataset (excluded from git)
│   └── processed/                # splits.csv, stats.json
├── models/                       # Checkpoints & class mappings
│   ├── class_mapping.json        # {"background": 0, "crop": 1, "weed": 2}
│   └── best.pt                   # Trained model weights (excluded from git)
├── outputs/
│   ├── plots/                    # EDA, confusion matrices, PR curves, samples
│   ├── reports/                  # metrics.json, eval_report.md, train_log.csv
│   └── predictions/              # Annotated inference outputs & JSON results
├── src/
│   ├── config.py                 # Configuration loader and validator
│   ├── data/
│   │   ├── parse_annotations.py  # Bbox CSV parsing, boundary clipping, label mapping
│   │   ├── splits.py             # Session-grouped split creation & verification
│   │   └── dataset.py            # PyTorch Dataset, augmentations, collate_fn
│   ├── models/
│   │   ├── baseline.py           # Patch CNN baseline (upper-bound classifier)
│   │   └── detector.py           # Faster R-CNN MobileNetV3-FPN builder
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
│   ├── download_dataset.py       # Dataset downloader and setup
│   ├── download_dataset.sh       # Bash wrapper for download
│   ├── prepare_dataset.py        # Validation, EDA stats, plots, and splits
│   └── make_sample_predictions.py# Batch sample visualizer
├── tests/                        # 12 automated unit and integration tests
├── requirements.txt              # Frozen project dependencies
└── pytest.ini                    # Test configuration
```

---

## 3. Environment & Installation

### Prerequisites
- Python 3.10 – 3.12
- PyTorch >= 2.2 with CUDA (GPU recommended; CPU fully supported)

### Setup Instructions
```bash
# 1. Activate your virtual environment
# Windows PowerShell:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Verify environment and hardware
python scripts/check_env.py
```

---

## 4. Dataset Acquisition & Preparation

### Download CropAndWeed Dataset
The official tooling is cloned into `third_party/cropandweed-dataset`, downloading images and annotations directly from the official repository endpoint:
```bash
python scripts/download_dataset.py --parts annotations images1
```

### Run Validation, EDA, and Session-Grouped Splitting
```bash
python scripts/prepare_dataset.py --config configs/default.yaml
```
This generates:
- `data/processed/splits.csv`: 1,381 train, 338 val, 284 test images with **zero session overlap**.
- `data/processed/stats.json`: Comprehensive counts, resolution checks, and box size distributions.
- `outputs/plots/samples.png`: 12-sample grid visualizer with color-coded ground-truth boxes.
- `outputs/plots/class_distribution.png`: Bar chart of crop vs weed frequencies.
- `outputs/plots/box_size_hist.png`: Breakdown of tiny, small, medium, and large instances.

---

## 5. Execution Workflow

### Step 1: Run Patch Baseline Classifier (Phase 5a)
Trains a 3-layer CNN on ground-truth cropped plant patches (96x96) to evaluate separability assuming perfect localization:
```bash
python -m src.models.baseline --config configs/default.yaml --epochs 5
```
*Outputs:* `outputs/reports/baseline_metrics.json`, `outputs/plots/baseline_confusion_matrix.png`

### Step 2: Smoke Test the Full Pipeline (Phase 5b)
Quick end-to-end sanity test on 20 images for 1 epoch:
```bash
python -m src.training.train --config configs/smoke.yaml
```

### Step 3: Train the Faster R-CNN Detector (Phase 6)
Fine-tune the detector on available hardware:
```bash
# Laptop configuration (GPU/CPU friendly):
python -m src.training.train --config configs/laptop.yaml

# Full dataset training:
python -m src.training.train --config configs/default.yaml
```
*Checkpoints saved to:* `models/best.pt` and `models/last.pt`.
*Training history:* `outputs/reports/train_log.csv`.

### Step 4: Evaluate on Untouched Test Split (Phase 7)
Evaluate detection metrics (mAP, per-class AP, weed precision/recall/F1, threshold sweep, detection confusion matrix):
```bash
python -m src.evaluation.evaluate --config configs/default.yaml --checkpoint models/best.pt
```
*Outputs:* `outputs/reports/metrics.json`, `outputs/reports/eval_report.md`, `outputs/plots/pr_curves.png`, `outputs/plots/detection_confusion_matrix.png`.

### Step 5: Run Inference on Single Images (Phase 8)
Run detection on any field photograph (JPEG/PNG) and export annotated imagery and structured JSON:
```bash
python -m src.inference.predict --image data/raw/cropandweed/images/ave-0000-0001.jpg --checkpoint models/best.pt --score-threshold 0.4 --out outputs/predictions/
```

### Step 6: Launch Streamlit Web Application (Phase 9)
Launch the interactive web demo:
```bash
streamlit run app/streamlit_app.py
```

### Step 7: Run Automated Test Suite (Phase 10)
```bash
pytest -v
```

---

## 6. Observed Dataset & EDA Summary

From the initial 2,003 validated field photographs (`1920x1088` resolution, 0 corrupt images):
- **Crop Instances**: 5,128 annotations
- **Weed Instances**: 14,657 annotations (74.1% weed prevalence)
- **Bounding Box Size Breakdown**:
  - `Tiny (< 32x32 px)`: 4,314 boxes (21.8%)
  - `Small (32 - 96 px)`: 9,150 boxes (46.2%)
  - `Medium (96 - 256 px)`: 4,702 boxes (23.8%)
  - `Large (> 256 px)`: 1,619 boxes (8.2%)
- Over **68% of all instances are small or tiny**, verifying the necessity of Feature Pyramid Networks (FPN) for multi-scale feature resolution.

### 6.1 Baseline Patch Classifier (Phase 5a)
Evaluated on 1,000 test patches with ground-truth boxes:
- **Accuracy**: 75.60% (Majority baseline: 80.70%)
- **Weed Precision**: 91.09% | **Weed Recall**: 77.32% | **Weed F1**: 83.65%

### 6.2 Detector Performance on Untouched Test Set (Phase 7)
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

## 7. Limitations & Agricultural Risk Analysis

- **Site & Lighting Specificity**: The model was trained on ground-level handheld agricultural photos in Central Europe. Generalization to aerial drone imagery, tractor-mounted boom cameras, different crop cultivars, or arid soils cannot be assumed without domain adaptation.
- **False Positive vs False Negative Costs**:
  - *False Positive Weed* (Crop flagged as weed): In targeted herbicide spot spraying, this results in crop chemical injury and yield loss (high cost).
  - *False Negative Weed* (Weed missed): The weed escapes treatment, competes with crops for water/nutrients, and sheds seeds (moderate to high cost).
  - Operators should adjust the **Confidence Score Threshold** (e.g. 0.60–0.70 for crop safety vs 0.35–0.45 for aggressive weed eradication) based on specific field priorities.
- **Non-Commercial License**: The dataset is released by AIT under a non-commercial academic license. This software is an educational research prototype and is not certified for autonomous herbicide spraying without human supervisor oversight.

---

## 8. License & Citations

The CropAndWeed dataset is made available by AIT Austrian Institute of Technology GmbH under a custom non-commercial license for academic, research, and educational purposes.

If using this project, please cite:
```bibtex
@InProceedings{Steininger_2023_WACV,
    author    = {Steininger, Daniel and Trondl, Andreas and Croonen, Gerardus and Simon, Julia and Widhalm, Verena},
    title     = {The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {3729-3738}
}
```
