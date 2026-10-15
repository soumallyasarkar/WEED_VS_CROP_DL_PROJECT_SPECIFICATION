# Weed vs Crop Discrimination Using Deep Learning and Computer Vision
> **Department of Computer Science and Engineering, Indian Institute of Information Technology Kottayam**  
> **Project Authors:** Soumallya Sarkar (2023BCS0172) · N Kedar Vaishnav (2023BCS0163) · Veeramalla Balaji Bhargav (2023BCS0162)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Non--Commercial-lightgrey.svg)](#license--citations)
[![Tests](https://img.shields.io/badge/Tests-12%20Passed-brightgreen.svg)](#run-mode-3-automated-test-suite-12-tests)

Field-image weed localization: detecting and separating crop plants from weeds in real agricultural field photographs using deep learning and object detection.

---

## 1. Project Overview

Manual weed scouting across vast crop fields is labor-intensive and subjective, while blanket chemical herbicide application wastes agrochemicals, harms soil biology, and risks crop damage. Precision agriculture and site-specific weed management (SSWM) require high-accuracy spatial localization to answer: **"Where are the weeds, where are the crops, and what is the weed pressure in this frame?"**

Whole-image classification cannot solve this problem because field images contain **both** crops and weeds simultaneously. This project formulates the problem as **object detection**, predicting a bounding box, category label (`crop` vs `weed`), and confidence score for every plant instance.

### Key Highlights
- **Architecture**: Faster R-CNN with `MobileNetV3-Large FPN` backbone (COCO-pretrained weights) fine-tuned for foreground classes: `0: background`, `1: crop`, `2: weed`.
- **Dataset**: Official **CropAndWeed** dataset (WACV 2023) mapped to the `CropOrWeed2` binary variant.
- **Strict Leakage Prevention**: Session-grouped data splitting (`prefix-session`) ensuring **strict zero session overlap** between Train, Validation, and Test sets.
- **Full Stack Deliverables**: EDA pipeline, CNN patch baseline, detector training pipeline, test-set evaluator, CLI inference engine, Streamlit interactive web application, and a 12-test pytest suite.

---

## 2. Quick Setup & Evaluation Guide (For Evaluators & Reviewers)

> [!IMPORTANT]
> **Academic Project Evaluation Guide:**  
> **[Click here to open the Complete Setup & Evaluation Guide (`docs/SETUP_AND_EVALUATION_GUIDE.md`)](docs/SETUP_AND_EVALUATION_GUIDE.md)** for detailed walkthroughs, hardware benchmarks, evaluation rubrics, and dynamic training instructions.

Follow these quick steps to clone, set up, and evaluate the project from GitHub:

### 1. Clone and Activate Environment
```bash
# Clone the repository
git clone https://github.com/soumallyasarkar/WEED_VS_CROP_DL_PROJECT_SPECIFICATION.git
cd WEED_VS_CROP_DL_PROJECT_SPECIFICATION

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (CMD):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install frozen dependencies
pip install -r requirements.txt
```

---

### 2. Choose Your Evaluation Path

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CHOOSE YOUR EVALUATION PATH                           │
├────────────────────────────────────────┬────────────────────────────────────────┤
│  OPTION A: INSTANT EVALUATION          │  OPTION B: RETRAIN FROM SCRATCH        │
│  (No Dataset Download, No Training)    │  (Download Dataset & Custom Epochs)    │
├────────────────────────────────────────┼────────────────────────────────────────┤
│ • Pre-packaged model: models/best.pt   │ • Download official dataset (~3.5 GB)  │
│ • 20 real field images in sample_images│ • Automated session-grouped split EDA  │
│ • Launch interactive Streamlit web demo│ • Set epochs dynamically (--epochs N)  │
│ • Run 12-test automated test suite     │ • Monitor live tqdm terminal progress  │
│ • CLI single-image inference           │ • Re-evaluate test metrics & PR curves │
└────────────────────────────────────────┴────────────────────────────────────────┘
```

#### Option A: Instant Visual Evaluation (Zero Download, Zero Wait)
The default high-accuracy model weights (`models/best.pt`, 72.5 MB) and 20 real agricultural field photos ([`sample_images/`](sample_images/)) are **already bundled in the GitHub repository**!
```bash
streamlit run app/streamlit_app.py
```
- **Access the Dashboard:** Automatically opens in your browser at `http://localhost:8501`.
- **Pre-packaged Sample Images:** Upload any real field photo from [`sample_images/`](sample_images/) (`ave-0000-0001.jpg`, `ave-0035-0001.jpg`, etc.) without downloading the full multi-GB dataset.
- **Features for Evaluators:** Switch model checkpoints, toggle CPU vs. GPU, adjust the confidence slider (0.10–0.95), and inspect real-time KPIs (**Crop Count**, **Weed Count**, and **Weed Pressure Index %**).

#### Option B: Download Dataset & Retrain with Custom Epochs
If you wish to download the raw CropAndWeed dataset and train the detector yourself:

1. **Download Raw Dataset & Annotations:**
   ```bash
   python scripts/download_dataset.py --parts annotations images1
   ```
2. **Generate Session-Grouped Splits (Zero Leakage) & Run EDA:**
   ```bash
   python scripts/prepare_dataset.py --config configs/default.yaml
   ```
3. **Train with Dynamic Epochs According to Your Available Time:**
   Evaluators can dynamically set the number of training epochs directly via the CLI (`--epochs <N>`) without editing configuration files. Every run displays a real-time progress bar (`tqdm`) with live loss and ETA tracking.

| Available Time | PowerShell / Bash Command | Samples | Epochs | GPU Time (RTX 2050) | CPU Time (Standard Laptop) | Evaluation Objective |
|---|---|:---:|:---:|:---:|:---:|---|
| **~20 Seconds** | `python src/training/train.py --samples 40 --epochs 1` | 40 imgs | 1 | **~20 sec** | **~45 sec** | **Live Viva Smoke Test**: Verify pipeline, loss backprop, and GPU execution in 20s. |
| **~1 Minute** | `python src/training/train.py --epochs 1` | 300 imgs | 1 | **~1.0 min** | **~3 min** | **1-Epoch Verification**: Complete epoch pass, loss reduction & checkpoint saving. |
| **~3 Minutes** | `python src/training/train.py --epochs 3` | 300 imgs | 3 | **~3.2 min** | **~8 min** | **3-Epoch Quick Demo**: Watch training loss decrease and validation mAP climb dynamically. |
| **~5 Minutes** | `python src/training/train.py --epochs 5` | 300 imgs | 5 | **~4.9 min** | **~15 min** | **Standard Baseline**: Full baseline convergence (~44.18% validation mAP@0.5). |
| **~28 Minutes** | `python src/training/train.py --config configs/high_accuracy.yaml` | 750 imgs | 10 | **~28 min** | **~50 min** | **High-Accuracy Run**: 750 images, achieving 54.86% test mAP@0.5. |

> [!TIP]
> **No GPU available? Force CPU mode:** Add `--device cpu` to any command (e.g., `python src/training/train.py --samples 40 --epochs 1 --device cpu`).

---

### 3. 10-Second Test Suite Verification
Run all 12 unit and integration tests to verify model architecture, dataset pipelines, and strict zero session leakage:
```bash
pytest -v
```
*(Runs in ~10 seconds with 12/12 tests passing)*

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
│   ├── best.pt                   # Best model weights checkpoint (mAP@0.5: 0.4091)
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
│   ├── test_splits.py            # Zero data leakage verification
│   ├── test_checkpoint.py        # Model checkpoint save/load integrity
│   ├── test_labels.py            # Class mapping & background handling
│   ├── test_model.py             # Detector forward pass (train & eval)
│   ├── test_dataset.py           # PyTorch dataset & collate_fn
│   └── test_inference.py         # End-to-end inference & prediction summaries
├── requirements.txt              # Frozen project dependencies (PyTorch 2.5.1 + CUDA 12.1)
└── pytest.ini                    # Test suite configuration
```

---

## 4. Detailed CLI Workflows & Evaluation Modes

> [!TIP]
> **Instant Try Out for First-Time Users:**
> A curated folder of **20 real agricultural field photographs** ([`sample_images/`](sample_images/)) is pre-packaged in the repository so you can test the CLI detector and the Streamlit web demo immediately without needing to download the multi-gigabyte dataset!

### 4.1 Environment Activation

Before running commands, activate the virtual environment or use direct paths:

* **Windows PowerShell:**
  ```powershell
  .\.venv\Scripts\activate
  ```
* **Windows Command Prompt (CMD):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
* **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

---

### 4.2 Run Mode 1: Interactive Streamlit Web Application (Recommended)

Launch the interactive web demo to upload field photographs, adjust detection confidence thresholds, and inspect real-time weed vs. crop bounding boxes:

```powershell
streamlit run app/streamlit_app.py
```

* **Access the App:** Open your browser at **`http://localhost:8501`**.
* **Device Selection (CPU vs. CUDA):**
  * In the left sidebar, use the **Device** dropdown.
  * Select **`cuda`** to accelerate inference on your NVIDIA GPU (~15–35 ms per image).
  * Select **`cpu`** for standard CPU computation (~250–600 ms per image).
* **Confidence Threshold Slider:** Adjust between `0.10` and `0.95` (recommended default: `0.50`). Lower thresholds catch more subtle/tiny weeds; higher thresholds avoid false herbicide alarms.
* **Sample Images to Upload:**
  * Ready-to-use sample images: Pick any image from [`sample_images/`](sample_images/) (e.g., `sample_images/ave-0000-0001.jpg`, `sample_images/ave-0035-0005.jpg`)
  * External test samples: `outputs/predictions/non_dataset_test_sample.jpg`

---

### 4.3 Run Mode 2: Single-Image CLI Inference Engine

Run inference directly from the command line on any field photograph (JPEG/PNG) and export both an annotated visual image and structured JSON results:

```powershell
python -m src.inference.predict --image sample_images/ave-0000-0001.jpg --checkpoint models/best.pt --score-threshold 0.50 --out outputs/predictions/
```

* **Outputs Generated:**
  * Visual annotated image: `outputs/predictions/ave-0000-0001_annotated.jpg` (crops in green, weeds in red).
  * Machine-readable detections: `outputs/predictions/ave-0000-0001_detections.json` containing bounding boxes, class labels, and confidence scores.

---

### 4.4 Run Mode 3: Automated Test Suite (12 Tests)

Execute the full suite of 12 unit and integration tests covering dataset loading, model forward passes, checkpoint integrity, label mapping, and zero data leakage:

```powershell
pytest -v
```

All 12 tests are located in [`tests/`](tests/) and validate:
1. `test_splits_zero_leakage`: Confirms zero session overlap between Train, Val, and Test splits.
2. `test_verify_splits_integrity_catches_leakage`: Verifies split verifier detects simulated leakage.
3. `test_checkpoint_save_and_load`: Tests full model state dictionary serialization and deserialization.
4. `test_label_mapping_values`: Validates background (0), crop (1), and weed (2) mappings.
5. `test_unmapped_species_excluded`: Ensures out-of-vocabulary species are filtered.
6. `test_build_detector_classes`: Asserts MobileNetV3-FPN output head dimensions.
7. `test_model_forward_pass_train_and_eval`: Validates train mode loss dictionaries and eval mode predictions.
8. `test_freeze_unfreeze_backbone`: Tests 2-stage fine-tuning backbone freezing logic.
9. `test_dataset_loading_and_tensors`: Validates image tensor types, bounding box scaling, and clipping.
10. `test_collate_fn`: Tests DataLoader variable-length bounding box batch collation.
11. `test_predict_image_and_summarize`: Tests PIL image inference and summary metrics generation.
12. `test_invalid_image_handling`: Tests error handling on invalid image inputs.

---

### 4.5 Run Mode 4: Test-Set Evaluation & Metric Reporting

Evaluate the trained Faster R-CNN detector (`models/best.pt`) on the untouched 284 held-out test field images:

```powershell
python -m src.evaluation.evaluate --config configs/default.yaml --checkpoint models/best.pt
```

* **Outputs Generated:**
  * Metrics snapshot: `outputs/reports/metrics.json` (mAP@0.5, mAP@[0.5:0.95], per-class AP, precision/recall/F1).
  * Markdown evaluation report: `outputs/reports/eval_report.md`.
  * Visual charts: `outputs/plots/pr_curves.png` and `outputs/plots/detection_confusion_matrix.png`.

---

### 4.6 Run Mode 5: Model Training & Dynamic Epoch Overrides

Train or validate models with dynamic CLI flags:

* **Dynamic Epochs & Sample Count (Evaluator Mode - No Config Edits Needed):**
  ```powershell
  # 20-second live test (40 samples, 1 epoch):
  python src/training/train.py --samples 40 --epochs 1

  # 3-minute demo (300 samples, 3 epochs):
  python src/training/train.py --epochs 3

  # Force CPU execution:
  python src/training/train.py --samples 40 --epochs 1 --device cpu
  ```
* **Smoke Test Config (20 images, 1 epoch):**
  ```powershell
  python -m src.training.train --config configs/smoke.yaml
  ```
* **Standard Laptop Baseline (300 images, 5 epochs):**
  ```powershell
  python -m src.training.train --config configs/laptop.yaml
  ```
* **High-Accuracy Training (750 images, 10 epochs, 54.86% mAP@0.5):**
  ```powershell
  python -m src.training.train --config configs/high_accuracy.yaml
  ```
* **Full Dataset Training (All 1,381 images, 10 epochs):**
  ```powershell
  python -m src.training.train --config configs/full_dataset.yaml
  ```
* **Patch CNN Baseline Classifier (Upper-bound classifier assuming perfect localization):**
  ```powershell
  python -m src.models.baseline --config configs/default.yaml --epochs 5
  ```

---

## 5. Dataset Acquisition & Preparation

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

## 6. Hardware & CUDA Acceleration Note

* **GPU Execution:** When running on an NVIDIA GPU with CUDA 12.1, PyTorch utilizes cuDNN accelerated convolution and FPN operations.
* **VRAM Requirements:**
  * Inference (Streamlit / CLI): **< 2.0 GB VRAM**
  * Smoke / Laptop Training: **~3.5 – 4.5 GB VRAM**
  * Full Resolution Training (`configs/default.yaml`): **~6.0 – 8.0 GB VRAM**
* **Inference Latency Comparison:**
  * **CUDA (GPU):** ~15 – 35 ms per image.
  * **CPU:** ~250 – 600 ms per image.
* **Verification:** Run `python scripts/check_env.py` to verify whether CUDA is detected and view available GPU hardware details.

---

## 7. Observed Dataset & EDA Summary

From the initial 2,003 validated field photographs (`1920x1088` resolution, 0 corrupt images):
- **Crop Instances**: 5,128 annotations
- **Weed Instances**: 14,657 annotations (74.1% weed prevalence)
- **Bounding Box Size Breakdown**:
  - `Tiny (< 32x32 px)`: 4,314 boxes (21.8%)
  - `Small (32 - 96 px)`: 9,150 boxes (46.2%)
  - `Medium (96 - 256 px)`: 4,702 boxes (23.8%)
  - `Large (> 256 px)`: 1,619 boxes (8.2%)
- Over **68% of all instances are small or tiny**, verifying the necessity of Feature Pyramid Networks (FPN) for multi-scale feature resolution.

### 7.1 Baseline Patch Classifier (Phase 5a)
Evaluated on 1,000 test patches with ground-truth boxes:
- **Accuracy**: 75.60% (Majority baseline: 80.70%)
- **Weed Precision**: 91.09% | **Weed Recall**: 77.32% | **Weed F1**: 83.65%

### 7.2 Detector Performance on Untouched Test Set (Phase 7)
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

## 8. Limitations & Agricultural Risk Analysis

- **Site & Lighting Specificity**: The model was trained on ground-level handheld agricultural photos in Central Europe. Generalization to aerial drone imagery, tractor-mounted boom cameras, different crop cultivars, or arid soils cannot be assumed without domain adaptation.
- **False Positive vs False Negative Costs**:
  - *False Positive Weed* (Crop flagged as weed): In targeted herbicide spot spraying, this results in crop chemical injury and yield loss (high cost).
  - *False Negative Weed* (Weed missed): The weed escapes treatment, competes with crops for water/nutrients, and sheds seeds (moderate to high cost).
  - Operators should adjust the **Confidence Score Threshold** (e.g. 0.60–0.70 for crop safety vs 0.35–0.45 for aggressive weed eradication) based on specific field priorities.
- **Non-Commercial License**: The dataset is released by AIT under a non-commercial academic license. This software is an educational research prototype and is not certified for autonomous herbicide spraying without human supervisor oversight.

---

## 9. Authors & Project Team

This project was developed at the **Indian Institute of Information Technology Kottayam (IIIT Kottayam)**, Department of Computer Science and Engineering:

| Author | Roll Number | Department | Institution | Contact |
|---|:---:|---|---|---|
| **Soumallya Sarkar** | `2023BCS0172` | Computer Science and Engineering | Indian Institute of Information Technology Kottayam | [soumallya23bcs172@iiitkottayam.ac.in](mailto:soumallya23bcs172@iiitkottayam.ac.in) |
| **N Kedar Vaishnav** | `2023BCS0163` | Computer Science and Engineering | Indian Institute of Information Technology Kottayam | [kedar23bcs163@iiitkottayam.ac.in](mailto:kedar23bcs163@iiitkottayam.ac.in) |
| **Veeramalla Balaji Bhargav** | `2023BCS0162` | Computer Science and Engineering | Indian Institute of Information Technology Kottayam | [balaji23bcs162@iiitkottayam.ac.in](mailto:balaji23bcs162@iiitkottayam.ac.in) |

---

## 10. License & Citations

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
