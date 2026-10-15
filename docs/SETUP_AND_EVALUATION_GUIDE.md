# Weed vs Crop Discrimination: Setup, Execution & Evaluation Guide
## Comprehensive Guide for Evaluators, Reviewers & Collaborators

> **Course / Academic Context:** Department of Computer Science and Engineering, Indian Institute of Information Technology Kottayam (IIIT Kottayam)  
> **Project Team:** Soumallya Sarkar (`2023BCS0172`), N Kedar Vaishnav (`2023BCS0163`), Veeramalla Balaji Bhargav (`2023BCS0162`)  
> **Repository:** `https://github.com/soumallyasarkar/WEED_VS_CROP_DL_PROJECT_SPECIFICATION`  

---

## Table of Contents
1. [System Requirements & Prerequisites](#1-system-requirements--prerequisites)
2. [5-Minute Quickstart from GitHub](#2-5-minute-quickstart-from-github)
3. [Two Evaluation Paths for Reviewers](#3-two-evaluation-paths-for-reviewers)
4. [Instant 10-Second Test Suite Verification](#4-instant-10-second-test-suite-verification)
5. [Interactive Web Application (Visual Evaluation Mode)](#5-interactive-web-application-visual-evaluation-mode)
6. [Dynamic Training Guide: Time vs. Epoch Matrix for Evaluators](#6-dynamic-training-guide-time-vs-epoch-matrix-for-evaluators)
7. [Full Dataset Download & Preparation (Optional)](#7-full-dataset-download--preparation-optional)
8. [Quantitative Test-Set Evaluation](#8-quantitative-test-set-evaluation)
9. [Command-Line Inference](#9-command-line-inference)
10. [Troubleshooting & FAQ](#10-troubleshooting--faq)
11. [Academic Citations & Contact](#11-academic-citations--contact)

---

## 1. System Requirements & Prerequisites

* **Operating System:** Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
* **Python Runtime:** Python `3.10` or `3.11` (Python `3.11` recommended)
* **Hardware:**
  * **GPU (Optional, Recommended):** NVIDIA GPU with CUDA support (e.g. RTX 2050 / 3060 / 4060 or Google Colab T4)
  * **CPU Mode:** Fully supported out of the box! All scripts automatically detect CUDA and fall back to multi-core CPU.
* **Disk Space:**
  * **Option A (Pre-trained evaluation only):** ~500 MB total (repository, dependencies & pre-trained model `best.pt`).
  * **Option B (Full raw dataset & training):** ~3.5 GB disk space.

---

## 2. 5-Minute Quickstart from GitHub

### Step 1: Clone the Repository
Open your terminal (PowerShell, Command Prompt, or Bash) and clone the repository:
```bash
git clone https://github.com/soumallyasarkar/WEED_VS_CROP_DL_PROJECT_SPECIFICATION.git
cd WEED_VS_CROP_DL_PROJECT_SPECIFICATION
```

### Step 2: Create and Activate Virtual Environment
* **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
* **On Windows (Command Prompt - CMD):**
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```
* **On Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Frozen Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Environment & Hardware Acceleration
Run the automated pre-flight diagnostics check:
```bash
python scripts/check_env.py
```
*(Prints your Python version, PyTorch release, CUDA availability, GPU device name, and VRAM capacity)*

---

## 3. Two Evaluation Paths for Reviewers

We provide two distinct evaluation paths to accommodate varying evaluator time constraints:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CHOOSE YOUR EVALUATION PATH                           │
├────────────────────────────────────────┬────────────────────────────────────────┤
│  OPTION A: INSTANT EVALUATION          │  OPTION B: RETRAIN FROM SCRATCH        │
│  (No Dataset Download, No Training)    │  (Download Dataset & Custom Epochs)    │
├────────────────────────────────────────┼────────────────────────────────────────┤
│ • Pre-packaged model: models/best.pt   │ • Official dataset download script     │
│ • 20 real field images in sample_images│ • Automated session-grouped split EDA  │
│ • Run interactive Streamlit web demo   │ • Set epochs dynamically (--epochs N)  │
│ • Run 12-test automated test suite     │ • Track real-time training progress    │
│ • CLI single-image inference           │ • Re-evaluate test metrics & PR curves │
│                                        │                                        │
│ -> Start in Section 4 & 5 below        │ -> Start in Section 6 & 7 below        │
└────────────────────────────────────────┴────────────────────────────────────────┘
```

---

## 4. Instant 10-Second Test Suite Verification

To verify that the entire codebase, data pipeline, loss formulations, and bounding box math are 100% intact with **zero data leakage**, run:

```bash
pytest -v
```

### Expected Output:
```text
============================= test session starts =============================
tests/test_checkpoint.py::TestCheckpoint::test_checkpoint_save_and_load PASSED [  8%]
tests/test_dataset.py::TestDataset::test_collate_fn PASSED               [ 16%]
tests/test_dataset.py::TestDataset::test_dataset_loading_and_tensors PASSED [ 25%]
tests/test_inference.py::TestInference::test_invalid_image_handling PASSED [ 33%]
tests/test_inference.py::TestInference::test_predict_image_and_summarize PASSED [ 41%]
tests/test_labels.py::TestLabels::test_label_mapping_values PASSED       [ 50%]
tests/test_labels.py::TestLabels::test_unmapped_species_excluded PASSED  [ 58%]
tests/test_model.py::TestModel::test_build_detector_classes PASSED       [ 66%]
tests/test_model.py::TestModel::test_freeze_unfreeze_backbone PASSED     [ 75%]
tests/test_model.py::TestModel::test_model_forward_pass_train_and_eval PASSED [ 83%]
tests/test_splits.py::TestSplits::test_splits_zero_leakage PASSED        [ 91%]
tests/test_splits.py::TestSplits::test_verify_splits_integrity_catches_leakage PASSED [100%]
============================= 12 passed in ~10s =============================
```

---

## 5. Interactive Web Application (Visual Evaluation Mode)

The most intuitive way for evaluators and professors to examine model detections is through the interactive Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```
*(The dashboard opens automatically in your browser at `http://localhost:8501`)*

### Key Features for Evaluators:
1. **Preloaded Model (`models/best.pt`):** The best checkpoint is loaded by default — no manual file selection required. You can also upload any custom `.pt` file directly via the UI uploader.
2. **Hardware Toggle:** Choose between **CPU** or **CUDA GPU** inference.
3. **Confidence Threshold Slider:** Interactively sweep confidence from `0.10` to `0.95` to observe the trade-off between weed eradication and crop safety.
4. **20 Preloaded Curated Field Images:** Drop in any field photo from [`sample_images/`](../sample_images/) (e.g. `ave-0000-0001.jpg` or `ave-0035-0001.jpg`).
5. **Real-Time KPIs:** Live calculations for **Crop Count**, **Weed Count**, **Total Plants**, and **Weed Pressure Index (%)**.
6. **Tabular Dataframe:** Exact bounding box pixel coordinates and confidence scores for every detected plant.

---

## 6. Dynamic Training Guide: Time vs. Epoch Matrix for Evaluators

Evaluators often have strict time constraints during project presentations or grading sessions. We have implemented dynamic CLI arguments (`--epochs`, `--samples`, `--device`) into [`src/training/train.py`](../src/training/train.py), allowing evaluators to dynamically adjust training parameters to match their available time.

### Training Time vs. Epoch Matrix:

| Evaluation Scenario | PowerShell / Bash Command | Samples | Epochs | GPU Time (RTX 2050) | CPU Time (Standard Laptop) | Purpose & Objective for Evaluator |
|---|---|:---:|:---:|:---:|:---:|---|
| **20-Second Live Viva Smoke Test** | `python src/training/train.py --samples 40 --epochs 1` | 40 imgs | 1 | **~20 seconds** | **~45 seconds** | Rapid proof that data ingestion, loss calculation, and backpropagation execute without errors. |
| **1-Minute Rapid Verification** | `python src/training/train.py --epochs 1` | 300 imgs | 1 | **~1.0 minute** | **~3 minutes** | Verify complete epoch forward/backward pass, loss convergence, and checkpoint saving. |
| **3-Epoch Quick Demo** | `python src/training/train.py --epochs 3` | 300 imgs | 3 | **~3.2 minutes** | **~8 minutes** | Watch training loss decrease and validation mAP climb dynamically in the terminal. |
| **5-Epoch Standard Baseline** | `python src/training/train.py --epochs 5` | 300 imgs | 5 | **~4.9 minutes** | **~15 minutes** | Full baseline reproduction (converges to **44.18% validation mAP@0.5**). |
| **10-Epoch High-Accuracy Run** | `python src/training/train.py --config configs/high_accuracy.yaml` | 750 imgs | 10 | **~28 minutes** | **~50 minutes** | Scaled dataset reproduction (achieves **54.86% test mAP@0.5** and triples Weed AP). |
| **100% Full Dataset Training** | `python src/training/train.py --config configs/full_dataset.yaml` | 1,381 imgs | 10 | **~50 minutes** | **~1.5 hours** | Complete deep training across all 135 recording sessions. |

### Real-Time Terminal Progress Tracking (`tqdm`)
When training runs, you will see a live progress bar showing elapsed time, ETA, batch processing speed, and loss:
```text
Epoch  1/5:  45%|████▍     | 68/150 [00:22<00:27, 3.01batch/s, loss=0.5412, avg_loss=0.5821]
```

> [!TIP]
> **No GPU on your machine? Force CPU mode:** Add `--device cpu` to any command:
> ```bash
> python src/training/train.py --samples 40 --epochs 1 --device cpu
> ```

---

## 7. Full Dataset Download & Preparation (Optional)

If you wish to download the raw multi-gigabyte CropAndWeed dataset and generate the session-grouped splits from scratch:

### Step 1: Download Official Raw Annotations & Images
```bash
python scripts/download_dataset.py --parts annotations images1
```

### Step 2: Run Dataset Verification, EDA, and Zero-Leakage Splitting
```bash
python scripts/prepare_dataset.py --config configs/default.yaml
```
This produces:
- `data/processed/splits.csv`: 1,381 train, 338 val, 284 test images with **zero session overlap**.
- `data/processed/stats.json`: Statistical distribution and resolution checks.
- `outputs/plots/samples.png`: 12-sample grid visualizer.
- `outputs/plots/class_distribution.png`: Weed vs crop counts.
- `outputs/plots/box_size_hist.png`: Bounding box scale breakdown.

---

## 8. Quantitative Test-Set Evaluation

To evaluate the default model checkpoint (`models/best.pt`) against the **284 held-out test images** (with strict zero session leakage) and generate updated numerical reports, PR curves, and confusion matrices:

```bash
python src/evaluation/evaluate.py --config configs/laptop.yaml --checkpoint models/best.pt
```

### Generated Artifacts in `outputs/`:
* Numerical benchmark JSON: [`outputs/reports/metrics.json`](../outputs/reports/metrics.json)
* Formatted evaluation report: [`outputs/reports/eval_report.md`](../outputs/reports/eval_report.md)
* Detection Confusion Matrix heatmap: [`outputs/plots/detection_confusion_matrix.png`](../outputs/plots/detection_confusion_matrix.png)
* Precision-Recall Curve: [`outputs/plots/pr_curves.png`](../outputs/plots/pr_curves.png)

---

## 9. Command-Line Inference

To run inference on an individual image and save visual annotations and a JSON detection summary:

```bash
python src/inference/predict.py --image sample_images/ave-0000-0001.jpg --output-dir outputs/predictions --threshold 0.40
```

### Sample Output:
```text
[INFO] Loaded model checkpoint from models/best.pt (cuda:0)
[INFO] Image: sample_images/ave-0000-0001.jpg (1920x1088)
[INFO] Detections Summary: 5 Crops, 14 Weeds (Weed Pressure: 73.7%)
[INFO] Saved annotated visual to outputs/predictions/ave-0000-0001_annotated.jpg
[INFO] Saved detection metadata to outputs/predictions/ave-0000-0001_detections.json
```

---

## 10. Troubleshooting & FAQ

#### Q: Do I need to download the full dataset or train the model to evaluate the project?
**A:** No! The default pre-trained model checkpoint (`models/best.pt`) and 20 real agricultural test images ([`sample_images/`](../sample_images/)) are included directly in the repository. You can evaluate the project immediately by running `streamlit run app/streamlit_app.py` or `pytest -v`.

#### Q: I don't have an NVIDIA GPU. Can I still evaluate the project?
**A:** Yes, absolutely! Every script includes automatic CPU fallback. You can also explicitly force CPU mode by adding `--device cpu`:
```bash
python src/training/train.py --samples 40 --epochs 1 --device cpu
```

#### Q: How do I verify that there is no data leakage?
**A:** Run `pytest tests/test_splits.py -v`. It programmatically verifies that the set intersection between train, validation, and test recording sessions is strictly empty ($\emptyset$).

#### Q: What model checkpoint is uploaded to GitHub?
**A:** Only the best-performing model checkpoint (`models/best.pt`, 72.5 MB) is tracked and uploaded to GitHub. Intermediate and temporary checkpoints (`last.pt`, `*.ckpt`) are ignored to keep the repository fast, clean, and within GitHub file size limits.

---

## 11. Academic Citations & Contact

For questions, evaluations, or academic inquiries, please contact the project team at **IIIT Kottayam**:
* **Soumallya Sarkar:** [soumallya23bcs172@iiitkottayam.ac.in](mailto:soumallya23bcs172@iiitkottayam.ac.in)
* **N Kedar Vaishnav:** [kedar23bcs163@iiitkottayam.ac.in](mailto:kedar23bcs163@iiitkottayam.ac.in)
* **Veeramalla Balaji Bhargav:** [balaji23bcs162@iiitkottayam.ac.in](mailto:balaji23bcs162@iiitkottayam.ac.in)
