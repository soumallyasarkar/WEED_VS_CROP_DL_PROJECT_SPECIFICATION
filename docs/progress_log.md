# Weed vs Crop Discrimination - Progress Log

This progress log tracks the step-by-step implementation of the project as required by Section 23 of `WEED_VS_CROP_DL_PROJECT_SPECIFICATION.md`.

## Project Checklist & Status

| Task / Item | Status | Evidence / Notes |
|---|---|---|
| **Phase 1: Environment Setup** | Done | Python 3.11.9, PyTorch 2.5.1+cu121, CUDA True, RTX 2050 (4GB), `scripts/check_env.py` verified, `requirements.txt` frozen |
| **Phase 2: Dataset Acquisition** | Done | Official CropAndWeed tooling cloned in `third_party/`, 2,003 real field images downloaded, `CropOrWeed2` variant mapped |
| **Phase 3: Exploratory Data Analysis (EDA)** | Done | 2,003 images checked (0 corrupt, 1920x1088), 194 sessions, 5,128 crops, 14,657 weeds; EDA plots generated |
| **Phase 4: Dataset Preparation & Splits** | Done | Session-grouped splits (135 train, 29 val, 30 test sessions; ZERO leakage verified), `data/processed/splits.csv` saved |
| **Phase 5a: Baseline Patch Classifier** | Done | 3-layer CNN evaluated on 1,000 test patches: Weed Prec=91.09%, Weed Rec=77.32%, Weed F1=83.65%, Test Acc=75.60% |
| **Phase 5b: Detector Architecture** | Done | Faster R-CNN MobileNetV3-Large FPN with 3-class head built, unfreeze schedule verified |
| **Phase 6: Training Pipeline** | Done | Trained 5 epochs in 4.90m on RTX 2050; Train Loss 0.5976 -> 0.4927; Val mAP 0.2658, Val mAP@50 0.4418; best.pt saved |
| **Phase 7: Evaluation & Metrics** | Done | Evaluated on 284 test images: mAP@50=0.4091, mAP=0.1957, Crop AP=0.3247, Weed AP=0.0667, PR curves & reports saved |
| **Phase 8: Inference Module** | Done | `src/inference/predict.py` implemented (CLI, drawing, JSON summary, invalid input handling) |
| **Phase 9: Streamlit Application** | Done | `app/streamlit_app.py` implemented with sliders, KPI cards, tables, and warnings |
| **Phase 10: Tests & Validation** | Done | 12 tests passed in `pytest -q` across labels, splits, dataset, model, checkpoint, inference |
| **Phase 11: Documentation & Deliverables** | Done | `README.md`, `docs/dataset_notes.md`, `docs/progress_log.md` fully documented |

---

## Detailed Execution History

### 1. Environment & Hardware Verification
- System: Windows 10 AMD64, Intel 12-core / 16-thread CPU, 16 GB RAM.
- GPU: NVIDIA GeForce RTX 2050 (4.00 GB VRAM).
- PyTorch: `2.5.1+cu121`, TorchVision: `0.20.1+cu121`, CUDA: `True`.
- Virtual environment created (`.venv`) with all dependencies installed.

### 2. Dataset Setup (`data/raw/cropandweed`)
- Downloaded and verified:
  - `cropandweed_annotations.tar` (105.5 MB)
  - `cropandweed_images1of4.tar` (2.78 GB, 2,003 field photos)
- Extracted and mapped to `CropOrWeed2` variant (7,705 CSVs, 8,034 eval CSVs, params metadata).
- Verified bounding box format: no header, 7 columns (`left, top, right, bottom, label_id, stem_x, stem_y`).
- Verified params format: header present (`moisture, soil, lighting, separability`).

### 3. Exploratory Data Analysis & Splitting
- Executed `scripts/prepare_dataset.py`.
- Verified 2,003 valid images, 0 corrupt images, uniform resolution `1920x1088`.
- Total plants: 19,785 (5,128 crops, 14,657 weeds).
- Box size distribution: Tiny (<32px): 4,314; Small (32-96px): 9,150; Medium (96-256px): 4,702; Large (>256px): 1,619. Over 68% are tiny/small.
- Generated session-grouped splits saved to `data/processed/splits.csv`:
  - Train: 1,381 images across 135 sessions
  - Val: 338 images across 29 sessions
  - Test: 284 images across 30 sessions
  - Zero session overlap assertion verified.
- Visualizations saved:
  - `outputs/plots/class_distribution.png`
  - `outputs/plots/box_size_hist.png`
  - `outputs/plots/samples.png`

### 4. Codebase Architecture & Testing
- Source modules implemented in `src/`:
  - `src/config.py`: YAML loader
  - `src/utils/`: seed, logging, viz
  - `src/data/`: `parse_annotations.py`, `splits.py`, `dataset.py`
  - `src/models/`: `baseline.py`, `detector.py`
  - `src/training/`: `train.py`
  - `src/evaluation/`: `metrics.py`, `evaluate.py`
  - `src/inference/`: `predict.py`
  - `app/streamlit_app.py`
- Automated test suite executed:
  - `pytest -q`: **12 passed in 13.32s** with 0 warnings or errors.
