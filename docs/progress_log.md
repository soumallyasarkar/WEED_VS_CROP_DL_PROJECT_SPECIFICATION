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
| **Phase 6: Training Pipeline** | Done | Trained 5 epochs in 4.90m on RTX 2050; Train Loss 0.5976 -> 0.4927; Val mAP 0.2658, Val mAP@50 0.4418; `models/best.pt` saved |
| **Phase 7: Evaluation & Metrics** | Done | Evaluated on 284 test images: mAP@50=0.4091, mAP=0.1957, Crop AP=0.3247, Weed AP=0.0667, PR curves & reports saved |
| **Phase 8: Inference Module** | Done | `src/inference/predict.py` implemented (CLI, drawing, JSON summary, invalid input handling) |
| **Phase 9: Streamlit Application** | Done | `app/streamlit_app.py` implemented with sliders, KPI cards, tables, CPU/CUDA switch, and warnings |
| **Phase 10: Tests & Validation** | Done | 12 tests passed in `pytest -v` across labels, splits, dataset, model, checkpoint, inference |
| **Phase 11: Documentation & Deliverables** | Done | `README.md` (with dedicated Quickstart), `docs/dataset_notes.md`, `docs/progress_log.md`, `docs/WEED_VS_CROP_DL_PROJECT_SPECIFICATION.md` |

---

## Detailed Execution History

### 1. Environment & Hardware Verification
- **System**: Windows 10 AMD64, Intel 12-core / 16-thread CPU, 16 GB RAM.
- **GPU**: NVIDIA GeForce RTX 2050 Laptop GPU (4.00 GB Dedicated VRAM).
- **Runtimes**: Python `3.11.9`, PyTorch `2.5.1+cu121`, TorchVision `0.20.1+cu121`, CUDA: `True`.
- **Environment**: Isolated project virtual environment (`.venv`) with all locked dependencies in `requirements.txt`.
- **Verification Script**: Executed `scripts/check_env.py` confirming GPU driver compatibility and CUDA runtime acceleration.

### 2. Dataset Setup (`data/raw/cropandweed`)
- Downloaded and verified from official AIT endpoint:
  - `cropandweed_annotations.tar` (105.5 MB)
  - `cropandweed_images1of4.tar` (2.78 GB, 2,003 field photos)
- Unpacked and mapped to `CropOrWeed2` variant (7,705 CSVs, 8,034 eval CSVs, params metadata).
- Bounding box format validated: 7 headerless columns (`left, top, right, bottom, label_id, stem_x, stem_y`).
- Metadata format validated: header present (`moisture, soil, lighting, separability`).

### 3. Exploratory Data Analysis & Splitting
- Executed `scripts/prepare_dataset.py`.
- Verified 2,003 valid images, 0 corrupt images, uniform native resolution `1920x1088`.
- Total plants: 19,785 (5,128 crops, 14,657 weeds; 74.1% weed prevalence).
- Bounding box sizes: Tiny (<32px): 4,314; Small (32-96px): 9,150; Medium (96-256px): 4,702; Large (>256px): 1,619. Over 68% are tiny/small.
- Generated session-grouped splits saved to `data/processed/splits.csv`:
  - **Train**: 1,381 images across 135 sessions
  - **Validation**: 338 images across 29 sessions
  - **Test**: 284 images across 30 sessions
  - **Session Leakage**: Strictly 0.0% overlap between splits.
- Visualizations saved to `outputs/plots/`:
  - `class_distribution.png`
  - `box_size_hist.png`
  - `samples.png`

### 4. Baseline Patch Classifier (Phase 5a)
- Trained 3-layer CNN classifier on 96×96 cropped plant patches to evaluate intrinsic separability assuming perfect localization.
- Evaluated on 1,000 held-out test patches:
  - Test Accuracy: 75.60% (Majority baseline: 80.70%)
  - Weed Precision: 91.09% | Weed Recall: 77.32% | Weed F1: 83.65%
- Artifacts saved: `outputs/reports/baseline_metrics.json`, `outputs/plots/baseline_confusion_matrix.png`.

### 5. Detector Architecture & Training (Phases 5b & 6)
- **Architecture**: Faster R-CNN with `MobileNetV3-Large FPN` backbone (COCO-pretrained weights).
- **Fine-Tuning**: 2-stage unfreeze schedule (frozen backbone warm-up followed by full fine-tuning).
- **Training**: Executed for 5 epochs in 4.90 minutes on NVIDIA RTX 2050:
  - Train loss decreased steadily: `0.5976` -> `0.4927`.
  - Validation mAP@50 reached `0.4418` (mAP@[0.5:0.95]: `0.2658`).
- Checkpoints saved: `models/best.pt` (151.8 MB), `models/last.pt`, `models/class_mapping.json`.
- Training trajectory recorded in `outputs/reports/train_log.csv` and plotted in `outputs/plots/learning_curves.png`.

### 6. Held-Out Test Set Evaluation (Phase 7)
- Evaluated `models/best.pt` on the 284 held-out test field images (`scripts/evaluation/evaluate.py`):
  - **mAP@0.5**: `0.4091` (40.91%)
  - **mAP@[0.5:0.95]**: `0.1957` (19.57%)
  - **Crop Average Precision (AP)**: `0.3247`
  - **Weed Average Precision (AP)**: `0.0667`
  - **Crop P / R / F1 (Score ≥ 0.50, IoU ≥ 0.50)**: `0.6360` / `0.6011` / `0.6180`
  - **Weed P / R / F1 (Score ≥ 0.50, IoU ≥ 0.50)**: `0.4709` / `0.2086` / `0.2891`
- Generated deliverables:
  - Numerical metrics: `outputs/reports/metrics.json`
  - Written report: `outputs/reports/eval_report.md`
  - Precision-Recall curves: `outputs/plots/pr_curves.png`
  - Detection confusion matrix: `outputs/plots/detection_confusion_matrix.png`

### 7. Inference Engine & Non-Dataset Qualitative Testing (Phase 8)
- Standalone inference implemented in `src/inference/predict.py`.
- Validated on both dataset photographs and arbitrary external non-dataset agricultural photographs:
  - Annotated visual images and JSON detections saved in `outputs/predictions/` (e.g. `ave-0040-0001_annotated.jpg`, `non_dataset_test_sample_annotated.jpg`).
- Robust error handling validated: rejects non-image inputs, corrupted files, and zero-dimension images with clean error messages.

### 8. Interactive Streamlit Web Application (Phase 9)
- Implemented in `app/streamlit_app.py`:
  - Interactive file uploader accepting JPEG/PNG field images.
  - Sidebar confidence threshold slider (`0.10` to `0.95`).
  - Hardware device selector supporting seamless toggling between **CPU** and **CUDA** runtime.
  - Side-by-side original vs. detection result comparison with color-coded bounding boxes (green: crop, red: weed).
  - KPI summary metrics: Crop count, Weed count, Total plants, Weed Pressure index.
  - Structured detections table with class names, confidence percentages, and box coordinates.

### 9. Automated Test Suite (Phase 10)
- Executed `pytest -v`: **12 passed** across 6 test modules:
  1. `tests/test_splits.py`: `test_splits_zero_leakage`, `test_verify_splits_integrity_catches_leakage`
  2. `tests/test_checkpoint.py`: `test_checkpoint_save_and_load`
  3. `tests/test_labels.py`: `test_label_mapping_values`, `test_unmapped_species_excluded`
  4. `tests/test_model.py`: `test_build_detector_classes`, `test_model_forward_pass_train_and_eval`, `test_freeze_unfreeze_backbone`
  5. `tests/test_dataset.py`: `test_dataset_loading_and_tensors`, `test_collate_fn`
  6. `tests/test_inference.py`: `test_predict_image_and_summarize`, `test_invalid_image_handling`

### 10. Documentation & Final Deliverables (Phase 11)
- `README.md`: Complete quickstart guide with dedicated run instructions for Web App, CLI inference, PyTest, training, and CUDA acceleration.
- `docs/dataset_notes.md`: Official source links, non-commercial license terms, raw directory layouts, label mapping rules, and verified local dataset statistics.
- `docs/progress_log.md`: Step-by-step audit of execution history and milestone evidence.
- `docs/WEED_VS_CROP_DL_PROJECT_SPECIFICATION.md`: Master specification with fully verified completion checklist.

