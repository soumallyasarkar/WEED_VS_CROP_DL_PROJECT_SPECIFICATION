# Weed vs Crop Discrimination Using Deep Learning and Computer Vision

**Field-image weed localization: detecting and separating crop plants from weeds in real field photographs**

> **How to use this document.** Place this file in your Antigravity workspace and tell the coding agent:
> *"Read `WEED_VS_CROP_DL_PROJECT_SPECIFICATION.md` completely, then follow Section 23 to implement the project phase by phase."*
>
> **Evidence labels used throughout:**
> - **[Verified]** = checked against an official repository, paper, or dataset card while this document was prepared.
> - **[Secondary]** = reported by a third-party page (survey, blog, tutorial); the agent must re-check it at the official source.
> - **[Unverified]** = not confirmed; the agent must verify before relying on it.
>
> Nothing in this document is a result produced by this project. No accuracy, mAP, or training time is promised.

---

## 1. Project Title and Overview

**Weed vs Crop Discrimination Using Deep Learning and Computer Vision (Field-Image Detection Edition)**

The system takes an ordinary RGB photograph of a crop field (many plants in one frame), finds each plant, draws a bounding box around it, and labels it **`crop`** or **`weed`** with a confidence score. It also reports per-image summaries (number of weeds, number of crops, approximate weed share of detected plants).

The project is built with PyTorch/torchvision, runs on a standard laptop (GPU optional), and is designed so an AI coding agent can build it end to end.

## 2. Problem Statement

Manual weed scouting is slow and subjective, and blanket herbicide spraying wastes chemicals and can damage crops. Site-specific weed management needs a system that can answer, *for each plant in a field image*: "Is this a crop or a weed, and where is it?"

Whole-image classification cannot answer that question, because a real field image usually contains **both** crops and weeds at the same time. This project therefore treats the problem as **object detection**, with semantic segmentation as an extension.

## 3. Objectives and Scope

### 3.1 The three possible task formulations

| Task | Output | Fits "weeds among crops in a field image"? | Role in this project |
|---|---|---|---|
| **Binary image classification** | One label (`crop` or `weed`) for the whole image | **No.** A mixed scene has no single correct label. Valid only for single-plant close-up crops. | Used only as a small **baseline** on cropped plant patches (Phase 5a). |
| **Object detection** | A bounding box + class + confidence for every plant | **Yes.** Localizes each plant; directly usable for counting, mapping, spot-spraying decisions. | **Primary task (required).** |
| **Semantic / instance segmentation** | Per-pixel crop/weed/soil labels | **Yes**, with finer detail (weed coverage area, leaf-level boundaries). | **Optional extension** (Section 21 / Phase 12). |

### 3.2 In scope (required)
1. Detect plants in field images and classify each as `crop` or `weed`.
2. Report confidence scores per detection.
3. Evaluate with detection metrics (mAP, per-class AP, precision/recall at a fixed IoU, F1) on an untouched test set.
4. Inference script (CPU-capable) that draws boxes on a new image and saves a JSON result.
5. Streamlit demo that uses the real trained model.

### 3.3 Out of scope (optional extensions)
- Weed **species** identification (e.g., Fine24 labels).
- Semantic segmentation of pixel regions.
- Live video / camera streaming, embedded deployment (Jetson, Raspberry Pi).
- Drone/orthomosaic mapping.
- Actuation (spray control).

## 4. Learning Task and Assumptions

### 4.1 Exact prediction target
- **Input:** one RGB field image (JPEG/PNG), any resolution (resized internally).
- **Output:** a list of detections, each with `box = [x1, y1, x2, y2]` (pixels in the original image), `label ∈ {crop, weed}`, and `score ∈ [0, 1]`.
- **Model:** a COCO-pretrained **Faster R-CNN with MobileNetV3-Large FPN** backbone (torchvision), fine-tuned for 2 foreground classes (+ background).
- **Positive class:** `weed` (the thing we want to find and act on).

### 4.2 Consequences of errors in agricultural use
- **False positive weed** (crop or soil flagged as weed): in a spot-spraying system this means **herbicide on the crop** (yield damage) or wasted chemical. Cost: high.
- **False negative weed** (weed missed): the weed keeps competing with the crop and can spread. Cost: moderate to high, depends on timing.
- Because the costs differ by application, the confidence threshold is a **configurable operating point**, not a fixed constant. The evaluation must report the precision/recall trade-off, not just one number.

### 4.3 Assumptions
- Images are RGB, taken from a camera pointed at the ground at a moderate distance (handheld or low-mounted), **not** high-altitude drone imagery.
- Annotations come from a human-labelled public dataset; label noise exists.
- Performance on a new farm, crop, lighting, or camera is **not guaranteed** (see Section 21).

### 4.4 Limitations of image-level classification (why we do not use it as the main task)
A single label per image cannot represent a scene with many plants; it forces an arbitrary rule ("weed if any weed present"), hides where the weeds are, and cannot count or measure coverage. Image-level classifiers are only sensible when each image is a tight crop of one plant.

---

## 5. Dataset Research and Comparison

Research performed while preparing this document. **The implementing agent must re-verify the items marked [Secondary] or [Unverified] before relying on them.**

### 5.1 Comparison table

| Dataset | Official source | Annotation type | Images / instances | Crop & weed content | Real crop-vs-weed task? | License | Fit for this project |
|---|---|---|---|---|---|---|---|
| **CropAndWeed** | https://github.com/cropandweed/cropandweed-dataset | Bounding boxes + stem positions + semantic masks + per-image meta-params | "More than 8k" images, ~112k plant instances **[Verified]**; the Hugging Face card lists 8,034 samples **[Secondary]** | 74 species: 16 crop, 58 weed **[Verified]**; ready-made `CropOrWeed2` binary variant exists in the repo **[Verified]** | **Yes** | Custom **non-commercial** license; no redistribution of dataset or modified versions **[Verified/Secondary]** | **Best fit. Selected.** |
| **DeepWeeds** | https://github.com/AlexOlsen/DeepWeeds | Image-level class labels | 17,509 images **[Verified]** | 8 weed species + one negative class of non-target plants/background **[Verified]**. **No crop class.** | **No** (weed species recognition, not crop vs weed) | CC BY 4.0 (images/annotations), Apache-2.0 (code) **[Verified]** | Not suitable as the main dataset. Could pretrain or add negatives, but must not be called a crop-vs-weed dataset. |
| **WeedMap** | https://projects.asl.ethz.ch/datasets/weedmap-2018/ (DOI 10.3929/ethz-c-000788571) | Pixel-level segmentation: background / crop / weed | 18,746 images across 8 datasets, ~5.36 GB **[Verified]** | Sugar beet only; multispectral UAV (RedEdge-M, Sequoia), ~1 cm GSD, Germany & Switzerland **[Verified]** | Yes, but **aerial multispectral** | Not confirmed on the page **[Unverified]** | Not suitable for ground-level RGB field photos; possible extension for segmentation. |
| **PhenoBench** | https://www.phenobench.org (paper: arXiv 2306.04557) | Semantic + plant-instance + leaf-instance masks | 2,872 images, 1024×1024 UAV RGB; 1,407 train / 772 val / 693 test **[Verified/Secondary]** | Sugar beet and weeds, single field **[Verified]**; ~5 weeds per image **[Secondary]** | Yes | CC BY-SA 4.0 **[Secondary]** | Good segmentation extension; low weed density, single crop, single field. |
| **Sugar Beets 2016 (Bonn)** | University of Bonn IPB (see tutorial in refs) | Pixel masks (crop/weed/background) | 11,552 RGB images in the full release **[Secondary]**; only a small annotated subset (280 images listed in PhenoBench's comparison table) **[Secondary]** | Sugar beet from a ground robot | Yes | **[Unverified]** | Possible cross-dataset generalization test. Check annotated count before use. |
| **Sesame Crop and Weed Detection (Kaggle)** | https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes | YOLO bounding boxes | 1,300 images at 512×512, built by augmenting 546 originals **[Verified]** | Sesame + weeds | Yes | CC0 (as shown on page) **[Verified]** | Good **quick smoke-test** set only. Augmented copies of the same photo can leak across splits. Single crop. |
| **WeedsGalore** | https://github.com/GFZ/weedsgalore (arXiv 2502.13103) | Semantic + instance masks | 156 multispectral UAV tiles **[Secondary]** | Maize + 4 weed classes **[Secondary]** | Yes, aerial | CC BY 4.0 **[Secondary]** | Segmentation extension only. |

### 5.2 Key findings
1. **Only CropAndWeed combines**: ground-level RGB photos, bounding boxes, many crops, many weeds, a published paper, and a ready-made binary crop/weed label mapping.
2. **DeepWeeds is not a crop-vs-weed dataset.** It contains weed species plus a "negative" class (other plants/background). Mapping "negative" to "crop" would be scientifically wrong.
3. **WeedMap, PhenoBench, WeedsGalore are aerial (UAV).** Their scale and appearance differ strongly from ground-level photos; a model trained on one will not transfer without adaptation.
4. **A published caveat about CropAndWeed:** a later paper (WeedsGalore) notes CropAndWeed images were collected manually with a handheld camera, which is not the acquisition mode of large-scale automated systems **[Secondary]**. Treat results as "handheld/low-height field photos", not as drone or tractor-mounted performance.

---

## 6. Selected Dataset and Access Instructions

### 6.1 Selection
**Primary dataset: CropAndWeed (WACV 2023), variant `CropOrWeed2`.**

Why: it is the only candidate that genuinely supports **detection of crops and weeds in real field images** with public access and ready-made binary mapping.

### 6.2 Official sources
| Item | URL |
|---|---|
| Dataset repo & download scripts | https://github.com/cropandweed/cropandweed-dataset |
| License text | https://github.com/cropandweed/cropandweed-dataset/blob/main/LICENCE |
| Paper (WACV 2023) | https://openaccess.thecvf.com/content/WACV2023/papers/Steininger_The_CropAndWeed_Dataset_A_Multi-Modal_Learning_Approach_for_Efficient_Crop_WACV_2023_paper.pdf |
| Supplementary material | https://openaccess.thecvf.com/content/WACV2023/supplemental/Steininger_The_CropAndWeed_Dataset_WACV_2023_supplemental.pdf |
| Institutional page (AIT) | https://publications.ait.ac.at/en/datasets/cropandweed-dataset/ |

### 6.3 License and registration
- **No registration or application was mentioned** in the repository README; the README says to run `setup.py` to download and extract images and annotations **[Verified]**. The agent must confirm this when it runs the script.
- **License: non-commercial** use (academic research, teaching, publications, personal experimentation) **[Verified]**. Secondary sources state redistribution of the dataset or modified versions is forbidden **[Secondary]**; the agent must read `LICENCE` itself.
- **Consequences for this project:**
  - Do **not** commit images, masks, or derived/processed copies of the dataset to a public repo.
  - Do **not** present the project as a commercial product.
  - Keep model checkpoints out of public releases unless the license text clearly permits it **[Unverified]**.
  - Cite the paper (BibTeX in Section 25).

### 6.4 Download procedure (agent must follow)
```bash
# 1. Check disk space first (dataset size NOT verified; assume it may be many GB)
df -h .

# 2. Get the official tooling into a third_party folder (not committed)
mkdir -p third_party
git clone https://github.com/cropandweed/cropandweed-dataset.git third_party/cropandweed-dataset

# 3. Read the scripts before running them
sed -n '1,200p' third_party/cropandweed-dataset/cnw/setup.py

# 4. Run the official setup (downloads + extracts + creates mapped variants)
cd third_party/cropandweed-dataset
pip install -r requirements.txt   # if the file exists; otherwise inspect imports
python cnw/setup.py               # see --help for options such as output paths/variants

# 5. If the CropOrWeed2 variant was not created by default:
python cnw/map_dataset.py --dataset_target CropOrWeed2
```
**Notes**
- The README states `setup.py` creates mapped annotations for all pre-defined variants by default **[Verified]**, and `map_dataset.py --dataset_target <Variant>` builds a specific one **[Verified]**. The exact flags and output folders must be read from `--help` and the scripts. Do not guess.
- **Manual fallback:** if the automatic download fails (network, blocked host, changed URL), inspect the error, open the repository README/issues for the current download location, download manually in a browser, and extract to `data/raw/cropandweed/`. Document the exact steps in the README. If the data truly cannot be obtained, **stop and report the blocker; do not substitute fake or synthetic data.**
- **Quick fallback for pipeline testing only:** the Kaggle Sesame dataset (CC0) may be used to test the code path, clearly labelled as *smoke-test data*, never as the project's result dataset.

---

## 7. Dataset Structure and Class Mapping

### 7.1 Annotation format (from the official README) **[Verified]**
For each dataset variant there are directories:
- `bboxes/`: one CSV per image, instance columns: `Left, Top, Right, Bottom, Label ID, Stem X, Stem Y`
- `labelIds/`: semantic mask per image (used only by the segmentation extension)
- `params/`: per-image metadata: `moisture` (0 dry, 1 medium, 2 wet), `soil` (0 fine, 1 medium, 2 coarse), `lighting` (0 sunny, 1 diffuse), `separability` (0 easy, 1 medium, 2 hard)

File names are prefixed `ave` or `vwg` (the paper's *Application* and *Experimental* sets), followed by a 4-digit **recording session** and a 4-digit image id **[Verified]**.

> **Agent must check:** whether CSVs have a header row, whether image files are `.jpg` or `.png`, and the actual folder names created by `setup.py` for `CropOrWeed2`. Record these in `docs/dataset_notes.md`.

### 7.2 Label mapping: `CropOrWeed2` **[Verified from `cnw/utilities/datasets.py`]**
| Variant label ID | Name | Original species IDs mapped into it |
|---|---|---|
| 0 | `Crop` | 1–12 (maize and sugar-beet growth stages), 94 (soybean), 24 (sunflower), 18 (potato), 13 (pea), 26 and 27 (beans), 15 (pumpkin) |
| 1 | `Weed` | 31, 48, 62, 65, 68, 69, 74, 75, 81, 84, 86, 32, 29, 33, 37, 49, 30, 44, 66, 87, 89, 91, 61, 79, 34, 41, 52, 35, 36, 78, 38, 39, 71, 72, 88, 42, 45, 70, 47, 51, 54, 58, 60, 80, 83, 96, 22, 63, 85, 56, 57, 64, 77, 50, 59, 67, 76 |

Species IDs not in either list (e.g., many herbs/vegetables like parsley, sage, garlic, radish) are **not** mapped to either class in `CropOrWeed2`; they are excluded by the official mapping. Do not silently reassign them.

The README notes that evaluation box sets include a fallback **`Vegetation`** class (ID 255 in the full label list) for tiny instances and ambiguous/excluded species **[Verified]**.

### 7.3 Project-internal label convention (torchvision detection)
torchvision detection reserves label `0` for background. Use:
```text
0 = background (implicit)
1 = crop       (variant label 0 + 1)
2 = weed       (variant label 1 + 1)
```
Save this mapping in `models/class_mapping.json` next to the checkpoint:
```json
{"background": 0, "crop": 1, "weed": 2}
```

### 7.4 Treatment of `Vegetation` / ignore regions
- **Training:** use the *training* box set (no `Vegetation` fallback).
- **Evaluation:** report the standard protocol on the evaluation-style box set only if its semantics are understood; otherwise evaluate on the same training-style boxes of the held-out images and state this clearly. The agent must read the paper/supplement to decide and document the decision.

### 7.5 Splitting strategy (leakage control)
Images from the same recording session are visually near-duplicate (same field, same day, same light). A random image-level split would leak.
- **Group** by `(prefix, session)` parsed from the filename.
- Use `sklearn.model_selection.GroupShuffleSplit` (or `StratifiedGroupKFold`) with a fixed seed.
- Target ~70 / 15 / 15 train/val/test **by sessions**; report resulting image and instance counts per class per split.
- Write the split to `data/processed/splits.csv` (columns: `image_id, session, split`) and **never regenerate it silently**; tests must assert zero session overlap.
- **Optional stronger test:** train on one prefix/group and test on another (e.g., `ave` vs `vwg`) to estimate cross-domain generalization. Only do this after reading the paper to understand how the two sets differ.

---

## 8. Technology Stack

| Component | Choice | Purpose |
|---|---|---|
| Language | Python 3.10–3.12 | Main language |
| DL framework | **PyTorch + torchvision** (single framework throughout) | Model, training, pretrained weights |
| Detection model | `fasterrcnn_mobilenet_v3_large_fpn` (torchvision) | Light, COCO-pretrained detector (BSD-licensed code) |
| Metrics | `torchmetrics` (MeanAveragePrecision; needs `pycocotools` or `faster-coco-eval`) | mAP and per-class AP |
| Data handling | NumPy, Pandas, Pillow, OpenCV (`opencv-python-headless`) | IO, box drawing, EDA |
| Splitting & stats | scikit-learn | Group splits, classification-style summaries |
| Plotting | Matplotlib, Seaborn | EDA and result plots |
| Config | PyYAML | Hyperparameters and paths |
| UI | Streamlit | Optional demo app |
| Tests | pytest | Unit and integration tests |
| Notebooks | Jupyter | EDA only (not required to run the pipeline) |
| VCS | Git + GitHub | Source control |

**Versions.** Do not pin ancient versions. Install the latest stable PyTorch/torchvision pair compatible with your Python, then freeze exact versions into `requirements.txt` after the smoke test passes. Minimum guidance (re-check on pytorch.org): `torch>=2.2`, `torchvision` matching that torch, `numpy`, `pandas`, `scikit-learn`, `torchmetrics`, `streamlit`, `pyyaml`, `pytest`.

**Why not Ultralytics YOLO as primary?** It is a strong, easy option, but its license is AGPL-3.0 **[Unverified here; agent should check]**, which can complicate sharing. torchvision keeps one consistent framework and license-friendly code. YOLO is listed as an optional comparison in Section 21.

**Hardware.** Must run on a laptop CPU for the smoke test and inference. Full training is practical on a GPU (local, or free-tier cloud notebook); on CPU, use the reduced `configs/laptop.yaml` (subset, small image size).

---

## 9. System Architecture

```text
            ┌──────────────────────────────┐
            │ CropAndWeed (official script)│
            └──────────────┬───────────────┘
                           ▼
        scripts/prepare_dataset.py  ──►  data/processed/{splits.csv, stats.json}
                           ▼
  src/data/dataset.py  (CSV boxes → tensors; label map; augmentation for train only)
                           ▼
  src/models/detector.py (Faster R-CNN MobileNetV3-FPN, 3-class head)
                           ▼
  src/training/train.py ──► models/best.pt + models/class_mapping.json + outputs/logs
                           ▼
  src/evaluation/evaluate.py ──► outputs/reports/*.json, outputs/plots/*.png
                           ▼
  src/inference/predict.py ──► outputs/predictions/*.jpg|json
                           ▼
  app/streamlit_app.py (uploads image → same predict code → boxes + counts)
```
Principle: **one shared preprocessing/inference module** is used by training-time evaluation, the CLI, and the Streamlit app, so preprocessing cannot drift.

---

## 10. Project Directory Structure

```text
weed-vs-crop-discrimination/
├── README.md                     # Setup, commands, observed results, progress log
├── WEED_VS_CROP_DL_PROJECT_SPECIFICATION.md   # This file
├── requirements.txt              # Frozen versions after smoke test
├── .gitignore                    # Excludes data, checkpoints, third_party, venv
├── configs/
│   ├── default.yaml              # Paths, hyperparameters (full run)
│   ├── laptop.yaml               # Small image size, subset, few epochs
│   └── smoke.yaml                # ~20 images, 1 epoch, for pipeline testing
├── docs/
│   ├── dataset_notes.md          # Observed file layout, header/no-header, versions
│   └── progress_log.md           # Agent checklist and status (see Section 23)
├── data/
│   ├── raw/                      # Downloaded dataset (NOT committed)
│   └── processed/                # splits.csv, stats.json (small files may be committed only if license allows; image-derived data is not)
├── notebooks/
│   └── 01_eda.ipynb              # Optional exploratory analysis
├── src/
│   ├── __init__.py
│   ├── config.py                 # YAML loader + dataclass/dict validation
│   ├── data/
│   │   ├── parse_annotations.py  # Read bbox CSVs, map labels, drop invalid boxes
│   │   ├── splits.py             # Session-grouped split creation + verification
│   │   └── dataset.py            # torch Dataset, transforms, collate_fn
│   ├── models/
│   │   ├── baseline.py           # Patch classifier baseline
│   │   └── detector.py           # Faster R-CNN builder, checkpoint save/load
│   ├── training/
│   │   └── train.py              # Train/val loops, scheduler, early stopping
│   ├── evaluation/
│   │   ├── metrics.py            # mAP, precision/recall/F1 at IoU, matching
│   │   └── evaluate.py           # Test-set run + plots + report
│   ├── inference/
│   │   └── predict.py            # Load model, preprocess, predict, draw, JSON
│   └── utils/
│       └── seed.py, logging.py, viz.py
├── scripts/
│   ├── download_dataset.sh       # Wrapper around the official setup (documented)
│   ├── check_env.py              # Python/torch/CUDA check
│   ├── prepare_dataset.py        # Validate files, EDA stats, splits
│   └── make_sample_predictions.py
├── tests/
│   ├── test_labels.py, test_splits.py, test_dataset.py,
│   ├── test_model.py, test_checkpoint.py, test_inference.py
├── models/                       # best.pt + class_mapping.json (NOT committed by default)
├── outputs/
│   ├── plots/                    # curves, PR curves, class distributions
│   ├── reports/                  # metrics.json, eval_report.md
│   └── predictions/              # annotated sample images + JSON
├── app/
│   └── streamlit_app.py
└── third_party/                  # cloned official dataset tools (NOT committed)
```
Do not create empty placeholder modules. Create a file only when it contains working code.

---

## 11. Detailed Implementation Roadmap

### Phase 1: Environment setup
1. Print OS, Python version, free disk (`df -h .`), RAM, CPU cores.
2. Create venv: `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).
3. Install PyTorch/torchvision per https://pytorch.org/get-started/locally/ (CPU build if no NVIDIA GPU), then the rest of the dependencies.
4. Run `python scripts/check_env.py`: prints versions, `torch.cuda.is_available()`, device chosen.

### Phase 2: Dataset acquisition
Follow Section 6.4. Output: `data/raw/cropandweed/` populated, `docs/dataset_notes.md` written with the true layout. Paths come from the config, never hard-coded.

### Phase 3: Exploratory data analysis (`scripts/prepare_dataset.py` + optional notebook)
- Count images and **instances per class** (crop vs weed) overall and per session.
- Detect missing images, unreadable images (try `PIL.Image.open().verify()`), CSVs with no boxes, boxes with zero/negative area or outside image bounds.
- Image size distribution (the paper mentions 1920×1088 for the Fine24 statistics; verify actual sizes).
- Box size distribution (tiny / small / medium / large) since small objects are the main difficulty.
- Class imbalance: instances crop vs weed; images with zero weeds.
- Visualize ≥ 12 random images with boxes (colour by class) → `outputs/plots/samples.png`.
- Plot distributions → `outputs/plots/class_distribution.png`, `box_size_hist.png`.
- Near-duplicate check: perceptual hash (e.g., `imagehash`, optional) within and across sessions; report how many near-duplicates cross session boundaries.
- Document biases: geography (Austria per the AIT page/paper **[verify]**), handheld camera, limited crop set, lighting/soil conditions via `params`.

### Phase 4: Dataset preparation
- Parse annotations → internal table: `image_path, x1, y1, x2, y2, label`.
- Remove/clip invalid boxes; log how many.
- Create session-grouped splits (Section 7.5); write `splits.csv`.
- **Resizing:** the detector resizes internally (`min_size`/`max_size` in config); do not pre-resize files on disk.
- **Normalization:** torchvision detection models normalize internally with ImageNet statistics; supply tensors in `[0,1]`. Do not normalize twice.
- **Augmentation (train only):** horizontal flip (p=0.5), small brightness/contrast/hue jitter, optional vertical flip, optional random scale/crop that keeps boxes valid (drop boxes with < ~30–40% of area left, configurable). Boxes must be transformed together with the image.
- **Validation/test:** no augmentation, same resize as inference.

### Phase 5: Models

**5a. Baseline (simple, quick).**
Patch classifier: crop every annotated box from the **training** images (with small padding), resize to 96×96 (or 64×64), train a small 3-conv-layer CNN for crop vs weed. Evaluate on patches from the **test** split. Purpose: shows how separable crop/weed is *when the plant is already located*, and supplies the classification-style metrics (accuracy, precision, recall, F1, confusion matrix). It is **not** the final system, because it needs boxes as input.

Also compute a trivial reference: majority-class patch accuracy, so readers can see the baseline is better than guessing.

**5b. Primary: transfer-learning detector.**
- Architecture: `torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn` with COCO-pretrained weights.
- Replace the box predictor head: `FastRCNNPredictor(in_features, num_classes=3)` (background, crop, weed).
- Transfer learning explanation: the backbone/FPN already learned generic edges, textures, shapes from large datasets (ImageNet/COCO); fine-tuning adapts the later layers and the box heads to plants.
- **Freezing schedule:** Stage 1 (epochs 1–2): freeze backbone body (`trainable_backbone_layers=0`-style behaviour), train FPN + RPN + ROI heads. Stage 2: unfreeze the last backbone blocks at a lower learning rate. Make this configurable.
- Loss: the built-in sum of RPN objectness/box losses and ROI classification/box-regression losses (returned by the model in training mode).
- Optimizer: SGD (momentum 0.9, weight decay 5e-4, lr ≈ 0.005–0.01) **or** AdamW (lr ≈ 1e-4). Choose one in config, run a short LR sanity check (a few hundred iterations at 2–3 values), pick the one with stable decreasing loss. Do not hard-claim a "best" value.
- Scheduler: warm-up for the first few hundred iterations + cosine or StepLR.
- Batch size: as large as memory allows (CPU/laptop 2–4, GPU 8–16); use gradient accumulation if needed.
- Epochs: smoke 1, laptop ~5–10 on a subset, full ~20–30 with early stopping on validation mAP.
- Overfitting controls: augmentation, weight decay, frozen early layers, early stopping on validation mAP, session-grouped validation.

### Phase 6: Training pipeline (`src/training/train.py`)
- Fixed seeds (`random`, `numpy`, `torch`, CUDA), `torch.backends.cudnn.deterministic` option; document that exact bit-reproducibility on GPU is not guaranteed.
- Device: `cuda` if available else `cpu` (override via config/CLI).
- Per epoch: train loop (log running total and component losses), validation loop (compute mAP@[.5:.95] and mAP@.5, per-class AP, and precision/recall/F1 at IoU 0.5 and the configured score threshold).
- Save `models/last.pt` each epoch, `models/best.pt` when validation mAP improves; include in the checkpoint: model state, class mapping, config snapshot, torch/torchvision versions, git commit hash if available, epoch, metrics.
- Early stopping with `patience` from config.
- Write a CSV log `outputs/reports/train_log.csv` for plotting.
- Mixed precision (`torch.autocast`) only when CUDA is available.

### Phase 7: Evaluation (`src/evaluation/evaluate.py`)
Run once on the untouched **test** split with `best.pt`:
- **mAP@0.5** and **mAP@[0.5:0.95]**, plus **per-class AP** (crop, weed) via torchmetrics.
- **Precision, recall, F1 for the `weed` class** at IoU=0.5 and a stated score threshold (greedy matching of predictions to ground truth by IoU, one-to-one).
- **Detection confusion matrix** with rows/cols: crop, weed, background(missed/false alarm). A prediction of the wrong class that overlaps a ground-truth box at IoU ≥ 0.5 counts as a crop↔weed confusion.
- **PR curves** for each class and a **threshold sweep** table (threshold → weed precision, recall, F1).
- **Learning curves:** training vs validation loss, and validation mAP per epoch.
- **Size breakdown:** AP for small/medium/large boxes.
- **Subgroup breakdown** using `params` (lighting, soil, moisture, separability): is performance worse in hard separability or particular lighting?
- If classes are imbalanced, report **macro-averaged F1** and per-class recall, not just a single aggregate.

How to interpret:
- *mAP@0.5* is lenient about box tightness; *mAP@[.5:.95]* rewards precise boxes.
- *Precision* answers "when it says weed, how often is it right?" (relates to crop-damage risk). *Recall* answers "how many real weeds did it find?" (relates to missed weeds).
- *Overfitting signs:* training loss keeps falling while validation loss/mAP stagnates or worsens; large gap between train-subset mAP and validation mAP; much worse test than validation performance.
- Do not state any expected accuracy before training. Report only measured numbers, and say which split they come from.

### Phase 8: Inference (`src/inference/predict.py`)
CLI:
```bash
python -m src.inference.predict --image path/to/field.jpg --checkpoint models/best.pt \
       --score-threshold 0.5 --out outputs/predictions/
```
Requirements:
- Load checkpoint + `class_mapping.json`; `map_location="cpu"` fallback so CPU works without CUDA.
- Open image with Pillow, convert to RGB; clear errors for: missing file, unreadable/corrupt file, unsupported extension, zero-size image.
- Same preprocessing as validation (convert to tensor in `[0,1]`; model's internal resize/normalize). No training augmentation.
- Apply score threshold and NMS (built into the model; configurable IoU).
- Output: annotated image (green = crop, red = weed, score printed), JSON `{image, width, height, detections:[{label, score, box}], summary:{n_crop, n_weed}}`.
- Exit non-zero on failure.

Testing on unseen images: use (a) the held-out **test** split (already evaluated), and (b) a few photos **not from the dataset** (own phone photos of a field/garden, or images from a different public source). State clearly that (b) is qualitative only unless labelled.

### Phase 9: Streamlit app (optional, `app/streamlit_app.py`)
- Upload JPG/PNG → preview → "Detect" button → annotated image.
- Slider for confidence threshold (default from config).
- Table of detections with labels and confidence; counts of crop and weed.
- Show a warning banner when all detections are below e.g. 0.6 or when no plants are detected: "Low confidence; results may be unreliable for this image."
- Disclaimer: research prototype; not for autonomous herbicide application.
- Uses `src.inference.predict` functions with the real checkpoint (cache model with `st.cache_resource`). If the checkpoint is missing, show an error; **never** return mock predictions.
- Run: `streamlit run app/streamlit_app.py`

### Phase 10: Testing
See Section 17.

### Phase 11: Documentation & deliverables
See Sections 19 and 22.

### Phase 12 (optional extension): Semantic segmentation
Use the `labelIds/` masks of a mapped variant. Add a second model (e.g., torchvision `lraspp_mobilenet_v3_large` or `deeplabv3_mobilenet_v3_large`) with 3 classes (soil, crop, weed) and report mIoU and per-class IoU. Keep it in a separate module (`src/segmentation/`) and a separate README section; do not mix its metrics with the detection results.

---

## 12. Model Architecture and Rationale

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Faster R-CNN MobileNetV3-Large FPN (torchvision)** | COCO-pretrained, native torchvision, relatively light, good with small objects thanks to FPN | Slower than single-shot detectors on CPU | **Primary** |
| Faster R-CNN ResNet-50 FPN | Higher accuracy potential | Heavy for laptops | Optional GPU comparison |
| RetinaNet / SSDlite (torchvision) | Single-stage, simpler | SSDlite weak on small objects | Optional |
| Ultralytics YOLO | Very easy training, fast | Different framework; AGPL-3.0 licensing consideration **[verify]** | Optional comparison only |

Model builder sketch (the agent should implement and test it; this is illustrative):
```python
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_detector(num_classes=3, min_size=640, max_size=1024, pretrained=True):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, min_size=min_size, max_size=max_size)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model
```
**Important:** `min_size`/`max_size` strongly affect both small-weed recall and speed. Treat them as tunable and record their value in the checkpoint. If the weights download is blocked, document it and use the manual-download workaround; do not silently train from scratch and call it transfer learning.

---

## 13. Training Methodology

- **Data loading:** `DataLoader` with `collate_fn=lambda b: tuple(zip(*b))`, `num_workers` from config (use 0 on Windows if issues).
- **Targets:** dict with `boxes` (float32, xyxy, absolute pixels), `labels` (int64, 1=crop, 2=weed), `image_id`. Images without boxes are allowed (empty tensors of shape `[0,4]`) or filtered, per config.
- **Hyperparameters (config example):**
```yaml
# configs/default.yaml
seed: 42
paths:
  raw_dir: data/raw/cropandweed
  processed_dir: data/processed
  models_dir: models
  outputs_dir: outputs
data:
  variant: CropOrWeed2
  split: {train: 0.70, val: 0.15, test: 0.15, group_by: session}
  subset_fraction: 1.0          # laptop.yaml: 0.2 ; smoke.yaml: ~20 images
model:
  name: fasterrcnn_mobilenet_v3_large_fpn
  num_classes: 3
  min_size: 640
  max_size: 1024
  pretrained: true
train:
  epochs: 20
  batch_size: 4
  optimizer: sgd               # sgd | adamw
  lr: 0.005
  momentum: 0.9
  weight_decay: 0.0005
  warmup_iters: 300
  scheduler: cosine
  freeze_backbone_epochs: 2
  early_stopping_patience: 5
  num_workers: 2
eval:
  iou_threshold: 0.5
  score_threshold: 0.5
inference:
  score_threshold: 0.5
  nms_iou: 0.5
```
- **Order of runs:** (1) `smoke.yaml` end-to-end; (2) `laptop.yaml`; (3) `default.yaml` on available hardware.
- **Best model rule:** highest validation mAP@[.5:.95] (ties broken by validation weed F1).
- **Compute honesty:** the agent logs actual wall-clock time and device; no estimates are presented as results.

---

## 14. Evaluation Methodology

1. Use the test split **only once** for the final report (after hyperparameters are fixed on validation).
2. Report: mAP@0.5, mAP@[.5:.95], per-class AP, weed precision/recall/F1 at the chosen threshold, detection confusion matrix, threshold sweep, size-bucket AP, params-subgroup table.
3. Include the patch-baseline metrics (accuracy, precision, recall, F1, confusion matrix) for context, labelled as a *different task*.
4. Compare to published numbers only qualitatively; the CropAndWeed paper's results come from its own protocol and variants. Quote them only after reading the paper, with citation, and label them "published results", never as this project's results.
5. Save: `outputs/reports/metrics.json`, `outputs/reports/eval_report.md`, plots in `outputs/plots/`, 20 annotated qualitative examples (10 correct, 10 failures) in `outputs/predictions/`.
6. Add an error analysis paragraph: where does the model fail (tiny weeds, occlusion, crop-like weeds, shadows, soil clutter)?

---

## 15. Inference Implementation

Public functions in `src/inference/predict.py` (shared by CLI, tests, Streamlit):
```python
def load_model(checkpoint_path: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]: ...
def predict_image(model, image: PIL.Image.Image, score_threshold: float, device: str) -> list[dict]: ...
def draw_detections(image: PIL.Image.Image, detections: list[dict]) -> PIL.Image.Image: ...
def summarize(detections: list[dict]) -> dict: ...  # counts, mean confidence, weed share
```
Preprocessing is defined once (RGB → float tensor `[0,1]`); the same function builds validation inputs in training. Add a test that the two paths produce identical tensors for the same image.

---

## 16. Optional Streamlit Application

Required behaviour listed in Phase 9. Keep to one file (`app/streamlit_app.py`) plus imports from `src`. No API keys. Handles: no file uploaded, invalid image, missing checkpoint, no detections.

---

## 17. Testing and Validation Plan

| Area | Test | Pass condition |
|---|---|---|
| Dataset loading | Load 5 samples | Tensors have correct dtype/shape; boxes within image; labels in {1,2} |
| Label mapping | Map a known species ID from each group | e.g., ID 1 → crop(1), ID 31 → weed(2), an unmapped ID → excluded |
| Parsing | Malformed CSV row / zero-area box | Skipped with a logged warning, no crash |
| Splits | Session sets of train/val/test | **Zero overlap**; all splits non-empty; both classes present in each split (or documented exception) |
| Reproducibility | Same seed → same `splits.csv`; same first-batch order | Identical |
| Model output | Forward pass in train and eval modes | Train returns loss dict; eval returns boxes/labels/scores; head has 3 classes |
| Checkpoint | Save then load | Same class mapping; same outputs on a fixed input (tolerance) |
| Inference (valid) | Run on a real image | JSON + annotated image written |
| Inference (invalid) | Missing path, text file renamed `.jpg`, 0-byte file | Clear error message, non-zero exit |
| CPU inference | Force `device=cpu` | Works |
| GPU inference | Skip with reason if no CUDA | Works when available |
| Preprocessing parity | Training-val transform vs inference transform | Identical tensors |
| Streamlit | App imports and loads model | No mock outputs; clear error when checkpoint missing |

Run: `pytest -q`. The agent must report actual pass/fail output.

**Pipeline checklist:** env verified → dataset downloaded and verified → EDA outputs exist → splits leakage-free → smoke run completes → baseline trained → detector trained → test evaluation produced → inference CLI works on unseen image → Streamlit works → tests pass → README updated.

---

## 18. Commands to Install, Train, Evaluate, and Run

```bash
# 0. Clone / open workspace, then:
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
# Install torch + torchvision for your platform from https://pytorch.org/get-started/locally/
pip install numpy pandas scikit-learn matplotlib seaborn pillow opencv-python-headless \
            pyyaml torchmetrics pycocotools streamlit pytest tqdm
python scripts/check_env.py

# 1. Dataset (see Section 6.4)
bash scripts/download_dataset.sh

# 2. Validate + EDA + splits
python scripts/prepare_dataset.py --config configs/default.yaml

# 3. Smoke test (tiny, CPU-friendly)
python -m src.training.train --config configs/smoke.yaml

# 4. Baseline patch classifier
python -m src.models.baseline --config configs/default.yaml

# 5. Main training
python -m src.training.train --config configs/laptop.yaml     # or default.yaml on GPU

# 6. Evaluation on test split
python -m src.evaluation.evaluate --config configs/default.yaml --checkpoint models/best.pt

# 7. Inference
python -m src.inference.predict --image path/to/image.jpg --checkpoint models/best.pt

# 8. App
streamlit run app/streamlit_app.py

# 9. Tests
pytest -q
```
If `pycocotools` fails to build on Windows, try `faster-coco-eval` as the torchmetrics backend and document it.

---

## 19. Expected Outputs and Deliverables

- Source code, configs, tests.
- `requirements.txt` (frozen), README with exact commands and **observed** results.
- `docs/dataset_notes.md`, `docs/progress_log.md`.
- `models/best.pt`, `models/class_mapping.json` (local; see Git rules).
- `outputs/plots/`: samples, class distribution, box-size histogram, loss/mAP curves, PR curves, confusion matrix.
- `outputs/reports/`: `metrics.json`, `eval_report.md`, `train_log.csv`.
- `outputs/predictions/`: annotated test images + images not from the dataset.
- Streamlit app.

---

## 20. Common Errors and Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `setup.py` download fails | Network/host change | Read error; check repo issues/README; manual download to `data/raw/` and record steps |
| `CUDA out of memory` | Batch/size too large | Lower `batch_size`, `max_size`; use gradient accumulation |
| Training very slow on CPU | Large images, many epochs | Use `laptop.yaml`, subset, `min_size=480` |
| Loss is `nan` | LR too high, degenerate boxes | Lower LR, filter boxes with width/height ≤ 1 px, add grad clipping |
| `AssertionError: All bounding boxes should have positive height and width` | Zero-area boxes after augmentation | Filter after every transform |
| mAP = 0 | Label off-by-one (background must be 0) | Check mapping `{1:crop, 2:weed}` |
| Wrong colors / predictions differ in app | Preprocessing mismatch (e.g., BGR from OpenCV) | Always convert to RGB with Pillow; use shared function |
| `pycocotools` build errors | Missing compiler | Use prebuilt wheel / `faster-coco-eval` |
| Windows DataLoader hangs | Multiprocessing | `num_workers: 0` |
| Weights can't download | Offline/proxy | Manual download of torchvision weights to torch cache; document it |

---

## 21. Limitations and Future Improvements

**Limitations (must appear in the README):**
- The model sees only what the dataset contains: specific crops, regions, and handheld-style photos. **Generalization to unseen farms, cameras, growth stages, and lighting is not guaranteed** and should be tested rather than assumed.
- Session-grouped splits reduce, but do not eliminate, same-field leakage; the test set still comes from the same collection campaign.
- Class imbalance and tiny instances reduce recall for small weeds.
- Weed class lumps dozens of species; some resemble crops (especially at early growth stages).
- Bounding boxes include background pixels; they are not exact plant outlines.
- Dataset license is non-commercial; this project is educational.
- The model is a research prototype, not validated for autonomous herbicide application.

**Future improvements (optional):**
1. Semantic segmentation (Phase 12) for weed coverage area.
2. Weed species classification via `Fine24`.
3. Cross-dataset tests (Sugar Beets 2016, PhenoBench) for domain shift.
4. Heavier backbone or YOLO comparison on GPU.
5. Test-time augmentation, hard-example mining, class-weighting.
6. Export to ONNX/TorchScript; video inference; embedded deployment.
7. Self-collected local images with your own labels (LabelImg, CVAT) to adapt to your target field.

---

## 22. Git and GitHub Workflow

```bash
git init
git checkout -b main
git add . && git commit -m "Initial project structure"
# Feature branches per phase, e.g.:
git checkout -b feat/data-pipeline
```
Commit messages: one logical change each (`data: add session-grouped splits`, `train: add early stopping`).

**Commit:** source, configs, tests, docs, README, small plots/reports (aggregate metrics, plots without raw dataset images are fine; annotated dataset images are derived data, so keep them out of a public repo).
**Do not commit (add to `.gitignore`):**
```gitignore
.venv/
__pycache__/
*.pyc
data/raw/
data/processed/*.pt
third_party/
models/*.pt
outputs/predictions/*
*.zip
.ipynb_checkpoints/
.env
```
**Dataset images and masks:** never public (license). **Checkpoints:** keep local or use Git LFS / a private release / cloud storage only if the dataset license allows sharing models trained on it **[verify]**. If `models/best.pt` is excluded, README must explain how to regenerate it.

---

## 23. Antigravity Execution Instructions

**Instructions to the AI coding agent (follow literally):**

1. **Read this entire specification** before changing anything.
2. **Inspect the workspace** (`ls`, `git status`). Do not overwrite existing files without reading them; if a file exists, extend it or ask.
3. **Create `docs/progress_log.md`** with the checklist below; update it after every phase with status (`not started / in progress / done / blocked`) and evidence (command output summaries).
4. **Verify the dataset** at the official URLs (Section 6.2): confirm the annotation format, variants (`CropOrWeed2`), license text, and whether any registration/terms acceptance exists. Write findings to `docs/dataset_notes.md`.
5. **Check disk space** (`df -h .`) before downloading anything large. If space is insufficient, stop and report.
6. Implement in the **small phases** of Section 11, in order. After each phase, run what you built.
7. **Set up the environment** (Phase 1) and run `scripts/check_env.py`.
8. **Implement dataset inspection and validation before any training.** Verify label mapping (Section 7.2) with unit tests and **verify split integrity** (zero session overlap).
9. Implement and test the **baseline patch classifier** (Phase 5a) on the smoke config.
10. Implement the **Faster R-CNN MobileNetV3 detector** (Phase 5b); run `smoke.yaml` to confirm the loop works.
11. **Train** with the config appropriate to the detected hardware (smoke → laptop → default). Log the device and wall-clock time.
12. **Evaluate on the test set exactly as in Phase 7**; produce the plots and reports.
13. Implement **inference** (Phase 8) and the **Streamlit app** (Phase 9) using the real checkpoint.
14. **Run `pytest`** and fix failures. Do not claim tests pass without running them.
15. **Update README** with actual commands and **observed** results only.
16. **Report** which phases are complete and which are incomplete or blocked.

**Hard constraints**
- Do not create fake datasets or synthetic evaluation results and present them as real. The Kaggle sesame set may be used only for pipeline smoke testing and must be labelled so.
- Do not claim training, tests, or metrics that were not actually executed.
- Do not silently change the dataset, task (detection), label mapping, or architecture. If a change is necessary, record why in the progress log and tell the user.
- Do not push dataset files or checkpoints to a public remote.
- Do not expose secrets; do not hard-code absolute or machine-specific paths.
- If the dataset cannot be accessed, **stop and explain the blocker**; do not invent data.
- Prefer readable, modular code with type hints, docstrings, and logging (see the code quality rules directly below).

**Code quality rules:** type hints on public functions; `logging` instead of bare `print` in library code; clear error messages; config-driven paths/hyperparameters; one shared preprocessing function; separate training/evaluation/inference modules; minimal dependencies; no dead code or empty modules.

---

## 24. Completion Checklist

- [x] **Environment created; versions recorded; device detected** (Python 3.11.9, PyTorch 2.5.1+cu121, CUDA True, RTX 2050 4GB; verified via `scripts/check_env.py` and frozen in `requirements.txt`)
- [x] **CropAndWeed downloaded via official route; license read and noted** (Cloned in `third_party/cropandweed-dataset`; official tar archives unpacked; documented in `docs/dataset_notes.md`)
- [x] **`CropOrWeed2` annotations present; layout documented** (7,705 CSVs mapped, layout and columns documented in `docs/dataset_notes.md`)
- [x] **EDA complete: counts, corrupt files, box stats, imbalance, sample plots** (2,003 valid images, 0 corrupt, 19,785 annotations, 74.1% weeds; plots saved in `outputs/plots/`)
- [x] **Session-grouped splits created; leakage test passes** (1,381 train / 338 val / 284 test in `data/processed/splits.csv`; 0% leakage verified by `tests/test_splits.py`)
- [x] **Label mapping tests pass (crop=1, weed=2, background=0)** (Verified by `tests/test_labels.py`)
- [x] **Smoke run completes end to end** (`python -m src.training.train --config configs/smoke.yaml`)
- [x] **Baseline patch classifier trained and evaluated** (Evaluated on 1,000 patches: Weed Prec=91.09%, Weed Rec=77.32%, Weed F1=83.65% in `outputs/reports/baseline_metrics.json`)
- [x] **Detector trained; `best.pt` + `class_mapping.json` saved** (Faster R-CNN MobileNetV3-Large FPN saved in `models/best.pt` [151.8 MB] and `models/class_mapping.json`)
- [x] **Test-set evaluation with mAP, per-class AP, weed P/R/F1, confusion matrix, curves** (mAP@0.5: 0.4091, mAP: 0.1957, Crop AP: 0.3247, Weed AP: 0.0667 in `outputs/reports/metrics.json`, `outputs/reports/eval_report.md`, `outputs/plots/pr_curves.png`)
- [x] **Error analysis written** (Documented in `outputs/reports/eval_report.md` Section 4 and `README.md` Section 6)
- [x] **Inference CLI works on valid image; fails cleanly on invalid input; works on CPU** (`src/inference/predict.py` tested and verified in `tests/test_inference.py`)
- [x] **At least a few non-dataset images run and shown (qualitative only)** (Generated and saved in `outputs/predictions/non_dataset_test_sample_annotated.jpg` and `non_dataset_test_sample_detections.json`)
- [x] **Streamlit app runs with the real model** (`app/streamlit_app.py` with CUDA/CPU toggle, threshold slider, KPI summary cards, and color-coded bounding boxes)
- [x] **`pytest` passes (actual output captured)** (All 12 unit & integration tests passing in `tests/`)
- [x] **README has commands, observed results, limitations** (Comprehensive dedicated quickstart section, observed metrics table, and risk analysis in `README.md`)
- [x] **`.gitignore` excludes data, checkpoints, third_party** (Verified `.gitignore`)
- [x] **Progress log final: phases complete/incomplete reported honestly** (Updated and verified in `docs/progress_log.md`)

---

## 25. References and Dataset Citations

> URLs below were opened or surfaced during research unless marked otherwise. The agent should confirm each is live.

**Primary dataset**
1. Steininger, D., Trondl, A., Croonen, G., Simon, J., Widhalm, V. *The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation.* WACV 2023, pp. 3729–3738. Paper: https://openaccess.thecvf.com/content/WACV2023/papers/Steininger_The_CropAndWeed_Dataset_A_Multi-Modal_Learning_Approach_for_Efficient_Crop_WACV_2023_paper.pdf. Supports: dataset composition, annotation types, protocol.
2. CropAndWeed repository (annotation format, variants, setup scripts, license note): https://github.com/cropandweed/cropandweed-dataset
3. Variant/label definitions (`CropOrWeed2`): https://github.com/cropandweed/cropandweed-dataset/blob/main/cnw/utilities/datasets.py
4. License text: https://github.com/cropandweed/cropandweed-dataset/blob/main/LICENCE
5. AIT dataset page: https://publications.ait.ac.at/en/datasets/cropandweed-dataset/ (date made available Jan 2023; production 2019–2022).
6. Hugging Face dataset card (secondary, 8,034 samples, license summary): https://huggingface.co/datasets/Voxel51/CropAndWeed

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

**Compared datasets**
7. Olsen et al. *DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning.* Scientific Reports, 2019. https://www.nature.com/articles/s41598-018-38343-3 ; repo https://github.com/AlexOlsen/DeepWeeds (CC BY 4.0).
8. Sa et al. *WeedMap.* Remote Sensing 2018. https://doi.org/10.3390/rs10091423 ; dataset page https://projects.asl.ethz.ch/datasets/weedmap-2018/ ; arXiv https://arxiv.org/abs/1808.00100
9. Weyler et al. *PhenoBench.* arXiv 2306.04557, https://arxiv.org/abs/2306.04557 ; https://www.phenobench.org
10. Milioto, Lottes, Stachniss. *Real-time Semantic Segmentation of Crop and Weed for Precision Agriculture Robots Leveraging Background Knowledge in CNNs.* https://arxiv.org/abs/1709.06764 (Bonn sugar beet context); download walkthrough: https://justdeepit.readthedocs.io/en/stable/tutorials/weed-segmentation.html
11. Kaggle sesame crop/weed boxes dataset: https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes
12. WeedsGalore: https://arxiv.org/abs/2502.13103 ; https://github.com/GFZ/weedsgalore
13. Curated list of agriculture datasets (secondary discovery aid): https://github.com/ricber/digital-agriculture-datasets

**Methods and tools (agent to confirm live URLs; these were not opened in this session)**
14. Ren et al. *Faster R-CNN.* https://arxiv.org/abs/1506.01497 : detector architecture.
15. Howard et al. *Searching for MobileNetV3.* https://arxiv.org/abs/1905.02244 : backbone.
16. Lin et al. *Feature Pyramid Networks for Object Detection.* https://arxiv.org/abs/1612.03144 : FPN for small objects.
17. torchvision detection models: https://pytorch.org/vision/stable/models/faster_rcnn.html and the object-detection fine-tuning tutorial on pytorch.org.
18. torchmetrics `MeanAveragePrecision`: https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
19. PyTorch install selector: https://pytorch.org/get-started/locally/
20. Streamlit docs: https://docs.streamlit.io

**Published vs. project results.** Any numbers taken from the papers above are *published results under their own protocols*. Numbers produced by running this repository are *project results*. Keep them in separate tables and never mix them.
