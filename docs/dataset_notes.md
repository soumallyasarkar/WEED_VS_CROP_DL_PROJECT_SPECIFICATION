# CropAndWeed Dataset Notes and Verification

This document records the official verification of the CropAndWeed dataset as mandated by Section 23 of `WEED_VS_CROP_DL_PROJECT_SPECIFICATION.md`.

## 1. Official Sources and URLs
- **Official Repository**: https://github.com/cropandweed/cropandweed-dataset
- **Paper**: Steininger et al., *The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation*, WACV 2023. [Open Access PDF](https://openaccess.thecvf.com/content/WACV2023/papers/Steininger_The_CropAndWeed_Dataset_A_Multi-Modal_Learning_Approach_for_Efficient_Crop_WACV_2023_paper.pdf)
- **Institutional Page**: Austrian Institute of Technology (AIT) GmbH [AIT Dataset Page](https://publications.ait.ac.at/en/datasets/cropandweed-dataset/)
- **Hosting Endpoint**: `https://vitro-testing.com/wp-content/uploads/2022/12/`
  - `cropandweed_annotations.tar`: 105.50 MB (verified HTTP 200)
  - `cropandweed_images1of4.tar`: 2,842.84 MB (2.78 GB) (verified HTTP 200)
  - `cropandweed_images2of4.tar`: 2,859.63 MB (2.79 GB) (verified HTTP 200)
  - `cropandweed_images3of4.tar`: 2,460.06 MB (2.40 GB) (verified HTTP 200)
  - `cropandweed_images4of4.tar`: 2,670.77 MB (2.61 GB) (verified HTTP 200)

## 2. License and Registration Terms
- **Registration**: None. The dataset archives are directly accessible without sign-up or authorization tokens.
- **License**: Custom Non-Commercial License (AIT Austrian Institute of Technology GmbH).
  - Academic and non-commercial purposes only (research, teaching, personal experimentation).
  - Attribution / Citation of the WACV 2023 publication is mandatory.
  - **No redistribution** of the raw or modified dataset images/annotations.
  - **Permissible**: Distributing derivative works that are *abstract representations* (such as trained model weights `models/best.pt` or aggregate metrics/plots) as long as they do not allow recovery of the original data.
  - Strictly no commercial use or licensing for commercial gain.

## 3. Directory Layout and File Formats
When unpacked, the dataset creates:
- `images/`: High-resolution field photographs (`.jpg`).
  - Filename format: `{prefix}-{session:04d}-{image_id:04d}.jpg`
  - `prefix`: `ave` (Application Set) or `vwg` (Experimental Set).
  - `session`: 4-digit recording session ID (indicates same field, lighting, camera position).
  - `image_id`: 4-digit frame identifier.
- `bboxes/{variant}/`: CSV annotations per image (e.g., `ave-0020-0010.csv`).
  - **Header row**: NONE (No header present in raw files).
  - **Columns**: `left, top, right, bottom, label_id, stem_x, stem_y` (7 columns).
  - Coordinates are in absolute pixel values (`float` or `int`).
- `labelIds/{variant}/`: Semantic segmentation masks (`.png`, 1-channel uint8).
- `params/`: CSV metadata per image containing:
  - `moisture`: 0 (dry), 1 (medium), 2 (wet)
  - `soil`: 0 (fine), 1 (medium), 2 (coarse)
  - `lighting`: 0 (sunny), 1 (diffuse)
  - `separability`: 0 (easy), 1 (medium), 2 (hard)

## 4. Label Mapping: `CropOrWeed2`
Defined in `cnw/utilities/datasets.py`:
- **Raw Class 0 (`Crop`)**: Maps species IDs:
  `[1, 2, 3, 4, 5, 6]` (Maize stages), `[7, 8, 9, 10, 11, 12]` (Sugar beet stages),
  `94` (Soybean), `24` (Sunflower), `18` (Potato), `13` (Pea), `[26, 27]` (Beans), `15` (Pumpkin).
- **Raw Class 1 (`Weed`)**: Maps 58 weed species IDs:
  `[31, 48, 62, 65, 68, 69, 74, 75, 81, 84, 86, 32, 29, 33, 37, 49, 30, 44, 66, 87, 89, 91, 61, 79, 34, 41, 52, 35, 36, 78, 38, 39, 71, 72, 88, 42, 45, 70, 47, 51, 54, 58, 60, 80, 83, 96, 22, 63, 85, 56, 57, 64, 77, 50, 59, 67, 76]`.
- **Excluded Species**: Any species IDs not listed (e.g. parsley, garlic, chives, asparagus) are excluded from `CropOrWeed2`.
- **Project-Internal Label Convention (TorchVision)**:
  TorchVision object detection reserves index 0 for background:
  - `0`: `background` (implicit)
  - `1`: `crop` (`CropOrWeed2` label 0 + 1)
  - `2`: `weed` (`CropOrWeed2` label 1 + 1)

## 5. Splitting and Leakage Prevention
- Images from the same recording session (`prefix-session`) must never be divided across train/val/test splits.
- Splitting is performed using group splitting on `(prefix, session)` into:
  - 70% Train
  - 15% Validation
  - 15% Test
- Split assignments are saved in `data/processed/splits.csv` and validated for zero session overlap.

---

## 6. Verified Local Dataset Statistics & Verification Evidence

As verified during Phase 3 EDA (`scripts/prepare_dataset.py`) and stored in `data/processed/stats.json`:

* **Total Images Processed**: 2,003 field photographs (`.jpg`) in `data/raw/cropandweed/images/`.
* **Image Integrity**: 100% valid (0 corrupt, 0 unreadable files).
* **Native Resolution**: Uniform 1920 × 1088 px.
* **Recording Sessions**: 194 distinct sessions.
* **Total Plant Annotations**: 19,785 instances.
  * **Crop Instances**: 5,128 annotations (25.9%)
  * **Weed Instances**: 14,657 annotations (74.1% weed prevalence)
* **Instance Bounding Box Size Distribution**:
  * Tiny (< 32 × 32 px): 4,314 boxes (21.8%)
  * Small (32 – 96 px): 9,150 boxes (46.2%)
  * Medium (96 – 256 px): 4,702 boxes (23.8%)
  * Large (> 256 px): 1,619 boxes (8.2%)
  * *Over 68% of plants are small or tiny, proving the necessity of Feature Pyramid Networks (FPN).*
* **Session-Grouped Splits (`data/processed/splits.csv`)**:
  * **Train Split**: 1,381 images across 135 sessions (13,678 total plants: 3,556 crops, 10,122 weeds).
  * **Validation Split**: 338 images across 29 sessions (3,315 total plants: 853 crops, 2,462 weeds).
  * **Test Split**: 284 images across 30 sessions (2,792 total plants: 719 crops, 2,073 weeds).
  * **Session Leakage**: **Strictly 0.0% overlap** across all three sets (verified by `tests/test_splits.py`).
* **Visual Artifacts**:
  * `outputs/plots/samples.png` (12-sample visualizer)
  * `outputs/plots/class_distribution.png` (Crop vs Weed counts)
  * `outputs/plots/box_size_hist.png` (Instance scale histogram)

