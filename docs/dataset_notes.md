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
- Split assignments will be saved in `data/processed/splits.csv` and validated for zero session overlap.
