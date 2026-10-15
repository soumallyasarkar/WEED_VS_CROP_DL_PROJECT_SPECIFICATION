# Field-Image Weed vs Crop Detection: Evaluation Report

## 1. Executive Summary
- **Evaluated Split**: Untouched Test Split (282 real field images)
- **Model**: Faster R-CNN MobileNetV3-Large FPN
- **mAP@[0.5:0.95]**: 0.3430
- **mAP@0.5**: 0.5486
- **Crop AP**: 0.5973
- **Weed AP**: 0.0888
- **Weed Precision**: 0.4328 | **Weed Recall**: 0.2538 | **Weed F1**: 0.3200

## 2. Detection Confusion Matrix (IoU ≥ 0.5, Score ≥ 0.5)
| True \ Pred | Crop | Weed | Background (Missed) |
|---|---|---|---|
| **Crop** | 479 | 18 | 53 |
| **Weed** | 97 | 554 | 1532 |
| **Background (False Alarm)** | 140 | 708 | - |

## 3. Confidence Threshold Sweep (Weed Class at IoU 0.5)
| Score Threshold | Weed Precision | Weed Recall | Weed F1 |
|---|---|---|---|
| 0.1 | 0.1665 | 0.4246 | 0.2392 |
| 0.2 | 0.2215 | 0.3834 | 0.2808 |
| 0.3 | 0.2775 | 0.3472 | 0.3084 |
| 0.4 | 0.3438 | 0.2996 | 0.3202 |
| 0.5 | 0.4328 | 0.2538 | 0.3200 |
| 0.6 | 0.5267 | 0.1988 | 0.2887 |
| 0.7 | 0.6183 | 0.1425 | 0.2316 |
| 0.8 | 0.6865 | 0.1003 | 0.1751 |
| 0.9 | 0.7832 | 0.0513 | 0.0963 |

## 4. Error Analysis
- **Small/Tiny Weeds**: Missed detections correlate strongly with instances under 32x32 pixels where features are subtle.
- **Crop-Weed Confusion**: Occurs primarily at early emergence when crop and weed seedling cotyledons exhibit near-identical leaf shapes.
- **Operating Point Trade-off**: Lowering threshold to 0.3-0.4 maximizes weed eradication recall, whereas raising to 0.6-0.7 minimizes herbicide crop injury risk.
