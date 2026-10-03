"""Object detection evaluation metrics for Weed vs Crop discrimination.

Implements COCO mAP (mAP@0.5, mAP@[0.5:0.95], per-class AP), IoU greedy matching,
precision/recall/F1 at specific IoU and confidence thresholds, and detection confusion matrix.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

logger = logging.getLogger(__name__)


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes [x1, y1, x2, y2]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_detection_confusion_matrix(
    all_preds: List[Dict[str, np.ndarray]],
    all_gts: List[Dict[str, np.ndarray]],
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Compute 3x3 detection confusion matrix:

    Classes: 0: Crop (label 1), 1: Weed (label 2), 2: Background (Missed / False Alarm)
    Rows = Ground Truth, Columns = Prediction.
    """
    # 3x3 matrix: [Crop, Weed, Background]
    cm = np.zeros((3, 3), dtype=np.int64)

    for preds, gts in zip(all_preds, all_gts):
        gt_boxes = gts["boxes"]
        gt_labels = gts["labels"]  # 1 or 2

        pred_boxes = preds["boxes"]
        pred_labels = preds["labels"]  # 1 or 2
        pred_scores = preds["scores"]

        # Filter predictions by score threshold
        mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]
        pred_scores = pred_scores[mask]

        matched_gt = set()

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)
            # Sort preds by score descending
            order = np.argsort(-pred_scores)

            for p_idx in order:
                p_label = pred_labels[p_idx]  # 1 or 2
                best_gt_idx = -1
                best_iou = iou_threshold

                for g_idx in range(len(gt_boxes)):
                    if g_idx in matched_gt:
                        continue
                    if ious[p_idx, g_idx] >= best_iou:
                        best_iou = ious[p_idx, g_idx]
                        best_gt_idx = g_idx

                if best_gt_idx >= 0:
                    matched_gt.add(best_gt_idx)
                    g_label = gt_labels[best_gt_idx]
                    # Map 1 -> index 0, 2 -> index 1
                    cm[g_label - 1, p_label - 1] += 1
                else:
                    # False alarm: Ground Truth is Background (index 2)
                    cm[2, p_label - 1] += 1
        elif len(pred_boxes) > 0:
            # All predictions are false alarms
            for p_label in pred_labels:
                cm[2, p_label - 1] += 1

        # Any unmatched GT are missed (Prediction is Background, index 2)
        for g_idx in range(len(gt_boxes)):
            if g_idx not in matched_gt:
                g_label = gt_labels[g_idx]
                cm[g_label - 1, 2] += 1

    return cm


def compute_prf1_at_iou(
    all_preds: List[Dict[str, np.ndarray]],
    all_gts: List[Dict[str, np.ndarray]],
    target_class: int = 2,  # 2 = Weed
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute Precision, Recall, F1 for target class at given IoU and score threshold."""
    cm = compute_detection_confusion_matrix(all_preds, all_gts, score_threshold, iou_threshold)
    idx = target_class - 1  # 0 for crop, 1 for weed

    tp = cm[idx, idx]
    fp = cm[2, idx] + cm[1 - idx, idx]  # Predicted target class when true was background or opposite class
    fn = cm[idx, 2] + cm[idx, 1 - idx]  # True target class missed or predicted as opposite class

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return float(prec), float(rec), float(f1)


def sweep_thresholds(
    all_preds: List[Dict[str, np.ndarray]],
    all_gts: List[Dict[str, np.ndarray]],
    thresholds: Optional[List[float]] = None,
    iou_threshold: float = 0.5,
) -> List[Dict[str, float]]:
    """Sweep score thresholds from 0.1 to 0.9 to evaluate weed precision/recall trade-off."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    for thr in thresholds:
        prec, rec, f1 = compute_prf1_at_iou(all_preds, all_gts, target_class=2, score_threshold=thr, iou_threshold=iou_threshold)
        results.append({
            "threshold": round(thr, 2),
            "weed_precision": round(prec, 4),
            "weed_recall": round(rec, 4),
            "weed_f1": round(f1, 4),
        })
    return results


def evaluate_model_map(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model on a DataLoader using torchmetrics COCO MeanAveragePrecision.

    Returns:
        Dictionary of computed metrics (mAP, mAP_50, per_class_AP).
    """
    metric = MeanAveragePrecision(class_metrics=True)
    model.eval()

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            preds = model(images)

            cpu_preds = []
            for p in preds:
                cpu_preds.append({
                    "boxes": p["boxes"].detach().cpu(),
                    "scores": p["scores"].detach().cpu(),
                    "labels": p["labels"].detach().cpu(),
                })

            cpu_targets = []
            for t in targets:
                cpu_targets.append({
                    "boxes": t["boxes"].detach().cpu(),
                    "labels": t["labels"].detach().cpu(),
                })

            metric.update(cpu_preds, cpu_targets)

    computed = metric.compute()

    map_50 = float(computed["map_50"].item()) if "map_50" in computed else 0.0
    map_val = float(computed["map"].item()) if "map" in computed else 0.0

    # Class metrics (indices: 0 -> Crop [1], 1 -> Weed [2])
    map_per_class = computed.get("map_per_class", None)
    crop_ap = float(map_per_class[0].item()) if map_per_class is not None and len(map_per_class) > 0 else 0.0
    weed_ap = float(map_per_class[1].item()) if map_per_class is not None and len(map_per_class) > 1 else 0.0

    return {
        "mAP": round(map_val, 4),
        "mAP_50": round(map_50, 4),
        "crop_AP": round(crop_ap, 4),
        "weed_AP": round(weed_ap, 4),
    }
