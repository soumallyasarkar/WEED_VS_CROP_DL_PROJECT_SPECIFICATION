"""Object detection evaluation metrics for Weed vs Crop discrimination.

Implements COCO mAP (mAP@0.5, mAP@[0.5:0.95], per-class AP), IoU greedy matching,
precision/recall/F1 at specific IoU and confidence thresholds, and detection confusion matrix.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import importlib
    _tm = importlib.import_module("torchmetrics.detection")
    MeanAveragePrecision = getattr(_tm, "MeanAveragePrecision")
    HAS_TORCHMETRICS = True
except (ImportError, AttributeError):
    HAS_TORCHMETRICS = False
    logger.info("torchmetrics not detected; falling back to built-in NumPy Average Precision evaluation.")


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
                p_label = int(pred_labels[p_idx])
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
                    g_label = int(gt_labels[best_gt_idx])
                    if 1 <= g_label <= 2 and 1 <= p_label <= 2:
                        cm[g_label - 1, p_label - 1] += 1
                else:
                    # False alarm: Ground Truth is Background (index 2)
                    if 1 <= p_label <= 2:
                        cm[2, p_label - 1] += 1
        elif len(pred_boxes) > 0:
            # All predictions are false alarms
            for p_label in pred_labels:
                p_l = int(p_label)
                if 1 <= p_l <= 2:
                    cm[2, p_l - 1] += 1

        # Any unmatched GT are missed (Prediction is Background, index 2)
        for g_idx in range(len(gt_boxes)):
            if g_idx not in matched_gt:
                g_label = int(gt_labels[g_idx])
                if 1 <= g_label <= 2:
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


def compute_ap_for_class(
    all_preds: List[Dict[str, np.ndarray]],
    all_gts: List[Dict[str, np.ndarray]],
    class_id: int = 1,
    iou_threshold: float = 0.5,
) -> float:
    """Compute Average Precision (AP@50) for a specific class using 11-point interpolation."""
    pred_records: List[Tuple[float, np.ndarray, int]] = []
    total_gts = 0

    for img_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
        gt_boxes = gts["boxes"]
        gt_labels = gts["labels"]
        total_gts += int(np.sum(gt_labels == class_id))

        p_boxes = preds["boxes"]
        p_labels = preds["labels"]
        p_scores = preds["scores"]

        for box, label, score in zip(p_boxes, p_labels, p_scores):
            if int(label) == class_id:
                pred_records.append((float(score), box, img_idx))

    if total_gts == 0 or len(pred_records) == 0:
        return 0.0

    # Sort detections by confidence score descending
    pred_records.sort(key=lambda x: x[0], reverse=True)

    matched_gt: Dict[int, Set[int]] = {i: set() for i in range(len(all_gts))}
    tp = np.zeros(len(pred_records))
    fp = np.zeros(len(pred_records))

    for p_idx, (_, p_box, img_idx) in enumerate(pred_records):
        gts = all_gts[img_idx]
        mask = gts["labels"] == class_id
        c_gt_boxes = gts["boxes"][mask]

        if len(c_gt_boxes) == 0:
            fp[p_idx] = 1
            continue

        ious = box_iou(np.array([p_box]), c_gt_boxes)[0]
        best_gt_idx = -1
        best_iou = iou_threshold

        for g_idx in range(len(c_gt_boxes)):
            if g_idx in matched_gt[img_idx]:
                continue
            if ious[g_idx] >= best_iou:
                best_iou = ious[g_idx]
                best_gt_idx = g_idx

        if best_gt_idx >= 0:
            matched_gt[img_idx].add(best_gt_idx)
            tp[p_idx] = 1
        else:
            fp[p_idx] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(1, total_gts)
    precisions = cum_tp / (cum_tp + cum_fp)

    # 11-point interpolated AP (standard VOC style)
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        prec_at_rec = precisions[recalls >= t]
        p = np.max(prec_at_rec) if len(prec_at_rec) > 0 else 0.0
        ap += p / 11.0

    return float(ap)


def evaluate_model_map(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model on a DataLoader using MeanAveragePrecision (or native NumPy fallback).

    Returns:
        Dictionary of computed metrics (mAP, mAP_50, per_class_AP).
    """
    model.eval()

    if HAS_TORCHMETRICS:
        try:
            metric = MeanAveragePrecision(class_metrics=True)
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
            map_50 = float(computed["map_50"].item()) if "map_50" in computed and computed["map_50"].item() >= 0 else 0.0
            map_val = float(computed["map"].item()) if "map" in computed and computed["map"].item() >= 0 else map_50

            map_per_class = computed.get("map_per_class", None)
            classes_tensor = computed.get("classes", None)

            crop_ap = 0.0
            weed_ap = 0.0
            if map_per_class is not None:
                if classes_tensor is not None:
                    for idx, c_id in enumerate(classes_tensor.tolist()):
                        val = float(map_per_class[idx].item())
                        if val >= 0:
                            if c_id == 1:
                                crop_ap = val
                            elif c_id == 2:
                                weed_ap = val
                else:
                    if len(map_per_class) > 0 and map_per_class[0].item() >= 0:
                        crop_ap = float(map_per_class[0].item())
                    if len(map_per_class) > 1 and map_per_class[1].item() >= 0:
                        weed_ap = float(map_per_class[1].item())

            return {
                "mAP": round(map_val, 4),
                "mAP_50": round(map_50, 4),
                "crop_AP": round(crop_ap, 4),
                "weed_AP": round(weed_ap, 4),
            }
        except Exception as e:
            logger.warning(f"torchmetrics evaluation encountered error ({e}); falling back to native AP.")

    # Native NumPy fallback evaluation
    all_preds: List[Dict[str, np.ndarray]] = []
    all_gts: List[Dict[str, np.ndarray]] = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            preds = model(images)

            for p in preds:
                all_preds.append({
                    "boxes": p["boxes"].detach().cpu().numpy(),
                    "scores": p["scores"].detach().cpu().numpy(),
                    "labels": p["labels"].detach().cpu().numpy(),
                })
            for t in targets:
                all_gts.append({
                    "boxes": t["boxes"].detach().cpu().numpy(),
                    "labels": t["labels"].detach().cpu().numpy(),
                })

    crop_ap = compute_ap_for_class(all_preds, all_gts, class_id=1, iou_threshold=0.5)
    weed_ap = compute_ap_for_class(all_preds, all_gts, class_id=2, iou_threshold=0.5)
    map_50 = (crop_ap + weed_ap) / 2.0

    return {
        "mAP": round(map_50, 4),
        "mAP_50": round(map_50, 4),
        "crop_AP": round(crop_ap, 4),
        "weed_AP": round(weed_ap, 4),
    }
