"""Session-grouped dataset splitting to prevent visual and spatial data leakage."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.data.parse_annotations import parse_image_session

logger = logging.getLogger(__name__)


def create_session_grouped_splits(
    image_names: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Create session-grouped splits ensuring zero session overlap across splits.

    Args:
        image_names: List of image filenames (e.g. 'ave-0020-0010.jpg').
        train_ratio: Fraction of sessions allocated to training (default 0.70).
        val_ratio: Fraction allocated to validation (default 0.15).
        test_ratio: Fraction allocated to test (default 0.15).
        seed: Random seed for group splitting.
        output_path: Optional path to save splits.csv.

    Returns:
        DataFrame with columns ['image_id', 'session', 'split'].
    """
    if not image_names:
        raise ValueError("Cannot create splits from an empty image list.")

    records = []
    for img_name in sorted(image_names):
        stem = Path(img_name).stem
        prefix, session = parse_image_session(img_name)
        records.append({"image_id": stem, "filename": img_name, "session": session})

    df = pd.DataFrame(records)
    unique_sessions = df["session"].unique()

    # Step 1: Split into Train vs (Val + Test)
    val_test_ratio = val_ratio + test_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
    train_idx, val_test_idx = next(gss1.split(df, groups=df["session"]))

    train_df = df.iloc[train_idx].copy()
    val_test_df = df.iloc[val_test_idx].copy()

    # Step 2: Split (Val + Test) into Val and Test
    relative_test_ratio = test_ratio / val_test_ratio
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_ratio, random_state=seed)
    val_rel_idx, test_rel_idx = next(gss2.split(val_test_df, groups=val_test_df["session"]))

    val_df = val_test_df.iloc[val_rel_idx].copy()
    test_df = val_test_df.iloc[test_rel_idx].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    splits_df = combined_df[["image_id", "session", "split"]].sort_values("image_id").reset_index(drop=True)

    # Verify split integrity immediately
    verify_splits_integrity(splits_df)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        splits_df.to_csv(out_p, index=False)
        logger.info(f"Saved session-grouped splits to {out_p}")

    return splits_df


def verify_splits_integrity(splits_df: pd.DataFrame) -> None:
    """Verify that train, validation, and test splits have strict ZERO session overlap.

    Args:
        splits_df: DataFrame with ['image_id', 'session', 'split'] columns.

    Raises:
        AssertionError: If any session is shared between any two splits.
    """
    train_sessions: Set[str] = set(splits_df[splits_df["split"] == "train"]["session"])
    val_sessions: Set[str] = set(splits_df[splits_df["split"] == "val"]["session"])
    test_sessions: Set[str] = set(splits_df[splits_df["split"] == "test"]["session"])

    overlap_train_val = train_sessions.intersection(val_sessions)
    overlap_train_test = train_sessions.intersection(test_sessions)
    overlap_val_test = val_sessions.intersection(test_sessions)

    if overlap_train_val:
        raise AssertionError(f"Data leakage detected! Sessions in both train and val: {overlap_train_val}")
    if overlap_train_test:
        raise AssertionError(f"Data leakage detected! Sessions in both train and test: {overlap_train_test}")
    if overlap_val_test:
        raise AssertionError(f"Data leakage detected! Sessions in both val and test: {overlap_val_test}")

    logger.info("Split integrity verified: STRICT ZERO SESSION OVERLAP between train, val, and test.")


def load_splits(splits_path: Path | str) -> pd.DataFrame:
    """Load splits from CSV and verify zero session overlap."""
    path = Path(splits_path)
    if not path.is_file():
        raise FileNotFoundError(f"Splits file not found: {path}")

    df = pd.read_csv(path)
    verify_splits_integrity(df)
    return df
