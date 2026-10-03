"""Unit and integrity tests for dataset session-grouped splits."""

from pathlib import Path
import pytest
import pandas as pd
from src.data.splits import load_splits, verify_splits_integrity


def test_splits_zero_leakage():
    """Verify that splits.csv has zero session overlap between any splits."""
    splits_file = Path("data/processed/splits.csv")
    if not splits_file.exists():
        pytest.skip("data/processed/splits.csv has not been generated yet.")

    df = load_splits(splits_file)

    train_sessions = set(df[df["split"] == "train"]["session"])
    val_sessions = set(df[df["split"] == "val"]["session"])
    test_sessions = set(df[df["split"] == "test"]["session"])

    # Strict zero intersection assertion
    assert len(train_sessions.intersection(val_sessions)) == 0, "Leakage between train and val!"
    assert len(train_sessions.intersection(test_sessions)) == 0, "Leakage between train and test!"
    assert len(val_sessions.intersection(test_sessions)) == 0, "Leakage between val and test!"

    # Verify non-empty splits
    assert len(train_sessions) > 0
    assert len(val_sessions) > 0
    assert len(test_sessions) > 0


def test_verify_splits_integrity_catches_leakage():
    """Verify that verify_splits_integrity raises AssertionError if leakage exists."""
    leaky_df = pd.DataFrame([
        {"image_id": "img1", "session": "session_A", "split": "train"},
        {"image_id": "img2", "session": "session_A", "split": "test"},
    ])
    with pytest.raises(AssertionError):
        verify_splits_integrity(leaky_df)
