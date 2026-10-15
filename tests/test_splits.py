"""Unit and integrity tests for dataset session-grouped splits."""

from pathlib import Path
import unittest
import pandas as pd
from src.data.splits import load_splits, verify_splits_integrity


class TestSplits(unittest.TestCase):
    """Unit and integrity tests for dataset session-grouped splits."""

    def test_splits_zero_leakage(self):
        """Verify that splits.csv has zero session overlap between any splits."""
        splits_file = Path("data/processed/splits.csv")
        if not splits_file.exists():
            self.skipTest("data/processed/splits.csv has not been generated yet.")

        df = load_splits(splits_file)

        train_sessions = set(df[df["split"] == "train"]["session"])
        val_sessions = set(df[df["split"] == "val"]["session"])
        test_sessions = set(df[df["split"] == "test"]["session"])

        # Strict zero intersection assertion
        self.assertEqual(len(train_sessions.intersection(val_sessions)), 0, "Leakage between train and val!")
        self.assertEqual(len(train_sessions.intersection(test_sessions)), 0, "Leakage between train and test!")
        self.assertEqual(len(val_sessions.intersection(test_sessions)), 0, "Leakage between val and test!")

        # Verify non-empty splits
        self.assertGreater(len(train_sessions), 0)
        self.assertGreater(len(val_sessions), 0)
        self.assertGreater(len(test_sessions), 0)

    def test_verify_splits_integrity_catches_leakage(self):
        """Verify that verify_splits_integrity raises AssertionError if leakage exists."""
        leaky_df = pd.DataFrame([
            {"image_id": "img1", "session": "session_A", "split": "train"},
            {"image_id": "img2", "session": "session_A", "split": "test"},
        ])
        with self.assertRaises(AssertionError):
            verify_splits_integrity(leaky_df)


if __name__ == "__main__":
    unittest.main()
