"""
Dataset loading and deterministic splitting for the University Support Chatbot.

Provides centralized dataset loading, validation, and train/test splitting
adhering to Phase 5A reproducibility requirements.
"""
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    SPLIT_RANDOM_STATE,
    TEST_SIZE,
)


def load_dataset(path: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """
    Load and validate the historical question-intent-response dataset from an Excel file.

    Parameters
    ----------
    path : Path or str, optional
        Path to the Excel dataset. Defaults to config.RAW_DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with validated columns and non-null required fields.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the specified path.
    ValueError
        If any REQUIRED_COLUMNS are missing from the dataset.
    ValueError
        If any required columns contain null or missing values.
    """
    file_path = Path(path) if path is not None else RAW_DATA_PATH

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    try:
        df = pd.read_excel(file_path)
    except Exception as exc:
        raise ValueError(f"Failed to read dataset from {file_path}: {exc}") from exc

    # Validate that all required columns are present
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset at {file_path} is missing required column(s): {missing_columns}. "
            f"Expected all of: {list(REQUIRED_COLUMNS)}"
        )

    # Validate that there are no null values in required fields
    null_counts = {
        col: int(df[col].isnull().sum())
        for col in REQUIRED_COLUMNS
        if df[col].isnull().sum() > 0
    }
    if null_counts:
        raise ValueError(
            f"Dataset at {file_path} contains null values in required field(s): {null_counts}"
        )

    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = SPLIT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministically split the dataset into training and test DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split.
    test_size : float, optional
        Fraction of dataset to allocate to the test split (default: TEST_SIZE = 0.20).
    random_state : int, optional
        Random seed for the split (default: SPLIT_RANDOM_STATE = 42).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) preserving original DataFrame indices.

    Notes
    -----
    Stratification is intentionally NOT used.
    The dataset contains multiple intents with extremely small support, including
    singleton classes (classes with only 1 example). Stratified sampling on such
    classes fails in scikit-learn (e.g., ValueError: The least populated class in
    y has only 1 member, which is too few) or distorts evaluation representation.
    Unstratified splitting with fixed random_state=42 guarantees reproducibility
    while preserving the exact canonical 160/40 train/test split.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_df, test_df
