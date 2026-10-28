"""Canonical Stage A data split for B1, with leakage assertions (spec §2.2)."""
from dataclasses import dataclass
from typing import Iterable, Tuple

import pandas as pd

from src.ml.data import load_dataset, split_dataset
from src.ml.preprocessing import clean_text

import config


class LeakageError(RuntimeError):
    """Raised when a B1 leakage rule would be violated."""


@dataclass(frozen=True)
class CanonicalSplit:
    full_df: pd.DataFrame
    train_df: pd.DataFrame  # sorted by original row id; this is the KB
    test_df: pd.DataFrame   # sorted by original row id; never indexed

    @property
    def train_ids(self) -> Tuple[int, ...]:
        return tuple(int(i) for i in self.train_df.index)

    @property
    def test_ids(self) -> Tuple[int, ...]:
        return tuple(int(i) for i in self.test_df.index)

    def intent_of(self, row_id: int) -> str:
        return str(self.train_df.at[row_id, "Intent"])

    def eligible_intents(self) -> Tuple[str, ...]:
        """Intents with >= ELIGIBLE_MIN_TRAIN_EXAMPLES training rows (spec §3.1)."""
        counts = self.train_df["Intent"].value_counts()
        return tuple(sorted(counts[counts >= config.ELIGIBLE_MIN_TRAIN_EXAMPLES].index))

    def all_intents(self) -> Tuple[str, ...]:
        return tuple(sorted(self.full_df["Intent"].astype(str).unique()))


def load_canonical_split() -> CanonicalSplit:
    """Load the Stage A dataset and canonical 160/40 split (seed 42, unstratified)."""
    full_df = load_dataset()
    train_df, test_df = split_dataset(full_df)
    split = CanonicalSplit(
        full_df=full_df,
        train_df=train_df.sort_index(),
        test_df=test_df.sort_index(),
    )
    _check_split(split)
    return split


def _check_split(split: CanonicalSplit) -> None:
    train, test = set(split.train_ids), set(split.test_ids)
    if len(train) != 160 or len(test) != 40:
        raise LeakageError(f"Expected 160/40 split, got {len(train)}/{len(test)}")
    if train & test:
        raise LeakageError("Train and test row ids overlap")
    if train | test != set(int(i) for i in split.full_df.index):
        raise LeakageError("Train ∪ test does not cover the dataset")


def assert_kb_is_train_only(row_ids: Iterable[int], split: CanonicalSplit) -> None:
    """The searchable KB must be exactly the 160 training rows (spec §2.2.1)."""
    ids = [int(r) for r in row_ids]
    if set(ids) & set(split.test_ids):
        raise LeakageError("KB index contains Stage A test row ids")
    if sorted(ids) != list(split.train_ids):
        raise LeakageError("KB index is not exactly the 160 training row ids")


def cleaned_questions(df: pd.DataFrame) -> Tuple[str, ...]:
    return tuple(clean_text(q) for q in df["User Message"])
