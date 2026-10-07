"""SST-2 (GLUE) loader for real runs (plan §3 default text-classification task).

Requires network on first use to download the dataset. The hidden GLUE test split
is unlabeled, so we evaluate on the validation split as the held-out test set — the
standard practice for SST-2 experiments.
"""

from __future__ import annotations


def load_sst2(num_train: int | None, num_test: int | None, seed: int):
    """Return (train_texts, train_labels, test_texts, test_labels)."""
    from datasets import load_dataset

    ds = load_dataset("glue", "sst2")
    train = ds["train"].shuffle(seed=seed)
    val = ds["validation"]
    if num_train is not None:
        train = train.select(range(min(num_train, len(train))))
    if num_test is not None:
        val = val.select(range(min(num_test, len(val))))
    return (
        list(train["sentence"]),
        list(train["label"]),
        list(val["sentence"]),
        list(val["label"]),
    )
