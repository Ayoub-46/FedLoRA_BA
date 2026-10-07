"""AG News loader — the TRANSFER testbed (4 balanced classes, topic classification).

SST-2 is the microscope the results were measured through; AG News exists to ask which of
them are properties of the LoRA parametrisation and which were properties of that
microscope. Three differences are deliberate rather than incidental:

* **4 classes, not 2.** The majority-class floor drops from ~0.55 to ~0.25, so "the model
  still trains under attack" becomes a much weaker statement and there is far more room
  for an availability attack to work in. Any degradation ceiling observed on SST-2 that
  was really the binary floor will move here.
* **Balanced classes.** SST-2's train split is ~56/44; AG News is exactly 25% each, so a
  Dirichlet(alpha) partition produces label skew from the Dirichlet alone rather than
  from skew already present in the pool.
* **A wider classifier head.** The head is aggregated separately from the adapters (it
  sits outside the gauge bundle), so a 4-way head changes the ratio of gauge-affected to
  gauge-free parameters — the one structural quantity in the setup that the class count
  actually moves.

Unlike GLUE, AG News ships a labelled test split, so the held-out set is the real one and
no validation-as-test substitution is needed.
"""

from __future__ import annotations


def load_ag_news(num_train: int | None, num_test: int | None, seed: int):
    """Return (train_texts, train_labels, test_texts, test_labels).

    ``seed`` shuffles the train pool before subsampling, exactly as the SST-2 loader does,
    so a fixed ``subsample_seed`` gives every cell in a sweep the same examples.
    """
    from datasets import load_dataset

    ds = load_dataset("ag_news")
    train = ds["train"].shuffle(seed=seed)
    test = ds["test"].shuffle(seed=seed)
    if num_train is not None:
        train = train.select(range(min(num_train, len(train))))
    if num_test is not None:
        test = test.select(range(min(num_test, len(test))))
    return (
        list(train["text"]),
        list(train["label"]),
        list(test["text"]),
        list(test["label"]),
    )
