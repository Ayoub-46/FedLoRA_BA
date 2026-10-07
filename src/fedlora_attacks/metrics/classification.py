"""Image-classification outcome metrics: top-1, macro-F1, and a confidence trace.

WHY THIS TESTBED REPORTS THREE NUMBERS AND NOT ONE.

``top1_accuracy`` is the headline and the reason EuroSAT is here at all: it degrades
CONTINUOUSLY toward a known random floor of ``1/num_classes``, so an untargeted attack has
a graded damage curve to move along. Spider's execution accuracy does not — generated SQL
either parses and runs or it does not, so damage there jumps from "works" to "emits nothing
parseable" with little in between.

``macro_f1`` is the honest secondary because EuroSAT's classes are IMBALANCED (2000 to 3000
images each). A coalition that destroyed one small class while leaving the rest intact would
cost top-1 about two points and macro-F1 about ten; reporting only the first would call that
attack weak.

``mean_max_softmax`` is a cheap confidence trace, and it is a DIAGNOSTIC rather than a
quality score — it is registered in ``HIGHER_IS_BETTER`` for the same reason
``sql_parse_rate`` is: so a comparison looks the sign up instead of re-deriving it. It will
matter once attacks are added, where "still confident, now wrong" and "uniformly unsure"
are different failures that top-1 alone cannot tell apart.

``random_floor`` rides along so a reader never has to know the class count to place a
number. RELATIVE DAMAGE for this task is measured against the ZERO-SHOT value, not the
floor::

    (U_benign - U_attacked) / (U_benign - U_zeroshot)

with the floor as the separate sanity check that an attack destroyed the model outright
rather than partially. Those are different questions and the two references are not
interchangeable: CLIP starts at ~0.45, four and a half times the floor.
"""

from __future__ import annotations

import torch


def classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> dict[str, float]:
    """Top-1, macro-F1, mean max-softmax and the random floor, from raw logits.

    Computed in fp32 regardless of the dtype the forward ran in: a bf16 softmax over ten
    classes is fine for an argmax and not fine for a mean confidence quoted to three
    decimals. Macro-F1 is computed here rather than via sklearn — it is fifteen lines, and
    the project's pinned stack does not carry scikit-learn.
    """
    if int(num_classes) < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    logits = torch.as_tensor(logits).detach().float().cpu()
    labels = torch.as_tensor(labels).detach().long().cpu()
    if logits.ndim != 2:
        raise ValueError(f"logits must be (N, C), got {tuple(logits.shape)}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(f"{logits.shape[0]} logit rows vs {labels.shape[0]} labels")
    if logits.shape[1] != int(num_classes):
        raise ValueError(f"logits have {logits.shape[1]} columns, expected {num_classes}")
    if logits.shape[0] == 0:
        raise ValueError("classification_metrics needs at least one example")

    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    correct = preds == labels

    # Macro-F1: the UNWEIGHTED mean over classes, so a rare class counts as much as a
    # common one. A class with no predictions and no truths contributes 0 — it cannot
    # contribute 1, which would let a model that never emits a class score perfectly on it.
    f1s = []
    for cls in range(int(num_classes)):
        pred_c = preds == cls
        true_c = labels == cls
        tp = float((pred_c & true_c).sum())
        denom = float(pred_c.sum()) + float(true_c.sum())
        f1s.append(0.0 if denom == 0.0 else 2.0 * tp / denom)

    return {
        "top1_accuracy": float(correct.float().mean()),
        "macro_f1": float(sum(f1s) / len(f1s)),
        "mean_max_softmax": float(probs.max(dim=-1).values.mean()),
        "random_floor": 1.0 / float(num_classes),
        "eval_examples": float(logits.shape[0]),
    }


@torch.no_grad()
def evaluate_images(
    model,
    images,
    labels,
    num_classes: int,
    device="cpu",
    batch_size: int = 128,
    limit: int | None = None,
) -> dict[str, float]:
    """Score a model over a held-out image set. THE shared entry point.

    Called by BOTH ``central.baseline.image_metrics`` and ``fl.server``, deliberately, for
    the same reason the text testbeds share ``response_perplexity``: the centralized and
    federated runs exist to be compared, so ``top1_accuracy`` must mean bit-for-bit the
    same thing in both. Two implementations would drift in exactly the way that is hardest
    to notice — a batching or averaging difference of a few tenths of a point.

    Logits are accumulated in fp32 and the metrics computed ONCE over the whole set rather
    than averaged per batch. For top-1 those agree only when every batch is full; for
    macro-F1 they never agree, because an F1 is not the mean of per-batch F1s.
    """
    n = len(images) if limit is None else min(int(limit), len(images))
    if n < 1:
        raise ValueError("evaluate_images needs at least one held-out image")
    model.model.eval()
    chunks = []
    for start in range(0, n, batch_size):
        rows = list(range(start, min(start + batch_size, n)))
        chunks.append(model.image_logits(images.take(rows), device).detach().float().cpu())
    logits = torch.cat(chunks)
    target = torch.tensor([int(x) for x in labels[:n]], dtype=torch.long)
    out = dict(classification_metrics(logits, target, num_classes))
    # Same key as the instruction testbeds' mean NLL: it plays the same role (the held-out
    # training signal), so a reader comparing summaries never special-cases the name.
    out["eval_loss"] = cross_entropy(logits, target)
    return out


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean cross-entropy over the held-out set — this task's training signal.

    Plays the role ``response_perplexity`` plays for the instruction testbeds, and replaces
    it: perplexity is undefined here because there is no autoregressive head to exponentiate
    a per-token loss from. Reported under the same ``eval_loss`` key so a reader comparing
    run summaries across testbeds never has to special-case the name.
    """
    logits = torch.as_tensor(logits).detach().float().cpu()
    labels = torch.as_tensor(labels).detach().long().cpu()
    return float(torch.nn.functional.cross_entropy(logits, labels))
