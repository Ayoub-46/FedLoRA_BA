"""Synthetic, learnable text-classification dataset (hermetic CI default).

Each label owns a set of high-signal "marker" words; a sentence's label is
determined by which markers it contains, buried among neutral filler words. The
signal is strong enough that even a tiny model learns above-chance in a few epochs,
so main-task accuracy is meaningful in the smoke test — while requiring no download.
The vocabulary is exposed so a ``SimpleTokenizer`` can be built to match.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.seeding import numpy_rng


@dataclass
class SyntheticSpec:
    num_labels: int = 2
    num_fillers: int = 12  # fewer neutral words -> markers dominate -> stronger signal
    markers_per_label: int = 4
    min_len: int = 6
    max_len: int = 12
    markers_per_sentence: tuple[int, int] = (3, 5)


def build_vocab(spec: SyntheticSpec) -> tuple[list[str], dict[int, list[str]]]:
    fillers = [f"w{i}" for i in range(spec.num_fillers)]
    markers = {
        c: [f"c{c}_m{j}" for j in range(spec.markers_per_label)] for c in range(spec.num_labels)
    }
    vocab = fillers + [w for ws in markers.values() for w in ws]
    return vocab, markers


def _generate(n, spec, fillers, markers, rng):
    texts, labels = [], []
    for _ in range(n):
        c = int(rng.integers(spec.num_labels))
        length = int(rng.integers(spec.min_len, spec.max_len + 1))
        words = list(rng.choice(fillers, size=length))
        k = int(rng.integers(spec.markers_per_sentence[0], spec.markers_per_sentence[1] + 1))
        for _ in range(k):
            pos = int(rng.integers(0, length))
            words[pos] = markers[c][int(rng.integers(len(markers[c])))]
        texts.append(" ".join(words))
        labels.append(c)
    return texts, labels


def make_synthetic(
    num_train: int,
    num_test: int,
    seed: int,
    spec: SyntheticSpec | None = None,
):
    """Return (train_texts, train_labels, test_texts, test_labels, vocab, spec)."""
    spec = spec or SyntheticSpec()
    fillers = [f"w{i}" for i in range(spec.num_fillers)]
    vocab, markers = build_vocab(spec)
    rng = numpy_rng(seed)
    tr_x, tr_y = _generate(num_train, spec, fillers, markers, rng)
    te_x, te_y = _generate(num_test, spec, fillers, markers, rng)
    return tr_x, tr_y, te_x, te_y, vocab, spec


# -- hermetic instruction-tuning variant ---------------------------------------------


def make_synthetic_instruct(
    num_train: int, num_test: int, seed: int, spec: SyntheticSpec | None = None
):
    """A learnable instruction dataset: the smoke-test analogue of Dolly.

    Same construction as the classification set, re-shaped into (prompt, response,
    category): the prompt carries a category's marker words and the response is that
    category's answer phrase, so the mapping is learnable in a handful of steps and a
    smoke run's perplexity actually falls. That matters — a smoke test whose metric never
    moves cannot tell a working pipeline from a broken one.

    Exists so the WHOLE causal-LM path (loader -> partition -> client loss -> perplexity
    -> generation metrics -> server) is exercised offline, with no download and no 3B
    model. The plumbing bugs it catches would otherwise surface on the cluster, after a
    queue wait.
    """
    spec = spec or SyntheticSpec()
    rng = numpy_rng(seed)
    fillers, markers = build_vocab(spec)[0][: spec.num_fillers], build_vocab(spec)[1]
    answers = {c: f"answer c{c}" for c in range(spec.num_labels)}
    vocab = list(build_vocab(spec)[0]) + ["answer"] + [f"c{c}" for c in range(spec.num_labels)]
    vocab += ["describe", "the", "topic", "of"]

    def gen(n):
        prompts, responses, cats = [], [], []
        for _ in range(n):
            c = int(rng.integers(spec.num_labels))
            length = int(rng.integers(spec.min_len, spec.max_len + 1))
            words = list(rng.choice(fillers, size=length))
            k = int(rng.integers(spec.markers_per_sentence[0], spec.markers_per_sentence[1] + 1))
            for _ in range(k):
                words[int(rng.integers(0, length))] = markers[c][int(rng.integers(len(markers[c])))]
            prompts.append("describe the topic of " + " ".join(words))
            responses.append(answers[c])
            cats.append(c)
        return prompts, responses, cats

    tr = gen(num_train)
    te = gen(num_test)
    return tr, te, vocab
