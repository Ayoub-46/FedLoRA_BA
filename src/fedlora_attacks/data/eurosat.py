"""EuroSAT land-cover classification: the vision testbed with a GRADED damage curve.

WHY THIS EXISTS ALONGSIDE DOLLY AND SPIDER. The two text testbeds fail in opposite
directions when an attack is added. Dolly gave ~1.7% of benign perplexity headroom and its
BLEU sweeps sat inside their own noise floor — too little dynamic range for damage to be
readable. Spider fixed the headroom but is DISCRETE: generated SQL either parses and
executes or it does not, so an untargeted attack tends to move execution accuracy from
"works" to "emits nothing parseable" with almost nothing in between, and the interesting
middle of the damage curve is never sampled.

Top-1 accuracy over 10 classes degrades CONTINUOUSLY toward a known random floor of 0.10,
and the benign gain is large (zero-shot ~45-50% -> fine-tuned ~96-97%). EuroSAT is also the
modality where the Byzantine-robust FL baselines (Krum, Trimmed-Mean, Bulyan, ALIE, IPM)
are conventionally benchmarked, so the numbers are comparable to that literature in a way
Spider's are not.

**THE MIRROR MATTERS.** ``timm/eurosat-rgb`` is used and the near-duplicates are refused.
EuroSAT ships NO official splits; this mirror derives train/validation/test from the
google-research ``remote_sensing_representations`` definitions, which is the split the
remote-sensing transfer literature reports on. Every other mirror on the Hub either carves
its own undocumented splits (``blanchon``, ``giswqs``), ships a single 27k ``train`` with
no held-out set (``jonathan-roberts1``), ships 13-band uint16 that CLIP's 3-channel
processor cannot consume (``giswqs/EuroSAT_MS``), or is a legacy script loader broken on
``datasets>=3.0`` (bare ``eurosat``). Silently swapping one in produces a plausible,
non-comparable number, so ``verify_splits`` checks the split sizes and the label vocabulary
before anything trains.

**THE HELD-OUT SET IS ``test``, AND ``validation`` IS FOR TUNING.** Both are 5400 images
and either can be loaded (``heldout_split``), but they are not interchangeable: anything
that SELECTS a setting — learning rate, rank, the per-round budget — must be read on
``validation``, or the reported number is a number the setting was chosen to maximise.
Which split a run used is recorded in ``TaskData.info`` and reaches ``summary.json``, so a
tuning number can never be mistaken for a reportable one.

**PREPROCESSING IS THE DOMINANT CPU COST, so it is paid ONCE.** The images are natively
64x64 and ``CLIPImageProcessor`` bicubic-upsamples them to 224x224 (that is what every
published CLIP-on-EuroSAT number does; it is not a defect). Re-running that resize every
round is the difference between a 40-second evaluation and a 6-minute one. So the resized,
normalised ``pixel_values`` are written ONCE to a memory-mapped float16 array under
``$REPO/.eurosat_cache/``, keyed by the dataset revision, the split, the processor and the
image size — see ``cache_key``. 27000 x 3 x 224 x 224 x 2 bytes is ~8.1 GB for all three
splits, ~6.5 GB for the two this module loads.

float16 for the CACHE is not a violation of invariant 6. That invariant governs the LoRA
factors and the probes that read them, which stay fp32/fp64; this is input data, fed to a
forward pass that runs in bf16 anyway — an 8-bit mantissa downstream of an 11-bit one.

**THE "LABEL" IS THE LAND-COVER CLASS**, and the ``TaskData`` contract is reused unchanged:
``train_labels`` holds the integer class and ``label_names[label]`` recovers the stored
CamelCase name. That drops straight into ``dirichlet_partition``.

PARTITION AXIS (a later federated arm; NOT wired here). Dirichlet over the 10 labels,
alpha in {0.1, 0.5, 100}. With only 10 classes and 20-50 clients, alpha=0.1 hands many
clients 1-2 classes — severe, but standard for this benchmark. Unlike Spider's 166 db_ids
there is no second natural axis, so a run must report the REALIZED per-client class
histograms rather than just quoting alpha.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The mirror, and the ONLY one. See the module docstring for the seven rejected
#: near-duplicates and what each of them would silently change.
EUROSAT_DATASET = "timm/eurosat-rgb"

#: Commit pin, UNSET until the first prefetch resolves it.
#:
#: Spider pins both of its halves by commit because a re-upload under a moving ref would
#: change the benchmark under a config hash that did not move. The same reasoning applies
#: here and the pin is the right end state — but a SHA must be READ from the Hub, never
#: guessed, so this stays ``None`` until ``scripts/slurm/prefetch_eurosat.sbatch`` prints
#: the resolved commit and a human writes it in.
#:
#: Until then the integrity guarantee is ``verify_splits``: exact split sizes and the exact
#: 10-name label vocabulary, in order. That catches every wrong mirror in the rejected list
#: and any re-derivation of the splits, which is what the pin is actually protecting.
EUROSAT_REVISION: str | None = "b4e28552cd5f3932b6abc37eb20d3e84901ad728"

#: Model AND processor: they must come from the same repo id, because the head is built
#: from that checkpoint's text tower and scored against that checkpoint's normalisation.
CLIP_MODEL = "openai/clip-vit-base-patch32"

#: Label index -> stored name, EXACTLY as ``timm/eurosat-rgb`` stores them (CamelCase, no
#: spaces). The order is authoritative: it is what ``features["label"].names`` returns, and
#: ``verify_splits`` asserts the loaded dataset agrees. CLIP's text tower scores these raw
#: names badly — the natural-language remapping used for the prompts lives in
#: ``models/clip_head.py``, deliberately separate, because the two vocabularies serve
#: different purposes and conflating them is how a several-point zero-shot drop gets in.
EUROSAT_CLASSES: tuple[str, ...] = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

#: The mirror's split sizes. Checked, not assumed: every rejected near-duplicate differs
#: here (18.9k/5.4k/2.7k, 16.2k/5.4k/5.4k-but-undocumented, a single 27k train, ...), so
#: this triple is the cheapest discriminator between the right dataset and a plausible
#: wrong one.
EUROSAT_SPLIT_SIZES: dict[str, int] = {"train": 16200, "validation": 5400, "test": 5400}

#: What CLIP ViT-B/32 consumes. Native EuroSAT tiles are 64x64.
IMAGE_SIZE = 224
NATIVE_IMAGE_SIZE = 64


class EuroSATError(ValueError):
    """Raised instead of returning a silently degraded dataset."""


# --- the pixel cache --------------------------------------------------------------------


# eq=False: the fields hold numpy arrays, so a generated __eq__ would compare them
# elementwise and raise on the truth value of the result. Identity is the honest semantic
# for a view over a memory map anyway.
@dataclass(frozen=True, eq=False)
class PixelView:
    """A row-indexed view on a memory-mapped ``(N, 3, H, W)`` float16 pixel cache.

    Two things are separated on purpose. The CACHE holds a whole split, keyed by the
    processor and the dataset revision, so it is shared by every run regardless of how much
    of the split that run uses. The VIEW holds the seeded subsample, which is a property of
    the run. Caching the subsample instead would make the 6.5 GB of preprocessing
    non-reusable the first time ``num_train`` changed.
    """

    array: np.ndarray  # the full-split memmap (N, 3, H, W) float16
    indices: np.ndarray  # int64 rows of `array`, in subsample order

    def __len__(self) -> int:
        return int(len(self.indices))

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self), *tuple(self.array.shape[1:]))

    def subset(self, rows: Sequence[int] | np.ndarray) -> PixelView:
        """A narrower view over the same cache — one federated client's shard.

        Composes indices rather than copying pixels: a 20-client partition of the 4.9 GB
        train cache costs 20 index vectors, not 20 slices of it.
        """
        rows = np.asarray(rows, dtype=np.int64)
        return PixelView(self.array, self.indices[rows])

    def take(self, rows: Sequence[int] | np.ndarray) -> np.ndarray:
        """Materialise the given VIEW positions as a real ``(B, 3, H, W)`` array.

        ``rows`` index the view, not the underlying cache — so a caller that drew batch
        indices from ``len(view)`` cannot accidentally address a row the subsample excluded.
        """
        rows = np.asarray(rows, dtype=np.int64)
        return np.asarray(self.array[self.indices[rows]])


def default_cache_root() -> Path:
    """Where preprocessed pixel values live. Override with ``EUROSAT_CACHE_DIR``."""
    return Path(os.environ.get("EUROSAT_CACHE_DIR", ".eurosat_cache"))


def cache_key(
    *,
    dataset: str = EUROSAT_DATASET,
    revision: str | None = EUROSAT_REVISION,
    split: str,
    processor: str = CLIP_MODEL,
    processor_revision: str | None = None,
    image_size: int = IMAGE_SIZE,
) -> str:
    """A stable digest of everything that changes the CACHED BYTES.

    The processor identity is in the key because the cache stores the OUTPUT of
    ``CLIPImageProcessor``, not the images: a different checkpoint's normalisation
    constants, a different resize, or a different crop size all produce different pixel
    values from the same JPEG. Without the processor in the key, swapping the model would
    silently reuse a cache built for the other one — and every number afterwards would be
    computed on inputs the model was never calibrated for.
    """
    payload = {
        "dataset": dataset,
        "revision": revision or "unpinned",
        "split": split,
        "processor": processor,
        "processor_revision": processor_revision or "unpinned",
        "image_size": int(image_size),
        "dtype": "float16",
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def cache_paths(root: str | Path, key: str, split: str) -> tuple[Path, Path]:
    """``(pixels.f16, pixels.meta.json)`` for one cached split."""
    base = Path(root) / f"{split}_{key}"
    return base.with_suffix(".f16"), base.with_suffix(".meta.json")


def build_pixel_cache(
    images: Callable[[int, int], Sequence],
    n: int,
    pixel_fn: Callable[[Sequence], np.ndarray],
    bin_path: str | Path,
    meta_path: str | Path,
    *,
    image_size: int = IMAGE_SIZE,
    chunk: int = 128,
    key: str = "",
) -> np.ndarray:
    """Preprocess ``n`` images into a memory-mapped float16 ``(n, 3, S, S)`` array.

    ``images(start, stop)`` returns that slice of raw images and ``pixel_fn`` turns a list
    of them into ``(B, 3, S, S)`` float32 — the two are injected rather than imported so the
    whole cache path unit-tests with no ``datasets``, no ``transformers`` and no download.

    Written to a ``.partial`` file and renamed only on success. A cache is either complete
    or absent; a half-written 6.5 GB array that reads as zeros for its tail would train
    perfectly happily on black images.
    """
    bin_path, meta_path = Path(bin_path), Path(meta_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    partial = bin_path.with_suffix(bin_path.suffix + ".partial")
    shape = (int(n), 3, int(image_size), int(image_size))
    out = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=shape)
    try:
        for start in range(0, int(n), int(chunk)):
            stop = min(start + int(chunk), int(n))
            batch = np.asarray(pixel_fn(images(start, stop)))
            if batch.shape != (stop - start, *shape[1:]):
                raise EuroSATError(
                    f"the image processor returned {batch.shape} for rows "
                    f"[{start}, {stop}); expected {(stop - start, *shape[1:])}. A different "
                    "crop size or channel count would make every cached run incomparable."
                )
            out[start:stop] = batch.astype(np.float16, copy=False)
        out.flush()
    finally:
        del out
    partial.replace(bin_path)
    meta_path.write_text(
        json.dumps(
            {
                "key": key,
                "shape": list(shape),
                "dtype": "float16",
                "image_size": int(image_size),
                "n": int(n),
            },
            indent=2,
        )
    )
    return np.load(bin_path, mmap_mode="r")


def load_pixel_cache(bin_path: str | Path, meta_path: str | Path, n: int) -> np.ndarray | None:
    """The cached array if it is present AND complete, else None.

    The row count is checked against the meta file, because "the file exists" is not the
    same claim as "the file holds this split". A cache built from a truncated first attempt
    would otherwise be reused forever.
    """
    bin_path, meta_path = Path(bin_path), Path(meta_path)
    if not bin_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if int(meta.get("n", -1)) != int(n):
        return None
    array = np.load(bin_path, mmap_mode="r")
    return array if array.shape[0] == int(n) else None


# --- splits -----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)  # holds PixelViews; see the note above
class EuroSATSplits:
    """Pixel views and labels for both splits, plus the class vocabulary."""

    train_pixels: PixelView
    train_labels: list[int]
    test_pixels: PixelView
    test_labels: list[int]
    class_names: tuple[str, ...]
    cache_key_train: str = ""
    cache_key_test: str = ""
    heldout_split: str = "test"
    """Which upstream split ``test_pixels`` actually came from — ``test`` or ``validation``.

    Carried rather than assumed because the two are the same size and the same shape, so
    nothing downstream could tell them apart. A tuning run and a reported run differ ONLY
    in this string.
    """

    def __post_init__(self) -> None:
        if len(self.train_pixels) != len(self.train_labels):
            raise EuroSATError(
                f"{len(self.train_pixels)} train images vs {len(self.train_labels)} labels; "
                "the subsample and the label list are out of step"
            )
        if len(self.test_pixels) != len(self.test_labels):
            raise EuroSATError(
                f"{len(self.test_pixels)} test images vs {len(self.test_labels)} labels"
            )


def subsample_order(n: int, limit: int | None, seed: int) -> np.ndarray:
    """A seeded permutation, optionally truncated. Same contract as ``spider.build_splits``.

    ``seed`` here is the SUBSAMPLE seed, not the run seed: it decides WHICH examples the
    subset contains, so every cell of a sweep trains on the identical subsample while the
    run seed still varies the init and the batch order (CLAUDE.md "Compute & hardware").
    """
    order = np.random.default_rng(seed).permutation(int(n))
    return order if limit is None else order[: int(limit)]


def build_splits(
    train_pixels: np.ndarray,
    train_labels: Sequence[int],
    test_pixels: np.ndarray,
    test_labels: Sequence[int],
    num_train: int | None,
    num_test: int | None,
    seed: int,
    class_names: Sequence[str] = EUROSAT_CLASSES,
    cache_key_train: str = "",
    cache_key_test: str = "",
    heldout_split: str = "test",
) -> EuroSATSplits:
    """Assemble both splits from full-split pixel caches. No HuggingFace import, so this
    unit-tests offline against plain numpy arrays.

    EuroSAT's splits are already disjoint upstream, so unlike Dolly nothing is carved here.
    """
    if len(train_labels) == 0 or len(test_labels) == 0:
        raise EuroSATError(
            f"EuroSAT did not load: {len(train_labels)} train, {len(test_labels)} test rows"
        )
    if len(train_pixels) != len(train_labels) or len(test_pixels) != len(test_labels):
        raise EuroSATError(
            f"pixel cache and labels disagree: train {len(train_pixels)}/{len(train_labels)}, "
            f"test {len(test_pixels)}/{len(test_labels)}. The cache was built from a "
            "different revision of the split than the labels were read from."
        )
    tr_idx = subsample_order(len(train_labels), num_train, seed)
    te_idx = subsample_order(len(test_labels), num_test, seed)
    return EuroSATSplits(
        train_pixels=PixelView(train_pixels, tr_idx),
        train_labels=[int(train_labels[int(i)]) for i in tr_idx],
        test_pixels=PixelView(test_pixels, te_idx),
        test_labels=[int(test_labels[int(i)]) for i in te_idx],
        class_names=tuple(class_names),
        cache_key_train=cache_key_train,
        cache_key_test=cache_key_test,
        heldout_split=heldout_split,
    )


def verify_splits(dataset_dict, *, strict: bool = True) -> tuple[str, ...]:
    """Assert the loaded dataset IS the pinned mirror, and return its label vocabulary.

    The whole reason ``timm/eurosat-rgb`` is named verbatim in the spec is that six other
    Hub datasets called some variation of "EuroSAT" would load, train, and report a
    plausible accuracy that is not comparable to anything. Two checks separate them: the
    exact split sizes (every rejected mirror differs) and the exact label vocabulary in
    order (a re-ordered vocabulary silently permutes the frozen zero-shot head, so every
    prediction maps to the wrong class while accuracy stays a number between 0 and 1).
    """
    sizes = {name: len(split) for name, split in dataset_dict.items()}
    wrong = {k: (sizes.get(k), v) for k, v in EUROSAT_SPLIT_SIZES.items() if sizes.get(k) != v}
    if wrong and strict:
        raise EuroSATError(
            f"{EUROSAT_DATASET} split sizes are {sizes}, expected {EUROSAT_SPLIT_SIZES}. "
            "This is not the google-research remote_sensing_representations split; the "
            "accuracy it produces would not be comparable to the published numbers."
        )
    names = tuple(dataset_dict["train"].features["label"].names)
    if names != EUROSAT_CLASSES and strict:
        raise EuroSATError(
            f"label vocabulary is {names}, expected {EUROSAT_CLASSES}. A permuted "
            "vocabulary silently mismatches the frozen zero-shot head against the labels."
        )
    return names


def clip_pixel_fn(processor) -> Callable[[Sequence], np.ndarray]:
    """Adapt a ``CLIPImageProcessor`` to the ``pixel_fn`` the cache builder wants.

    Returned as a closure rather than called inline so ``build_pixel_cache`` never imports
    ``transformers`` — the cache path is then testable with a two-line fake, and the
    torchvision-dependent image stack stays out of the hermetic suite.
    """

    def call(images: Sequence) -> np.ndarray:
        out = processor(images=list(images), return_tensors="np")
        return np.asarray(out["pixel_values"], dtype=np.float32)

    return call


def load_eurosat(
    num_train: int | None,
    num_test: int | None,
    seed: int,
    *,
    cache_root: str | Path | None = None,
    processor=None,
    image_size: int = IMAGE_SIZE,
    revision: str | None = EUROSAT_REVISION,
    heldout_split: str = "test",
) -> EuroSATSplits:
    """Download (once), preprocess (once), and assemble EuroSAT.

    ``seed`` is the SUBSAMPLE seed, not the run seed.

    ``heldout_split`` selects which upstream split fills the held-out slot. ``test`` is the
    reportable one; ``validation`` is for anything that SELECTS a setting. Only the chosen
    split is preprocessed, so a tuning run never pays for the other one's 1.6 GB cache.
    """
    if heldout_split not in ("test", "validation"):
        raise EuroSATError(
            f"heldout_split must be 'test' or 'validation', got {heldout_split!r}. "
            "'train' is not selectable: scoring on the training split is not an evaluation."
        )
    from datasets import load_dataset

    kwargs = {"revision": revision} if revision else {}
    ds = load_dataset(EUROSAT_DATASET, **kwargs)
    class_names = verify_splits(ds)

    if processor is None:
        from transformers import CLIPImageProcessor

        processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    pixel_fn = clip_pixel_fn(processor)
    root = Path(cache_root) if cache_root is not None else default_cache_root()

    caches: dict[str, tuple[np.ndarray, str]] = {}
    for split in ("train", heldout_split):
        rows = ds[split]
        key = cache_key(revision=revision, split=split, image_size=image_size)
        bin_path, meta_path = cache_paths(root, key, split)
        array = load_pixel_cache(bin_path, meta_path, len(rows))
        if array is None:
            array = build_pixel_cache(
                lambda a, b, rows=rows: rows[a:b]["image"],
                len(rows),
                pixel_fn,
                bin_path,
                meta_path,
                image_size=image_size,
                key=key,
            )
        caches[split] = (array, key)

    return build_splits(
        caches["train"][0],
        list(ds["train"]["label"]),
        caches[heldout_split][0],
        list(ds[heldout_split]["label"]),
        num_train,
        num_test,
        seed,
        class_names,
        cache_key_train=caches["train"][1],
        cache_key_test=caches[heldout_split][1],
        heldout_split=heldout_split,
    )
