"""Dataset assembly: standardized splits, tokenizer factory, and federated view.

Three task kinds share one container. For ``seq_cls`` a row is (text, label); for
``causal_lm`` it is (prompt, response, category) and the CATEGORY occupies the label
slot — which is the point, not a hack: the label is whatever the Dirichlet is drawn over
and whatever the skew index measures, so Dolly's eight categories slot into the role
SST-2's sentiment label plays and ``dirichlet_partition`` needs no change at all. For
``image_cls`` a row is (pixel values, class) and the class occupies that same slot, so
EuroSAT's ten land-cover classes drop into the Dirichlet unchanged too.

The image case is the one that does NOT fit the text fields, and it is not made to. The
pixel side lives in ``train_images``/``test_images`` and the text fields stay empty; PIL
objects are never stuffed into ``train_texts``, which would make the container lie about
what it holds and would silently feed images to a tokenizer. ``modality`` is the
discriminator, and it is DERIVED from ``task`` rather than stored beside it — a second
stored flag is a second thing that can disagree with the data.

Prompt and response stay in separate fields the whole way down. The causal-LM loss scores
response tokens only, so the boundary must survive to the tokenizer; joining them here
would make masking impossible and would train the model to emit the instruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ag_news import load_ag_news
from .banking77 import BANKING77_REPO, BANKING77_REVISION, load_banking77
from .dolly import load_dolly
from .eurosat import load_eurosat
from .partition import Partition, dirichlet_partition, disjoint_label_partition
from .spider import load_spider
from .sst2 import load_sst2
from .synthetic import SyntheticSpec, make_synthetic, make_synthetic_instruct
from .tokenization import SimpleTokenizer, build_hf_tokenizer

SEQ_CLS = "seq_cls"
CAUSAL_LM = "causal_lm"
IMAGE_CLS = "image_cls"
TASKS = (SEQ_CLS, CAUSAL_LM, IMAGE_CLS)

#: Task kinds whose input side is pixels rather than text.
_IMAGE_TASKS = (IMAGE_CLS,)


@dataclass
class TaskData:
    """One dataset, any of the three task kinds.

    ``train_texts`` holds the input side: the sentence for classification, the PROMPT for
    instruction tuning. ``train_labels`` holds the class for classification, the CATEGORY
    id for instruction tuning. ``train_responses`` is the generation target and is
    ``None`` for classification — its presence is what tells the client which loss to
    take, so it is the single discriminator rather than a flag that could disagree with
    the data.
    """

    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    num_labels: int
    vocab: list[str] | None = None  # present for the synthetic (hermetic) dataset
    train_responses: list[str] | None = None
    test_responses: list[str] | None = None
    task: str = SEQ_CLS
    label_names: tuple[str, ...] | None = None
    """Human-readable names for the integer labels (Dolly's categories, Spider's db_ids).

    Carried so ``partition.json``'s histograms stay interpretable without importing the
    loader that produced them. For Spider it is load-bearing rather than cosmetic:
    ``label_names[label]`` is the database an example must be executed against.
    """
    train_images: object | None = None
    """The training images as preprocessed pixel values; ``None`` for a text task.

    Typed loosely on purpose. What EuroSAT puts here is a ``data.eurosat.PixelView`` — a
    seeded subsample view over a memory-mapped float16 ``(N, 3, H, W)`` cache — and pinning
    the annotation to that class would make ``data/base.py`` import the vision stack (and
    with it torchvision) for every text run. The contract is ``len(x)`` and
    ``x.take(rows) -> (B, 3, H, W)``; anything satisfying it works.
    """
    test_images: object | None = None
    """The held-out images, same contract.

    Named ``test_images`` rather than ``eval_images`` to match the ``train_``/``test_``
    pairing every other field in this container already uses.
    """

    info: dict = field(default_factory=dict)
    """Loader-specific facts a run should REPORT, not act on.

    Spider puts ``schema_truncation_rate`` here — the fraction of examples whose schema did
    not fit the prompt budget. That number is the ceiling any execution accuracy was
    measured under, so it has to reach ``summary.json``; a generic bag keeps it out of the
    shared field list while still making it impossible to lose between the loader and the
    run summary.
    """

    @property
    def is_causal_lm(self) -> bool:
        return self.task == CAUSAL_LM

    @property
    def modality(self) -> str:
        """``"image"`` or ``"text"`` — the discriminator call sites dispatch on.

        DERIVED from ``task``, never stored. A stored modality field would be a second
        source of truth that can disagree with both the task and the data actually present,
        which is the failure mode ``train_responses`` is deliberately shaped to avoid.
        """
        return "image" if self.task in _IMAGE_TASKS else "text"

    @property
    def num_train(self) -> int:
        """Training examples, whichever modality holds them."""
        return len(self.train_images) if self.modality == "image" else len(self.train_texts)

    @property
    def num_test(self) -> int:
        return len(self.test_images) if self.modality == "image" else len(self.test_texts)


#: The classification case, under its original name. Kept so the seq-cls call sites and
#: any external reader do not have to change meaning along with the container.
TextClassificationData = TaskData


@dataclass
class FederatedData:
    data: TextClassificationData
    partition: Partition

    def client_data(self, client_id: int) -> tuple[list[str], list[int]]:
        idx = self.partition.client_indices[client_id]
        if self.data.modality == "image":
            # client_data is the TEXT accessor and returning the empty text lists would
            # hand a client a silently empty shard. Pixels come from client_images.
            raise NotImplementedError(
                "client_data returns TEXT shards; this dataset is image-modality. Use "
                "client_images(client_id), which returns (PixelView, labels)."
            )
        return (
            [self.data.train_texts[i] for i in idx],
            [self.data.train_labels[i] for i in idx],
        )

    def client_responses(self, client_id: int) -> list[str] | None:
        """This client's generation targets, aligned with ``client_data``; None for seq-cls.

        Indexed by the SAME partition indices, so prompt i and response i belong to the
        same example. A separate lookup would be a silent misalignment waiting to happen.
        """
        if self.data.train_responses is None:
            return None
        idx = self.partition.client_indices[client_id]
        return [self.data.train_responses[i] for i in idx]

    def client_images(self, client_id: int) -> tuple[object, list[int]]:
        """This client's image shard and its labels — the image analogue of ``client_data``.

        The pixels are a VIEW composed onto the shared cache, not a copy: partitioning the
        4.9 GB train cache across 20 clients costs 20 index vectors. Labels are taken with
        the SAME indices, in the same order, so image i and label i are the same example —
        the misalignment this ordering prevents would be invisible, since every image would
        still carry a valid label, just not its own.
        """
        if self.data.modality != "image":
            raise NotImplementedError(
                "client_images is the IMAGE accessor; this dataset is text-modality. "
                "Use client_data(client_id)."
            )
        idx = self.partition.client_indices[client_id]
        return (
            self.data.train_images.subset(idx),
            [int(self.data.train_labels[i]) for i in idx],
        )

    @property
    def num_clients(self) -> int:
        return self.partition.num_clients


def build_dataset(
    name: str,
    num_train: int | None,
    num_test: int | None,
    seed: int,
    num_labels: int = 2,
    subsample_seed: int | None = None,
    max_prompt_tokens: int | None = None,
    heldout_split: str = "test",
) -> TaskData:
    """Build the standardized splits.

    ``subsample_seed`` decides WHICH examples the subset contains; ``seed`` is the run
    seed. Keeping them separate is what makes a sweep comparable: across cells and
    across run seeds every cell must train on the IDENTICAL subsample (CLAUDE.md
    "Compute & hardware"), while the run seed still varies the partition, the init and
    the batch order. Defaults to ``seed`` for backward compatibility with single runs.

    ``max_prompt_tokens`` is consumed only by ``spider``, whose prompts carry a database
    schema that has to be budgeted BEFORE the tokenizer sees it (see ``data/spider.py``:
    the model's own right-truncation would eat the question instead). Pass the model
    spec's value; leaving it None falls back to spider.py's default and the two can then
    disagree, which shows up as a higher truncation rate rather than as an error.
    """
    subsample_seed = seed if subsample_seed is None else subsample_seed
    if name == "synthetic":
        spec = SyntheticSpec(num_labels=num_labels)
        tr_x, tr_y, te_x, te_y, vocab, _ = make_synthetic(
            num_train or 512, num_test or 128, subsample_seed, spec
        )
        return TextClassificationData(tr_x, tr_y, te_x, te_y, num_labels, vocab=vocab)
    if name == "synthetic_instruct":
        # Hermetic causal-LM smoke path: same shape as dolly15k, no download.
        spec = SyntheticSpec(num_labels=num_labels)
        (tr_p, tr_r, tr_c), (te_p, te_r, te_c), vocab = make_synthetic_instruct(
            num_train or 256, num_test or 64, subsample_seed, spec
        )
        return TaskData(
            train_texts=tr_p,
            train_labels=tr_c,
            test_texts=te_p,
            test_labels=te_c,
            num_labels=num_labels,
            vocab=vocab,
            train_responses=tr_r,
            test_responses=te_r,
            task=CAUSAL_LM,
            label_names=tuple(f"c{i}" for i in range(num_labels)),
        )
    if name == "sst2":
        tr_x, tr_y, te_x, te_y = load_sst2(num_train, num_test, subsample_seed)
        return TextClassificationData(tr_x, tr_y, te_x, te_y, num_labels=2, vocab=None)
    if name == "ag_news":
        # num_labels is PINNED to 4 rather than taken from the config, exactly as sst2
        # pins 2: the label count is a property of the dataset, and a config that
        # disagreed would silently build a head with the wrong width.
        tr_x, tr_y, te_x, te_y = load_ag_news(num_train, num_test, subsample_seed)
        return TextClassificationData(tr_x, tr_y, te_x, te_y, num_labels=4, vocab=None)
    if name == "banking77":
        # The label count and names belong to the pinned dataset, not the config.  Keeping
        # the official test split global matches this framework's Spider/EuroSAT metric
        # convention; only the training split is partitioned among clients.
        splits = load_banking77(num_train, num_test, subsample_seed)
        return TextClassificationData(
            splits.train_texts,
            splits.train_labels,
            splits.test_texts,
            splits.test_labels,
            num_labels=77,
            vocab=None,
            label_names=splits.class_names,
            info={
                "dataset_repo": BANKING77_REPO,
                "dataset_revision": BANKING77_REVISION,
                "official_train_size": 10_003,
                "official_test_size": 3_080,
            },
        )
    if name == "dolly15k":
        # num_labels is PINNED to the 8 Dolly categories for the same reason sst2 pins 2
        # and ag_news pins 4: the count is a property of the dataset. Here it sizes no
        # head at all (a causal LM has none) but it does size the skew index's ceiling,
        # which is what makes alpha comparable across testbeds.
        splits = load_dolly(num_train, num_test, subsample_seed)
        return TaskData(
            train_texts=splits.train_prompts,
            train_labels=splits.train_categories,
            test_texts=splits.test_prompts,
            test_labels=splits.test_categories,
            num_labels=len(splits.category_names),
            vocab=None,
            train_responses=splits.train_responses,
            test_responses=splits.test_responses,
            task=CAUSAL_LM,
            label_names=splits.category_names,
        )
    if name == "spider":
        # The "label" is the DATABASE. label_names[label] recovers the db_id the executor
        # runs against, and the same labels give a federated arm a database-valued axis
        # to partition over (mildly non-IID in practice — see data/spider.py, which
        # measures what a Dirichlet over 140 databases actually produces). num_labels is PINNED to
        # the observed database count for the same reason sst2 pins 2 — it is a property
        # of the dataset, not of the config.
        kwargs = {} if max_prompt_tokens is None else {"max_prompt_tokens": max_prompt_tokens}
        splits = load_spider(num_train, num_test, subsample_seed, **kwargs)
        return TaskData(
            train_texts=splits.train_prompts,
            train_labels=splits.train_db_ids,
            test_texts=splits.test_prompts,
            test_labels=splits.test_db_ids,
            num_labels=len(splits.db_names),
            vocab=None,
            train_responses=splits.train_responses,
            test_responses=splits.test_responses,
            task=CAUSAL_LM,
            label_names=splits.db_names,
            info={"schema_truncation_rate": splits.truncation_rate},
        )
    if name == "eurosat":
        # The label is the LAND-COVER CLASS, so label_names[label] is the stored CamelCase
        # name and the same labels are the Dirichlet axis for a future federated arm.
        # num_labels is PINNED to 10 for the same reason sst2 pins 2: it is a property of
        # the dataset, and here it also sizes the frozen zero-shot head — a config that
        # disagreed would build a classifier of the wrong width.
        splits = load_eurosat(num_train, num_test, subsample_seed, heldout_split=heldout_split)
        return TaskData(
            # The text fields are EMPTY, not a stand-in: see the module docstring on why
            # pixels never travel in train_texts.
            train_texts=[],
            train_labels=splits.train_labels,
            test_texts=[],
            test_labels=splits.test_labels,
            num_labels=len(splits.class_names),
            vocab=None,
            task=IMAGE_CLS,
            label_names=splits.class_names,
            train_images=splits.train_pixels,
            test_images=splits.test_pixels,
            info={
                "pixel_cache_key_train": splits.cache_key_train,
                "pixel_cache_key_test": splits.cache_key_test,
                # Which split every held-out number in this run was computed on. Reaches
                # summary.json, so a rank/lr/budget chosen on `validation` can never be
                # quoted as if it had been measured on `test`.
                "heldout_split": splits.heldout_split,
            },
        )
    raise KeyError(
        f"unknown dataset '{name}' (expected 'synthetic', 'sst2', 'ag_news', 'banking77', "
        "'dolly15k', 'spider' or 'eurosat')"
    )


def build_federated_data(
    name: str,
    num_train: int | None,
    num_test: int | None,
    num_clients: int,
    alpha: float,
    seed: int,
    num_labels: int = 2,
    subsample_seed: int | None = None,
    max_prompt_tokens: int | None = None,
    partition_scheme: str = "dirichlet",
) -> FederatedData:
    """Build the splits and the client partition.

    ``partition_scheme`` selects HOW clients are carved out, and the two are not variants
    of one axis:

    * ``dirichlet`` — split each label's examples across clients by a Dirichlet(alpha).
      Every client can see every label; alpha controls how unevenly.
    * ``disjoint_labels`` — give each label WHOLE to one client. Ownership, not
      proportion: a client has never seen the labels it does not own, and ``alpha`` is
      meaningless (recorded as -1.0). For Spider, where the label is the database, this is
      the realistic federation; the Dirichlet is not (see data/partition.py).
    """
    data = build_dataset(
        name, num_train, num_test, seed, num_labels, subsample_seed, max_prompt_tokens
    )
    if partition_scheme == "disjoint_labels":
        partition = disjoint_label_partition(data.train_labels, num_clients, seed)
    elif partition_scheme == "dirichlet":
        partition = dirichlet_partition(data.train_labels, num_clients, alpha, seed)
    else:
        raise KeyError(
            f"unknown partition scheme {partition_scheme!r}; "
            "expected 'dirichlet' or 'disjoint_labels'"
        )
    return FederatedData(data, partition)


def build_tokenizer(kind: str, data: TaskData, model_name: str | None = None):
    """Factory: 'simple' (hermetic, needs data.vocab) or 'hf' (needs model_name)."""
    if kind == "simple":
        if data.vocab is None:
            raise ValueError("SimpleTokenizer requires a dataset with a vocab (synthetic)")
        return SimpleTokenizer(data.vocab)
    if kind == "hf":
        if not model_name:
            raise ValueError("HF tokenizer requires model_name")
        return build_hf_tokenizer(model_name)
    raise KeyError(f"unknown tokenizer kind '{kind}' (expected 'simple' or 'hf')")
