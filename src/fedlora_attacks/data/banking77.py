"""BANKING77 loader with the official train/test split contract.

The ACL 2025 reference uses the original BANKING77 split: 10,003 training queries,
3,080 test queries, and 77 intents.  This loader pins the maintained PolyAI mirror to a
specific revision and refuses a changed split or label vocabulary instead of producing a
plausible but incomparable result.
"""

from __future__ import annotations

from dataclasses import dataclass

BANKING77_REPO = "PolyAI/banking77"
BANKING77_REVISION = "1fb62b1bb4635df59a8e1b2f2bc5e0643b2856c8"
BANKING77_SPLIT_SIZES = {"train": 10_003, "test": 3_080}

# Hugging Face's ClassLabel order at the pinned revision.  The capitalization and question
# mark in two upstream labels are intentional: changing them would change label ids.
BANKING77_CLASSES = (
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
)


class Banking77Error(RuntimeError):
    """The pinned dataset no longer satisfies the experimental contract."""


@dataclass(frozen=True)
class Banking77Splits:
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    class_names: tuple[str, ...] = BANKING77_CLASSES


def verify_banking77(dataset) -> tuple[str, ...]:
    """Verify split sizes, columns, label ids, and label-name order."""
    sizes = {name: len(dataset[name]) for name in ("train", "test") if name in dataset}
    if sizes != BANKING77_SPLIT_SIZES:
        raise Banking77Error(
            f"BANKING77 split sizes changed: expected {BANKING77_SPLIT_SIZES}, got {sizes}"
        )
    for split_name in ("train", "test"):
        split = dataset[split_name]
        if not {"text", "label"}.issubset(split.features):
            raise Banking77Error(
                f"BANKING77 {split_name} columns changed: {sorted(split.features)}"
            )
        names = tuple(split.features["label"].names)
        if names != BANKING77_CLASSES:
            raise Banking77Error(
                "BANKING77 label vocabulary/order changed; refusing to remap a 77-way head"
            )
        labels = {int(y) for y in split["label"]}
        if labels != set(range(77)):
            raise Banking77Error(
                f"BANKING77 {split_name} label ids changed: expected 0..76, got {sorted(labels)}"
            )
    return BANKING77_CLASSES


def _limited(split, limit: int | None, seed: int):
    """Keep the official split whole, or draw a stable smoke-test subset."""
    if limit is None or limit >= len(split):
        return split
    if limit < 1:
        raise ValueError(f"BANKING77 subset size must be positive, got {limit}")
    return split.shuffle(seed=seed).select(range(limit))


def load_banking77(
    num_train: int | None,
    num_test: int | None,
    seed: int,
) -> Banking77Splits:
    """Load BANKING77 without deriving a validation split or touching the official test."""
    from datasets import load_dataset

    dataset = load_dataset(BANKING77_REPO, revision=BANKING77_REVISION)
    class_names = verify_banking77(dataset)
    train = _limited(dataset["train"], num_train, seed)
    test = _limited(dataset["test"], num_test, seed)
    return Banking77Splits(
        train_texts=list(train["text"]),
        train_labels=[int(y) for y in train["label"]],
        test_texts=list(test["text"]),
        test_labels=[int(y) for y in test["label"]],
        class_names=class_names,
    )
