"""Construction du datamart Gold à partir des couches Silver."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _anonymize_email(email: str) -> str:
    if pd.isna(email) or "@" not in str(email):
        return "unknown@shopeuro.com"
    local, domain = str(email).split("@", 1)
    if len(local) <= 2:
        masked_local = "*" * len(local)
    else:
        masked_local = local[0] + "*" * (len(local) - 2) + local[-1]
    return f"{masked_local}@{domain}"


def _user_to_customer_id(user_id: str) -> str:
    digits = re.sub(r"\D", "", str(user_id))
    return f"CUST_{int(digits):05d}" if digits else "CUST_00000"


def build_gold_datamart(
    transactions_silver: pd.DataFrame,
    support_silver: pd.DataFrame,
    reviews_silver: pd.DataFrame,
    review_attach_rate: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fusionne transactions, support et avis en un datamart Gold unique.

    Environ 99 % des transactions n'ont pas d'avis — on conserve rating=-1
    et bot_score=0 pour ces lignes.
    """
    rng = np.random.default_rng(random_state)
    gold = transactions_silver.copy()

    if "order_id" not in gold.columns:
        gold["order_id"] = [f"ORD_{i:06d}" for i in range(len(gold))]

    support_counts = (
        support_silver.groupby("customer_id").size().reset_index(name="nb_tickets_support")
        if not support_silver.empty
        else pd.DataFrame(columns=["customer_id", "nb_tickets_support"])
    )
    gold = gold.merge(support_counts, on="customer_id", how="left")
    gold["nb_tickets_support"] = gold["nb_tickets_support"].fillna(0).astype(int)

    gold["rating"] = -1.0
    gold["bot_score"] = 0.0
    gold["review_text"] = "NO_REVIEW"

    if not reviews_silver.empty:
        reviews = reviews_silver.copy()
        reviews["customer_id"] = reviews["user_id"].map(_user_to_customer_id)
        reviews = reviews.drop_duplicates(subset=["review_id"])

        n_with_review = max(1, int(len(gold) * review_attach_rate))
        review_sample = reviews.sample(n=min(n_with_review, len(reviews)), random_state=random_state)
        review_idx = rng.choice(gold.index.to_numpy(), size=len(review_sample), replace=False)

        for idx, (_, review_row) in zip(review_idx, review_sample.iterrows()):
            gold.at[idx, "rating"] = review_row["rating"]
            gold.at[idx, "bot_score"] = review_row["bot_score"]
            gold.at[idx, "review_text"] = review_row["review_text"]
            if "country" in review_row and (pd.isna(gold.at[idx, "country"]) if "country" in gold.columns else True):
                gold.at[idx, "country"] = review_row["country"]

    if "country" not in gold.columns:
        gold["country"] = "FR"
    gold["country"] = gold["country"].fillna("AUTRE").astype(str).str.upper()

    if "customer_email" not in gold.columns:
        gold["customer_email"] = gold["customer_id"].str.lower() + "@mail.com"

    anonymize_mask = rng.random(len(gold)) < 0.35
    gold.loc[anonymize_mask, "customer_email"] = gold.loc[anonymize_mask, "customer_email"].map(_anonymize_email)

    if "is_suspect" not in gold.columns:
        gold["is_suspect"] = (
            (gold["quantity"] >= 10)
            | (gold["unit_price"] >= 350)
            | (gold["nb_tickets_support"] >= 3)
        ).astype(int)

    return gold
