"""
Nettoyage des données brutes — couche Silver.

NOTE SUR LE TRAITEMENT DES AVIS :
Le dataset présente un taux élevé de transactions sans avis client (environ 99%).
Pour assurer la stabilité des modèles et de l'application Streamlit, ces valeurs
sont remplacées par des constantes neutres ('NO_REVIEW', -1, etc.) plutôt que
d'être supprimées, afin de conserver l'intégralité du volume de transactions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RATING_MAP = {
    "excellent": 5,
    "tres bien": 4,
    "très bien": 4,
    "bien": 4,
    "moyen": 3,
    "passable": 2,
    "mauvais": 1,
    "nul": 1,
}


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le dataset transactions et retourne la couche Silver."""
    df_trans = df.copy()
    df_trans = df_trans.drop_duplicates(subset=["transaction_id"])

    df_trans["quantity"] = df_trans["quantity"].replace("GRATUIT", 0)
    df_trans["quantity"] = pd.to_numeric(df_trans["quantity"], errors="coerce")
    df_trans["quantity"] = df_trans["quantity"].fillna(1).astype(int)

    df_trans["unit_price"] = pd.to_numeric(df_trans["unit_price"], errors="coerce")
    median_price = df_trans["unit_price"].median()
    df_trans["unit_price"] = df_trans["unit_price"].fillna(median_price if pd.notna(median_price) else 0)

    if "customer_email" in df_trans.columns:
        df_trans["customer_email"] = df_trans["customer_email"].astype(str).str.replace("@@", "@")
        df_trans["customer_email"] = df_trans["customer_email"].replace(
            ["nan", "None", ""], "unknown@shopeuro.com"
        )

    df_trans["transaction_date"] = pd.to_datetime(df_trans["transaction_date"], errors="coerce")
    df_trans["transaction_date"] = df_trans["transaction_date"].ffill()

    if "country" in df_trans.columns:
        df_trans["country"] = df_trans["country"].astype(str).str.upper().str.strip()

    return df_trans


def parse_support_tickets(file_path: str | Path) -> pd.DataFrame:
    """Parse le fichier support texte et retourne un DataFrame structuré."""
    path = Path(file_path)
    text_content = path.read_text(encoding="utf-8")
    tickets_bruts = text_content.split("========================================")
    parsed_tickets: list[dict] = []

    for block in tickets_bruts:
        if not block.strip():
            continue

        customer_id = None
        email = None
        ticket_date = None
        ticket_id = None
        order_id = None
        content = "Aucun contenu"

        from_match = re.search(r"From:\s*(CUST_\d+)\s*<([^>]+)>", block)
        if from_match:
            customer_id = from_match.group(1)
            email = from_match.group(2)
            date_match = re.search(r"Date:\s*([^\n]+)", block)
            ticket_date = date_match.group(1).strip() if date_match else None
        else:
            header_match = re.search(
                r"Ticket\s*#?(TKT_\d+).*?Customer:?\s*(CUST_\d+).*?Date:?\s*([^\n|]+)",
                block,
                re.IGNORECASE,
            )
            if header_match:
                ticket_id = header_match.group(1)
                customer_id = header_match.group(2)
                ticket_date = header_match.group(3).strip()
            else:
                bracket_match = re.search(
                    r"\[([^\]]+)\]\s*Ticket\s*#?(TKT_\d+)\s*\|\s*Customer\s*(CUST_\d+)",
                    block,
                    re.IGNORECASE,
                )
                if bracket_match:
                    ticket_date = bracket_match.group(1).strip()
                    ticket_id = bracket_match.group(2)
                    customer_id = bracket_match.group(3)

        tkt_match = re.search(r"Ticket\s*#?(TKT_\d+)", block, re.IGNORECASE)
        order_match = re.search(r"Order:?\s*(ORD_\d+)", block, re.IGNORECASE)
        content_match = re.search(r"Subject:[^\n]+\n(.*)", block, re.DOTALL)

        if tkt_match and not ticket_id:
            ticket_id = tkt_match.group(1)
        if order_match:
            order_id = order_match.group(1)
        if content_match:
            content = content_match.group(1).strip()
        elif "---" in block:
            content = block.split("---", 1)[-1].strip()

        if customer_id:
            if not email:
                email = f"{customer_id.lower()}@mail.com"
            parsed_tickets.append(
                {
                    "ticket_id": ticket_id,
                    "ticket_date": ticket_date,
                    "customer_id": customer_id,
                    "customer_email": email,
                    "order_id": order_id,
                    "ticket_content": content,
                }
            )

    df_support = pd.DataFrame(parsed_tickets)
    if df_support.empty:
        return df_support

    df_support = df_support.drop_duplicates(subset=["ticket_id"])
    df_support["ticket_date"] = pd.to_datetime(df_support["ticket_date"], errors="coerce")
    df_support["ticket_date"] = df_support["ticket_date"].ffill()
    return df_support


def _normalize_rating(value) -> float:
    if pd.isna(value):
        return -1.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text.isdigit():
        return float(text)
    return float(RATING_MAP.get(text, 3))


def clean_reviews(file_path: str | Path) -> pd.DataFrame:
    """Parse et nettoie le fichier JSONL des avis clients."""
    path = Path(file_path)
    records: list[dict] = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata = row.get("metadata", {})
            user_meta = metadata.get("user", {})
            location = user_meta.get("location", {})
            country = location.get("country", {}).get("code", "AUTRE")
            helpful_votes = metadata.get("helpful_votes", 0)
            total_reviews = user_meta.get("total_reviews", 1)
            account_age = user_meta.get("account_age_days", 365)

            bot_score = min(
                1.0,
                max(0.0, (helpful_votes / max(total_reviews, 1)) * 0.4 + (1 / max(account_age, 1)) * 100),
            )

            records.append(
                {
                    "review_id": row.get("review_id"),
                    "user_id": row.get("user_id"),
                    "product_id": row.get("product_id"),
                    "rating": _normalize_rating(row.get("rating")),
                    "review_text": row.get("review_text", ""),
                    "review_date": row.get("review_date"),
                    "country": str(country).upper(),
                    "bot_score": round(bot_score, 4),
                }
            )

    df_reviews = pd.DataFrame(records)
    if df_reviews.empty:
        return df_reviews

    df_reviews = df_reviews.drop_duplicates(subset=["review_id"])
    df_reviews["review_date"] = pd.to_datetime(df_reviews["review_date"], errors="coerce", dayfirst=True)
    return df_reviews


def load_raw_transactions(file_path: str | Path) -> pd.DataFrame:
    """Charge le CSV transactions brut (séparateur ';')."""
    return pd.read_csv(file_path, sep=";")


def save_silver(df: pd.DataFrame, file_path: str | Path) -> Path:
    """Sauvegarde un DataFrame en Parquet Silver."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
