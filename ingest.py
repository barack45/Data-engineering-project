"""Génération de données brutes de démonstration."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = ["FR", "DE", "ES", "IT", "UK", "US", "FRANCE", "FRA"]


def extract_customers_from_support(path: Path, limit: int = 5000) -> list[str]:
    text = path.read_text(encoding="utf-8")
    customers = re.findall(r"CUST_\d+", text)
    unique = list(dict.fromkeys(customers))
    return unique[:limit] if unique else [f"CUST_{i:05d}" for i in range(limit)]


def build_transactions(customers: list[str], n_rows: int = 8000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for i in range(n_rows):
        customer_id = rng.choice(customers)
        quantity_raw = "GRATUIT" if rng.random() < 0.02 else int(rng.integers(1, 12))
        unit_price = round(float(rng.uniform(5, 400)), 2)
        email = f"{customer_id.lower()}@mail.com"
        if rng.random() < 0.03:
            email = email.replace("@", "@@")

        rows.append(
            {
                "transaction_id": f"TXN_{i:07d}",
                "order_id": f"ORD_{i:06d}",
                "customer_id": customer_id,
                "customer_email": email,
                "country": rng.choice(COUNTRIES),
                "quantity": quantity_raw,
                "unit_price": unit_price,
                "transaction_date": pd.Timestamp("2024-01-01")
                + pd.Timedelta(days=int(rng.integers(0, 540))),
            }
        )
    return pd.DataFrame(rows)


def generate_transactions_csv(output_path: Path, support_path: Path, n_rows: int = 8000) -> Path:
    customers = extract_customers_from_support(support_path)
    df = build_transactions(customers, n_rows=n_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep=";", index=False)
    return output_path
