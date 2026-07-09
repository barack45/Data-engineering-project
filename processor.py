import pandas as pd
import numpy as np


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les transformations features sur un DataFrame transactions."""
    df = df.copy()

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["total_amount"] = (df["quantity"] * df["unit_price"]).fillna(0)

    stats = df.groupby("customer_id")["total_amount"].mean().reset_index()
    stats.columns = ["customer_id", "avg_transaction_val"]

    df = df.merge(stats, on="customer_id", how="left")
    df["deviation_from_avg"] = (df["total_amount"] / df["avg_transaction_val"]).fillna(0)
    df["recent_txn_freq"] = df.groupby("customer_id")["transaction_date"].transform("count")

    return df.replace([np.inf, -np.inf], 0).fillna(0)


def load_and_process(file_path):
    """Charge un CSV transactions et retourne le DataFrame enrichi."""
    df = pd.read_csv(file_path, sep=";")
    return process_dataframe(df)
