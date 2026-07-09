"""Point d'entrée du pipeline de données e-commerce (Bronze → Silver → Gold → ML)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.ingest import generate_transactions_csv
from src import (
    build_gold_datamart,
    clean_reviews,
    clean_transactions,
    get_trained_model,
    get_trained_xgb_model,
    load_raw_transactions,
    parse_support_tickets,
    process_dataframe,
    save_silver,
)

FEATURE_COLUMNS = [
    "total_amount",
    "avg_transaction_val",
    "deviation_from_avg",
    "recent_txn_freq",
]

DEFAULT_PATHS = {
    "transactions_csv": Path("data/transactions.csv"),
    "reviews_jsonl": Path("data/customer_reviews_big 1.jsonl"),
    "support_txt": Path("data/support_tickets_big.txt"),
    "transactions_silver": Path("output/transactions_silver.parquet"),
    "support_silver": Path("output/support_silver.parquet"),
    "reviews_silver": Path("output/reviews_silver.parquet"),
    "gold_parquet": Path("datamart_gold_final.parquet"),
    "ml_results": Path("ml_results.npz"),
}


def ensure_raw_transactions(path: Path, support_path: Path) -> None:
    if path.exists():
        return
    print(f"→ Génération du fichier brut manquant : {path}")
    generate_transactions_csv(path, support_path)


def run_silver_layer(paths: dict[str, Path]) -> tuple:
    print("\n=== COUCHE SILVER ===")
    ensure_raw_transactions(paths["transactions_csv"], paths["support_txt"])

    df_raw = load_raw_transactions(paths["transactions_csv"])
    df_trans = clean_transactions(df_raw)
    save_silver(df_trans, paths["transactions_silver"])
    print(f"✅ Transactions Silver : {paths['transactions_silver']} ({len(df_trans)} lignes)")

    df_support = parse_support_tickets(paths["support_txt"])
    save_silver(df_support, paths["support_silver"])
    print(f"✅ Support Silver : {paths['support_silver']} ({len(df_support)} tickets)")

    df_reviews = clean_reviews(paths["reviews_jsonl"])
    save_silver(df_reviews, paths["reviews_silver"])
    print(f"✅ Avis Silver : {paths['reviews_silver']} ({len(df_reviews)} avis)")

    return df_trans, df_support, df_reviews


def run_gold_layer(df_trans, df_support, df_reviews, gold_path: Path):
    print("\n=== COUCHE GOLD ===")
    df_gold = build_gold_datamart(df_trans, df_support, df_reviews)
    save_silver(df_gold, gold_path)
    print(f"✅ Datamart Gold : {gold_path} ({len(df_gold)} lignes)")
    print(f"   Transactions avec avis : {(df_gold['rating'] != -1).sum()} ({(df_gold['rating'] != -1).mean():.1%})")
    return df_gold


def run_ml_layer(df_gold, target: str, output_path: Path):
    print("\n=== ENTRAÎNEMENT ML ===")
    df_features = process_dataframe(df_gold)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df_features.columns]
    if missing_features:
        raise ValueError(f"Colonnes features manquantes : {missing_features}")

    if target not in df_features.columns:
        raise ValueError(f"Colonne cible '{target}' absente du datamart.")

    X = df_features[FEATURE_COLUMNS]
    y = df_features[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    rf_model, rf_scaler = get_trained_model(X_train, y_train)
    xgb_model, xgb_scaler = get_trained_xgb_model(X_train, y_train)

    X_test_rf = rf_scaler.transform(X_test)
    X_test_xgb = xgb_scaler.transform(X_test)

    rf_preds = rf_model.predict(X_test_rf)
    rf_probs = rf_model.predict_proba(X_test_rf)[:, 1]
    xgb_preds = xgb_model.predict(X_test_xgb)
    xgb_probs = xgb_model.predict_proba(X_test_xgb)[:, 1]

    print("\n--- Random Forest ---")
    print(classification_report(y_test, rf_preds, target_names=["Légitime", "Suspect"]))
    print("\n--- XGBoost ---")
    print(classification_report(y_test, xgb_preds, target_names=["Légitime", "Suspect"]))

    np.savez(
        output_path,
        y_test=y_test.to_numpy(),
        rf_preds=rf_preds,
        rf_probs=rf_probs,
        xgb_preds=xgb_preds,
        xgb_probs=xgb_probs,
    )
    print(f"✅ Résultats ML : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline complet e-commerce")
    parser.add_argument("--target", default="is_suspect", help="Colonne cible ML")
    parser.add_argument("--skip-ml", action="store_true", help="Ne lance que Silver + Gold")
    args = parser.parse_args()

    paths = DEFAULT_PATHS.copy()
    paths["transactions_silver"].parent.mkdir(parents=True, exist_ok=True)

    print("=== PIPELINE E-COMMERCE ===")
    df_trans, df_support, df_reviews = run_silver_layer(paths)
    df_gold = run_gold_layer(df_trans, df_support, df_reviews, paths["gold_parquet"])

    if not args.skip_ml:
        run_ml_layer(df_gold, args.target, paths["ml_results"])

    print("\n🎉 Pipeline terminé avec succès.")


if __name__ == "__main__":
    main()
