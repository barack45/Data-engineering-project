"""Génère un CSV transactions brut pour le pipeline (données de démo)."""

from pathlib import Path

from src.ingest import generate_transactions_csv

ROOT = Path(__file__).resolve().parent.parent
SUPPORT_PATH = ROOT / "data" / "support_tickets_big.txt"
OUTPUT_PATH = ROOT / "data" / "transactions.csv"


def main() -> None:
    path = generate_transactions_csv(OUTPUT_PATH, SUPPORT_PATH)
    print(f"Dataset généré : {path}")


if __name__ == "__main__":
    main()
