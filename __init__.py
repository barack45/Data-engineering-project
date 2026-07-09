"""Package src — nettoyage, datamart, traitement et modélisation ML."""

from .cleaning import (
    clean_reviews,
    clean_transactions,
    load_raw_transactions,
    parse_support_tickets,
    save_silver,
)
from .datamart import build_gold_datamart
from .model import get_trained_model, get_trained_xgb_model
from .processor import load_and_process, process_dataframe

__all__ = [
    "build_gold_datamart",
    "clean_reviews",
    "clean_transactions",
    "get_trained_model",
    "get_trained_xgb_model",
    "load_and_process",
    "load_raw_transactions",
    "parse_support_tickets",
    "process_dataframe",
    "save_silver",
]
