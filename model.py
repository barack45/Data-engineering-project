from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def get_trained_model(X_train, y_train):
    """
    Entraîne un modèle Random Forest avec standardisation
    et pondération pour gérer le déséquilibre des classes.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def get_trained_xgb_model(X_train, y_train):
    """Entraîne un modèle XGBoost avec standardisation."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler
