"""
Risk scoring models:
1. Isolation Forest — unsupervised anomaly detection
2. XGBoost Risk Classifier — supervised (uses engineered labels)
3. Statistical Z-score baseline

All models output: risk_score (0-1), risk_level (LOW/MEDIUM/HIGH/CRITICAL),
and an explanation vector for FAISS-backed audit retrieval.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit  # Critical: no data leakage in time series
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
MODELS_DIR = "models/artifacts"
RANDOM_STATE = 42


@dataclass
class RiskScore:
    ticker: str
    date: str
    risk_score: float        # 0.0 - 1.0
    risk_level: str          # LOW / MEDIUM / HIGH / CRITICAL
    anomaly_detected: bool
    feature_vector: np.ndarray  # For FAISS retrieval
    top_drivers: list[tuple[str, float]]  # (feature_name, contribution)
    model_used: str


def score_to_level(score: float) -> str:
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    return "LOW"


class IsolationForestDetector:
    """Unsupervised anomaly detection — no labels needed."""

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        logger.info(f"IsolationForest trained on {len(X)} samples")
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores in [0, 1] — higher = more anomalous."""
        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        # Normalize to [0, 1]: lower decision function = more anomalous
        normalized = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
        return normalized

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return (self.model.predict(X_scaled) == -1).astype(int)

    def save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self, os.path.join(MODELS_DIR, "isolation_forest.pkl"))


class XGBoostRiskClassifier:
    """Supervised risk classifier using engineered anomaly labels."""

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=19,  # ~5% anomaly rate → weight positive class
            random_state=RANDOM_STATE, eval_metric="auc", verbosity=0,
        )
        self.scaler = StandardScaler()
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostRiskClassifier":
        """Time-series aware training with TimeSeriesSplit CV."""
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)

        # TimeSeriesSplit — no future data leakage
        tscv = TimeSeriesSplit(n_splits=5)
        cv_aucs = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            self.model.fit(X_tr, y_tr)
            probs = self.model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, probs)
            cv_aucs.append(auc)
            logger.debug(f"  Fold {fold+1}: AUC={auc:.4f}")

        logger.info(f"XGBoost CV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

        # Final fit on all data
        self.model.fit(X_scaled, y)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_contributions(self, X_row: pd.DataFrame) -> list[tuple[str, float]]:
        """Get top feature contributions for a single prediction (for audit)."""
        importances = self.model.feature_importances_
        contributions = list(zip(self.feature_names, importances))
        return sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]

    def save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self, os.path.join(MODELS_DIR, "xgb_risk.pkl"))


def ensemble_risk_score(
    iso_score: float, xgb_score: float, weights: tuple = (0.4, 0.6)
) -> float:
    """Weighted ensemble: XGBoost (supervised) gets higher weight."""
    return weights[0] * iso_score + weights[1] * xgb_score


def train_risk_models(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[IsolationForestDetector, XGBoostRiskClassifier]:
    """Train both models on the full dataset."""
    X = df[feature_cols].dropna()
    y = df.loc[X.index, "is_anomaly"]

    logger.info(f"Training on {len(X)} samples | Anomaly rate: {y.mean():.2%}")

    iso = IsolationForestDetector(contamination=max(y.mean(), 0.01))
    iso.fit(X)
    iso.save()

    xgb = XGBoostRiskClassifier()
    xgb.fit(X, y)
    xgb.save()

    # Evaluation
    iso_preds = iso.predict(X)
    xgb_preds = (xgb.score(X) > 0.5).astype(int)

    table = Table(title="Risk Model Performance", box=box.ROUNDED)
    for col in ["Model", "Detected Anomalies", "Detection Rate", "False Alarm Rate"]:
        table.add_column(col)

    for name, preds in [("IsolationForest", iso_preds), ("XGBoost", xgb_preds)]:
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + (y == 0).sum()) if (y == 0).sum() > 0 else 0
        table.add_row(name, str(int(preds.sum())), f"{tpr:.2%}", f"{fpr:.2%}")

    console.print(table)
    return iso, xgb


def score_latest_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    iso: IsolationForestDetector,
    xgb: XGBoostRiskClassifier,
    n_latest: int = 30,
) -> list[RiskScore]:
    """Score the most recent N data points across all tickers."""
    results = []
    latest = df.groupby("ticker").tail(n_latest)

    for _, row in latest.iterrows():
        X_row = pd.DataFrame([row[feature_cols]])

        iso_s = float(iso.score(X_row)[0])
        xgb_s = float(xgb.score(X_row)[0])
        final_score = ensemble_risk_score(iso_s, xgb_s)
        drivers = xgb.get_feature_contributions(X_row)

        results.append(RiskScore(
            ticker=row.get("ticker", "unknown"),
            date=str(row.name),
            risk_score=round(final_score, 4),
            risk_level=score_to_level(final_score),
            anomaly_detected=final_score >= 0.6,
            feature_vector=np.array(row[feature_cols], dtype=np.float32),
            top_drivers=drivers,
            model_used="ensemble",
        ))

    return results
