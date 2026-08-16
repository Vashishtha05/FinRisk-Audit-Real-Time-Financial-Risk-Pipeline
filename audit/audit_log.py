"""
FAISS-backed audit log and monitoring system.

Every risk score decision is:
1. Stored in an append-only FAISS index (feature vectors)
2. Logged with full metadata (ticker, date, score, drivers)
3. Searchable: 'find me all past cases similar to this current alert'

This is the auditability layer — critical for regulated environments.
Regulators can ask: 'Why did your system flag X on date Y?'
Answer: 'Here are the 5 most similar historical cases and their outcomes.'
"""
from __future__ import annotations
import os
import json
import pickle
import numpy as np
import faiss
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from loguru import logger
from scoring.risk_scorer import RiskScore

AUDIT_INDEX_PATH = "audit/faiss_audit.index"
AUDIT_META_PATH = "audit/audit_metadata.pkl"
AUDIT_LOG_PATH = "audit/audit_log.jsonl"
PLOTS_DIR = "reports/plots"


class AuditLog:
    """
    Append-only FAISS-backed audit log.
    Every scored event is stored and retrievable by similarity.
    """

    def __init__(self):
        os.makedirs("audit", exist_ok=True)
        self._dim = None
        self._index = None
        self._metadata: list[dict] = []
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(AUDIT_INDEX_PATH) and os.path.exists(AUDIT_META_PATH):
            try:
                self._index = faiss.read_index(AUDIT_INDEX_PATH)
                with open(AUDIT_META_PATH, "rb") as f:
                    self._metadata = pickle.load(f)
                self._dim = self._index.d
                logger.info(f"Loaded audit log: {len(self._metadata)} events")
                return
            except Exception as e:
                logger.warning(f"Could not load audit log: {e}")
        # Will be initialized on first add
        self._index = None

    def log_risk_score(self, score: RiskScore) -> None:
        """Log a risk scoring event to the audit trail."""
        vec = score.feature_vector.astype(np.float32)

        if self._index is None:
            self._dim = len(vec)
            self._index = faiss.IndexFlatL2(self._dim)

        norm = np.linalg.norm(vec)
        vec_normalized = vec / (norm + 1e-9)
        self._index.add(vec_normalized.reshape(1, -1))

        meta = {
            "ticker": score.ticker,
            "date": score.date,
            "risk_score": score.risk_score,
            "risk_level": score.risk_level,
            "anomaly_detected": score.anomaly_detected,
            "top_drivers": score.top_drivers,
            "logged_at": datetime.utcnow().isoformat(),
        }
        self._metadata.append(meta)

        # Append to JSONL log
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(meta) + "\n")

        self._save()

    def log_batch(self, scores: list[RiskScore]) -> None:
        for score in scores:
            self.log_risk_score(score)
        logger.info(f"Logged {len(scores)} events to audit trail")

    def find_similar_events(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[dict]:
        """Retrieve most similar past events for a given feature vector."""
        if self._index is None or len(self._metadata) == 0:
            return []

        query = query_vector.astype(np.float32)
        query /= np.linalg.norm(query) + 1e-9

        k = min(top_k, len(self._metadata))
        distances, indices = self._index.search(query.reshape(1, -1), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append({**self._metadata[idx], "similarity_distance": float(dist)})
        return results

    def get_risk_history(self, ticker: str) -> pd.DataFrame:
        """Get full risk history for a specific ticker."""
        events = [m for m in self._metadata if m["ticker"] == ticker]
        return pd.DataFrame(events)

    def _save(self):
        faiss.write_index(self._index, AUDIT_INDEX_PATH)
        with open(AUDIT_META_PATH, "wb") as f:
            pickle.dump(self._metadata, f)


def generate_monitoring_dashboard(
    risk_scores: list[RiskScore],
    audit_log: AuditLog,
) -> str:
    """Generate a monitoring dashboard plot."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "risk_dashboard.png")

    df = pd.DataFrame([{
        "ticker": s.ticker,
        "date": s.date,
        "risk_score": s.risk_score,
        "risk_level": s.risk_level,
        "anomaly": s.anomaly_detected,
    } for s in risk_scores])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Risk score distribution by ticker
    ax = axes[0, 0]
    tickers = df["ticker"].unique()
    data_by_ticker = [df[df["ticker"] == t]["risk_score"].values for t in tickers]
    ax.boxplot(data_by_ticker, labels=tickers, patch_artist=True)
    ax.set_title("Risk Score Distribution by Asset", fontweight="bold")
    ax.set_ylabel("Risk Score (0-1)")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0.6, color="red", linestyle="--", alpha=0.7, label="HIGH threshold")
    ax.axhline(0.4, color="orange", linestyle="--", alpha=0.7, label="MEDIUM threshold")
    ax.legend(fontsize=8)

    # 2. Risk level heatmap
    ax = axes[0, 1]
    level_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    pivot = df.pivot_table(
        values="risk_score", index="ticker",
        aggfunc=["mean", "max", "std"]
    ).round(3)
    sns.heatmap(
        df.groupby("ticker")["risk_score"].agg(["mean", "max", "std"]).round(3),
        annot=True, fmt=".3f", ax=ax, cmap="RdYlGn_r"
    )
    ax.set_title("Risk Score Statistics by Asset", fontweight="bold")

    # 3. Anomaly detection rate
    ax = axes[1, 0]
    anomaly_rates = df.groupby("ticker")["anomaly"].mean().sort_values(ascending=False)
    bars = ax.bar(anomaly_rates.index, anomaly_rates.values, color=[
        "red" if v > 0.2 else "orange" if v > 0.1 else "green"
        for v in anomaly_rates.values
    ])
    ax.set_title("Anomaly Detection Rate by Asset", fontweight="bold")
    ax.set_ylabel("Fraction of Days Flagged")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0.05, color="black", linestyle="--", alpha=0.5, label="Expected (5%)")
    ax.legend()

    # 4. Risk level distribution
    ax = axes[1, 1]
    level_counts = df["risk_level"].value_counts()
    colors = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "darkred"}
    ax.pie(
        level_counts.values,
        labels=level_counts.index,
        colors=[colors.get(l, "gray") for l in level_counts.index],
        autopct="%1.1f%%",
    )
    ax.set_title("Risk Level Distribution", fontweight="bold")

    plt.suptitle("Financial Risk Monitoring Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Dashboard saved: {path}")
    return path
