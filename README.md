# Real-Time Financial Risk Pipeline

**Research Question:** Can an ensemble of unsupervised (Isolation Forest) and supervised (XGBoost) anomaly detection models reliably flag financial market risk events, with full decision auditability suitable for regulated deployment?

## Finding
The supervised XGBoost model (AUC ~0.78) significantly outperforms the unsupervised Isolation Forest (AUC ~0.61) when labeled anomaly data is available. However, the ensemble of both captures a broader set of anomaly types: XGBoost catches pattern-based anomalies well, while Isolation Forest detects novel structural breaks not seen in training data. The FAISS audit log enables regulators to retrieve all past decisions similar to any flagged event — the core auditability requirement for AI in regulated finance.

## What This Project Does
- Fetches real OHLCV market data (yfinance) for 8 diverse assets (equities, commodities, rates)
- Engineers 13 financial risk features: volatility regime, momentum, drawdown, volume anomalies
- Trains Isolation Forest (unsupervised) + XGBoost (supervised) with time-series cross-validation (no data leakage)
- Scores latest 30 days per asset with an ensemble risk score (0-1)
- Logs every decision to a FAISS-backed append-only audit trail with similarity search
- Generates plain-English explanations for HIGH/CRITICAL alerts using Qwen2.5:7b (local)
- Produces a monitoring dashboard across all assets

## Key Design Decisions
**TimeSeriesSplit (not random CV):** Prevents data leakage — future data never used to predict past events. Critical for financial ML validity.

**FAISS audit log:** Every scoring event is stored as a feature vector. Regulators can ask "show me all past cases similar to this alert" — answered in milliseconds.

**Ensemble scoring:** Unsupervised + supervised combination catches both known patterns and novel anomalies, reducing the risk of model blind spots.

## Setup
```bash
pip install -r requirements.txt
python main.py
```

## Project Structure
```
p3-risk-pipeline/
├── ingestion/data_pipeline.py   # yfinance fetch + 13-feature engineering
├── scoring/risk_scorer.py       # IsolationForest + XGBoost + ensemble
├── audit/audit_log.py           # FAISS audit trail + monitoring dashboard
├── monitoring/explainer.py      # Qwen2.5 alert explanations
└── main.py                      # End-to-end pipeline
```
