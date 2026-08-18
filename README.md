# FinRisk-Audit: Real-Time Financial Risk Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/ML-XGBoost%20%7C%20IsolationForest-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Finance-Risk%20Detection-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Evaluation-ROC--AUC-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/VectorSearch-FAISS-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/LLM-Qwen2.5%3A7b-purple?style=for-the-badge">
</p>

<p align="center">
A financial risk detection pipeline combining supervised and unsupervised anomaly detection with time-series validation, FAISS-based auditability, and automated local LLM explanations.
</p>

---

## 📌 Overview

This project evaluates whether an ensemble of **Isolation Forest** and **XGBoost** can reliably detect financial market risk events while maintaining full decision auditability.

The pipeline uses **time-series cross-validation** to prevent future-data leakage and combines supervised detection of known risk patterns with unsupervised detection of novel market anomalies.

Every scoring decision is stored in a **FAISS-backed audit trail**, allowing historically similar risk events to be retrieved for investigation and regulatory review.

---

## ✨ Features

* Real OHLCV market data using **yfinance**
* Monitoring across **8 diverse financial assets**
* Engineering of **13 financial risk features**
* **Isolation Forest** for unsupervised anomaly detection
* **XGBoost** for supervised risk detection
* Time-series cross-validation using **TimeSeriesSplit**
* Ensemble risk scoring from **0–1**
* FAISS-based similarity search and audit logging
* Automated explanations using **Qwen2.5:7b** via Ollama
* Risk monitoring dashboard across all assets

---

## 📐 Evaluation Metrics

| Metric               | Purpose                             |
| -------------------- | ----------------------------------- |
| ROC-AUC              | Overall risk/anomaly discrimination |
| Ensemble Risk Score  | Combined model risk assessment      |
| Time-Series CV       | Prevents temporal data leakage      |
| Anomaly Coverage     | Measures detected risk patterns     |
| Similarity Retrieval | Finds similar historical events     |
| Audit Trail          | Records model decisions             |

---

## ⚙️ Tech Stack

| Technology          | Usage                                   |
| ------------------- | --------------------------------------- |
| Python              | Core development                        |
| yfinance            | Financial market data                   |
| Pandas / NumPy      | Data processing and feature engineering |
| Scikit-Learn        | Isolation Forest + TimeSeriesSplit      |
| XGBoost             | Supervised anomaly detection            |
| FAISS               | Similarity search and audit trail       |
| Matplotlib          | Visualization                           |
| Ollama + Qwen2.5:7b | Automated risk explanations             |

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Start Ollama *(Optional)*

```bash
ollama serve
ollama pull qwen2.5:7b
```

Required only for automated risk explanation generation.

### 3️⃣ Run the Pipeline

```bash
python main.py
```

---

## 🖥️ How It Works

1. OHLCV market data is automatically downloaded using **yfinance** for 8 assets.
2. The pipeline engineers **13 financial risk features** including volatility, momentum, drawdown, and volume anomalies.
3. **TimeSeriesSplit** creates leakage-free temporal training and evaluation splits.
4. **Isolation Forest** and **XGBoost** are trained for complementary anomaly detection.
5. Both predictions are combined into an ensemble risk score between **0 and 1**.
6. The latest **30 days** of each asset are scored for financial risk.
7. Every decision is stored in a **FAISS-backed audit trail** for similarity retrieval.
8. **Qwen2.5:7b** generates plain-English explanations for HIGH/CRITICAL alerts.

---

## 🔍 Key Finding

> The supervised **XGBoost model (AUC ~0.78)** significantly outperforms **Isolation Forest (AUC ~0.61)** when labeled anomaly data is available. However, Isolation Forest detects novel structural breaks that may be missed by supervised learning. Combining both models provides broader anomaly coverage, while the FAISS audit log enables rapid retrieval of historically similar financial risk decisions.

---

## 📁 Project Structure

```text
finrisk-audit/
├── ingestion/
│   └── data_pipeline.py           # yfinance + 13-feature engineering
├── scoring/
│   └── risk_scorer.py             # Isolation Forest + XGBoost
├── audit/
│   └── audit_log.py               # FAISS audit trail + retrieval
├── monitoring/
│   └── explainer.py               # Qwen2.5:7b alert explanations
├── reports/
│   ├── risk_results.json
│   └── plots/
├── models/
│   └── checkpoints/
├── main.py
└── requirements.txt
```

---

## 📄 Notes

* Market data is downloaded automatically using **yfinance**.
* TimeSeriesSplit is used to prevent temporal data leakage.
* The ensemble produces a risk score between **0 and 1**.
* FAISS enables similarity-based retrieval of historical decisions.
* Ollama is optional; the core pipeline runs without it.
* Model checkpoints and evaluation reports are stored locally.

---

## 👨‍💻 Author

**Vashishtha Verma**

* 🤖 Machine Learning & Generative AI
* 📈 Financial Machine Learning
* 🧠 NLP & Large Language Models
* 💻 Software Engineering & DSA
