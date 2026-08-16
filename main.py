"""
Main pipeline — Real-Time Financial Risk Pipeline.

Research Question: Can an ensemble of unsupervised (IsolationForest)
and supervised (XGBoost) models detect financial market anomalies
reliably, with full auditability and explainability for regulated deployment?

Steps:
1. Fetch real market data (yfinance) for 8 assets
2. Engineer 13 risk features (volatility, momentum, drawdown, volume)
3. Train IsolationForest + XGBoost risk models with time-series CV
4. Score latest 30 days for each asset
5. Log all decisions to FAISS-backed audit log
6. Generate Qwen2.5 explanations for HIGH/CRITICAL alerts
7. Build monitoring dashboard
"""
import os
import json
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ingestion.data_pipeline import (
    fetch_market_data, build_multi_asset_dataset, FEATURE_COLS
)
from scoring.risk_scorer import train_risk_models, score_latest_data
from audit.audit_log import AuditLog, generate_monitoring_dashboard
from monitoring.explainer import explain_risk_alert, generate_portfolio_risk_summary


def main():
    console.print(Panel(
        "[bold cyan]Real-Time Financial Risk Pipeline[/bold cyan]\n"
        "IsolationForest + XGBoost | FAISS Audit Log | Qwen2.5 Explanations",
        title="P3"
    ))

    # ── 1. Ingest market data ─────────────────────────────────────────────
    logger.info("Step 1: Fetching market data")
    market_data = fetch_market_data()
    df = build_multi_asset_dataset(market_data)
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/features_dataset.csv")
    logger.info(f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols")

    # ── 2. Train risk models ──────────────────────────────────────────────
    logger.info("Step 2: Training risk models")
    iso_model, xgb_model = train_risk_models(df, FEATURE_COLS)

    # ── 3. Score latest data ──────────────────────────────────────────────
    logger.info("Step 3: Scoring latest 30 days")
    risk_scores = score_latest_data(df, FEATURE_COLS, iso_model, xgb_model, n_latest=30)
    logger.info(f"Scored {len(risk_scores)} asset-days")

    # ── 4. Audit logging ──────────────────────────────────────────────────
    logger.info("Step 4: Logging to FAISS audit trail")
    audit = AuditLog()
    audit.log_batch(risk_scores)

    # ── 5. Risk alert table ───────────────────────────────────────────────
    high_alerts = [s for s in risk_scores if s.risk_level in ("HIGH", "CRITICAL")]
    logger.info(f"HIGH/CRITICAL alerts: {len(high_alerts)}")

    table = Table(title="Risk Alerts", box=box.ROUNDED)
    for col in ["Ticker", "Date", "Score", "Level", "Top Driver"]:
        table.add_column(col)
    for s in sorted(high_alerts, key=lambda x: x.risk_score, reverse=True)[:15]:
        level_color = {"HIGH": "yellow", "CRITICAL": "red"}.get(s.risk_level, "white")
        table.add_row(
            s.ticker, s.date[:10],
            f"{s.risk_score:.3f}",
            f"[{level_color}]{s.risk_level}[/{level_color}]",
            s.top_drivers[0][0] if s.top_drivers else "—",
        )
    console.print(table)

    # ── 6. Qwen2.5 explanations for top alerts ────────────────────────────
    logger.info("Step 5: Generating explanations with Qwen2.5:7b")
    explanations = {}
    top_alerts = sorted(high_alerts, key=lambda x: x.risk_score, reverse=True)[:5]

    for alert in top_alerts:
        similar = audit.find_similar_events(alert.feature_vector, top_k=3)
        explanation = explain_risk_alert(alert, similar)
        explanations[f"{alert.ticker}_{alert.date[:10]}"] = explanation
        logger.info(f"Explained: {alert.ticker} ({alert.risk_level})")

    with open("reports/alert_explanations.json", "w") as f:
        json.dump(explanations, f, indent=2)

    # Portfolio summary
    portfolio_summary = generate_portfolio_risk_summary(risk_scores)
    console.print(Panel(portfolio_summary[:800], title="Portfolio Risk Summary"))

    # ── 7. Monitoring dashboard ───────────────────────────────────────────
    logger.info("Step 6: Generating monitoring dashboard")
    generate_monitoring_dashboard(risk_scores, audit)

    # ── Summary ───────────────────────────────────────────────────────────
    avg_risk = sum(s.risk_score for s in risk_scores) / len(risk_scores)
    critical_count = sum(1 for s in risk_scores if s.risk_level == "CRITICAL")

    console.print(Panel(
        f"[green]Pipeline complete.[/green]\n\n"
        f"Assets monitored: {len(market_data)}\n"
        f"Avg risk score: {avg_risk:.3f}\n"
        f"HIGH/CRITICAL alerts: {len(high_alerts)} | CRITICAL: {critical_count}\n"
        f"Audit log entries: {len(risk_scores)}\n\n"
        f"Reports saved to: reports/\n"
        f"  - features_dataset.csv\n"
        f"  - alert_explanations.json\n"
        f"  - portfolio_risk_summary.md\n"
        f"  - plots/risk_dashboard.png\n"
        f"Audit trail: audit/faiss_audit.index",
        title="Done"
    ))


if __name__ == "__main__":
    main()
