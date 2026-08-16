"""
Qwen2.5:7b-powered risk explanation engine.

For every HIGH/CRITICAL alert, generates:
1. Plain-English explanation of why this alert was triggered
2. Historical context (similar past events from FAISS)
3. Recommended action for the risk analyst
"""
from __future__ import annotations
import json
import requests
from loguru import logger
from scoring.risk_scorer import RiskScore

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"


def _call_ollama(prompt: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 512},
            },
            timeout=90,
        )
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama failed: {e}")
        return f"Explanation unavailable: {e}"


def explain_risk_alert(
    score: RiskScore,
    similar_events: list[dict],
) -> str:
    """Generate explanation for a HIGH/CRITICAL risk alert."""
    drivers_str = "\n".join(
        f"  - {feat}: importance={imp:.4f}"
        for feat, imp in score.top_drivers
    )

    similar_str = ""
    if similar_events:
        similar_str = "Similar historical events:\n" + "\n".join(
            f"  - {e['ticker']} on {e['date']}: risk={e['risk_score']:.3f} ({e['risk_level']})"
            for e in similar_events[:3]
        )

    prompt = f"""You are a financial risk analyst AI. Explain the following risk alert to a compliance officer.

Alert Details:
- Asset: {score.ticker}
- Date: {score.date}
- Risk Score: {score.risk_score:.3f} ({score.risk_level})
- Anomaly Detected: {score.anomaly_detected}

Key risk drivers:
{drivers_str}

{similar_str}

Write a concise explanation (3-4 sentences) that:
1. States what the alert means in plain English
2. Identifies the primary driver of the elevated risk
3. Provides context from similar historical events if available
4. Recommends a concrete next action for the analyst

Be specific and avoid jargon where possible."""

    return _call_ollama(prompt)


def generate_portfolio_risk_summary(
    risk_scores: list[RiskScore],
) -> str:
    """Generate a portfolio-level risk summary for all assets."""
    critical = [s for s in risk_scores if s.risk_level == "CRITICAL"]
    high = [s for s in risk_scores if s.risk_level == "HIGH"]
    avg_score = sum(s.risk_score for s in risk_scores) / len(risk_scores)

    alert_summary = json.dumps([
        {"ticker": s.ticker, "score": s.risk_score, "level": s.risk_level,
         "top_driver": s.top_drivers[0][0] if s.top_drivers else "unknown"}
        for s in sorted(risk_scores, key=lambda x: x.risk_score, reverse=True)[:10]
    ], indent=2)

    prompt = f"""You are a portfolio risk manager. Summarize the current risk state of the portfolio.

Portfolio statistics:
- Total assets monitored: {len(set(s.ticker for s in risk_scores))}
- CRITICAL alerts: {len(critical)}
- HIGH alerts: {len(high)}
- Average portfolio risk score: {avg_score:.3f}

Top risk alerts:
{alert_summary}

Write a portfolio risk briefing (4-5 sentences) suitable for a senior risk officer:
1. Overall portfolio risk assessment
2. Most concerning positions and why
3. Any systemic/correlated risks across assets
4. Recommended portfolio actions"""

    summary = _call_ollama(prompt)

    with open("reports/portfolio_risk_summary.md", "w") as f:
        f.write("# Portfolio Risk Summary\n\n")
        f.write(summary)

    return summary
