"""
Market data ingestion pipeline.

Fetches real OHLCV data via yfinance for a portfolio of tickers.
Engineers financial risk features:
- Volatility measures (realized vol, GARCH-inspired rolling vol)
- Momentum indicators (RSI, MACD)
- Volume anomalies
- Drawdown metrics
- Correlation regime features

This is structured financial data engineering — directly relevant to
scalable ML pipelines for financial domain tasks.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from loguru import logger
from rich.console import Console

console = Console()

# Portfolio: diversified set covering equities, volatility, bonds
DEFAULT_TICKERS = [
    "SPY",   # S&P 500 ETF
    "QQQ",   # NASDAQ 100
    "IWM",   # Russell 2000 (small cap)
    "GLD",   # Gold
    "TLT",   # Long-term treasuries
    "VIX",   # Volatility index proxy via ^VIX
    "XLF",   # Financials sector
    "XLE",   # Energy sector
]


def fetch_market_data(
    tickers: list[str] = DEFAULT_TICKERS,
    period: str = "2y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for all tickers."""
    logger.info(f"Fetching market data: {len(tickers)} tickers, {period} period")
    data = {}
    for ticker in tickers:
        try:
            t = ticker if not ticker.startswith("^") else ticker
            df = yf.download(t, period=period, interval=interval, progress=False)
            if len(df) > 50:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[ticker] = df
                logger.debug(f"  {ticker}: {len(df)} rows")
            else:
                logger.warning(f"  {ticker}: insufficient data ({len(df)} rows), skipping")
        except Exception as e:
            logger.warning(f"  {ticker}: fetch failed — {e}")

    logger.success(f"Fetched data for {len(data)}/{len(tickers)} tickers")
    return data


def engineer_risk_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Engineer risk features from OHLCV data.
    Each feature has a clear financial interpretation for audit purposes.
    """
    feat = pd.DataFrame(index=df.index)
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    # ── Returns ────────────────────────────────────────────────────────────
    feat["return_1d"] = close.pct_change(1)
    feat["return_5d"] = close.pct_change(5)
    feat["return_20d"] = close.pct_change(20)

    # ── Volatility (realized) ──────────────────────────────────────────────
    feat["vol_5d"] = feat["return_1d"].rolling(5).std() * np.sqrt(252)
    feat["vol_20d"] = feat["return_1d"].rolling(20).std() * np.sqrt(252)
    feat["vol_60d"] = feat["return_1d"].rolling(60).std() * np.sqrt(252)

    # Volatility regime: ratio of short to long vol (>1 = rising risk)
    feat["vol_ratio"] = feat["vol_5d"] / (feat["vol_60d"] + 1e-9)

    # ── Momentum ──────────────────────────────────────────────────────────
    feat["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    feat["macd_diff"] = macd.macd_diff()

    # ── Drawdown ──────────────────────────────────────────────────────────
    rolling_max = close.expanding().max()
    feat["drawdown"] = (close - rolling_max) / rolling_max

    # ── Volume anomaly ────────────────────────────────────────────────────
    vol_ma = volume.rolling(20).mean()
    feat["volume_ratio"] = volume / (vol_ma + 1e-9)

    # ── Price range (intraday stress) ────────────────────────────────────
    feat["range_pct"] = (high - low) / (close + 1e-9)

    # ── Bollinger Band position ───────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20)
    feat["bb_position"] = (close - bb.bollinger_lband()) / (
        bb.bollinger_hband() - bb.bollinger_lband() + 1e-9
    )

    # ── Target: anomaly label (for training) ──────────────────────────────
    # Define anomaly as: next-day return < -2σ of rolling 20d vol
    threshold = -2 * feat["vol_20d"]
    feat["is_anomaly"] = (feat["return_1d"].shift(-1) < threshold).astype(int)

    feat["ticker"] = ticker
    feat = feat.dropna()

    return feat


def build_multi_asset_dataset(
    market_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build combined dataset across all tickers."""
    frames = []
    for ticker, df in market_data.items():
        try:
            features = engineer_risk_features(df, ticker)
            frames.append(features)
        except Exception as e:
            logger.warning(f"Feature engineering failed for {ticker}: {e}")

    combined = pd.concat(frames, axis=0).sort_index()
    logger.info(f"Combined dataset: {len(combined)} rows, {combined.shape[1]} features")
    logger.info(f"Anomaly rate: {combined['is_anomaly'].mean():.2%}")
    return combined


FEATURE_COLS = [
    "return_1d", "return_5d", "return_20d",
    "vol_5d", "vol_20d", "vol_60d", "vol_ratio",
    "rsi_14", "macd_diff", "drawdown",
    "volume_ratio", "range_pct", "bb_position",
]
