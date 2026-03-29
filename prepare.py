"""
Fixed infrastructure for autotrading experiments.
Downloads market data from HyperLiquid, computes features, runs backtests, evaluates strategies.

Usage:
    python prepare.py                      # download all data (default assets, 365 days)
    python prepare.py --lookback-days 180  # download 180 days of data
    python prepare.py --assets BTC ETH SOL # download specific assets

Data is stored in ~/.cache/autotrading/.
DO NOT MODIFY THIS FILE — it is the fixed evaluation harness.
"""

import os
import sys
import time
import math
import argparse
import importlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

BACKTEST_BUDGET = 300           # strategy optimization time budget in seconds (5 min)
LOOKBACK_DAYS = 365             # how much historical data to fetch
EVAL_DAYS = 90                  # last N days reserved for out-of-sample evaluation
TRAIN_DAYS = LOOKBACK_DAYS - EVAL_DAYS

CANDLE_INTERVAL = "1h"          # base candle resolution
CANDLE_INTERVAL_MS = 3600_000   # 1 hour in milliseconds

# Default tradeable assets — agent cannot change this list
ASSETS = ["BTC", "ETH"]

INITIAL_CAPITAL = 100_000.0     # USD starting capital for backtest
MAX_LEVERAGE = 3.0              # hard cap on leverage
COMMISSION_BPS = 3.5            # round-trip commission in basis points (HyperLiquid taker)
SLIPPAGE_BPS = 2.0              # estimated slippage per trade
FUNDING_RATE_INTERVAL_H = 8    # HyperLiquid funding rate interval

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autotrading")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# ---------------------------------------------------------------------------
# Data download from HyperLiquid
# ---------------------------------------------------------------------------

def _get_info_client():
    """Lazy import to avoid requiring SDK at module level during backtesting."""
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    return Info(constants.MAINNET_API_URL, skip_ws=True)


def download_candles(asset: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Download OHLCV candles from HyperLiquid API with pagination."""
    info = _get_info_client()

    end_time = int(time.time() * 1000)
    start_time = end_time - (lookback_days * 24 * 3600 * 1000)

    all_candles = []
    cursor = start_time

    print(f"  [{asset}] Downloading {CANDLE_INTERVAL} candles ({lookback_days} days)...")

    while cursor < end_time:
        chunk_end = min(cursor + 500 * CANDLE_INTERVAL_MS, end_time)
        try:
            candles = info.candles_snapshot(asset, CANDLE_INTERVAL, cursor, chunk_end)
        except Exception as e:
            print(f"  [{asset}] API error at {cursor}: {e}, retrying...")
            time.sleep(2)
            try:
                candles = info.candles_snapshot(asset, CANDLE_INTERVAL, cursor, chunk_end)
            except Exception as e2:
                print(f"  [{asset}] Failed after retry: {e2}")
                break

        if not candles:
            cursor = chunk_end
            continue

        all_candles.extend(candles)
        last_t = max(int(c["t"]) for c in candles)
        cursor = last_t + CANDLE_INTERVAL_MS

        # Rate limit
        time.sleep(0.1)

    if not all_candles:
        print(f"  [{asset}] WARNING: No candle data retrieved!")
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df["timestamp"] = pd.to_datetime(df["t"].astype(int), unit="ms", utc=True)
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = df[col].astype(float)
    df["n"] = df["n"].astype(int)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "num_trades"})
    df = df[["timestamp", "open", "high", "low", "close", "volume", "num_trades"]]
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    print(f"  [{asset}] Got {len(df)} candles ({df['timestamp'].min()} to {df['timestamp'].max()})")
    return df


def download_funding(asset: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Download funding rate history from HyperLiquid API."""
    info = _get_info_client()

    end_time = int(time.time() * 1000)
    start_time = end_time - (lookback_days * 24 * 3600 * 1000)

    print(f"  [{asset}] Downloading funding rate history...")

    all_funding = []
    cursor = start_time

    while cursor < end_time:
        try:
            funding = info.funding_history(asset, cursor, end_time)
        except Exception as e:
            print(f"  [{asset}] Funding API error: {e}, retrying...")
            time.sleep(2)
            try:
                funding = info.funding_history(asset, cursor, end_time)
            except Exception:
                break

        if not funding:
            break

        all_funding.extend(funding)
        last_t = max(int(f["time"]) for f in funding)
        cursor = last_t + 1
        time.sleep(0.1)

        # HyperLiquid returns all at once typically
        if len(funding) < 500:
            break

    if not all_funding:
        print(f"  [{asset}] WARNING: No funding data retrieved!")
        return pd.DataFrame()

    df = pd.DataFrame(all_funding)
    df["timestamp"] = pd.to_datetime(df["time"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df["premium"] = df["premium"].astype(float)
    df = df[["timestamp", "funding_rate", "premium"]]
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    print(f"  [{asset}] Got {len(df)} funding rate entries")
    return df


def download_all_data(assets: list[str] = None, lookback_days: int = LOOKBACK_DAYS,
                      max_age_hours: float = 2.0):
    """Download and cache all market data for given assets.

    Args:
        max_age_hours: Skip download if cached data is newer than this (default 2h).
                       Set to 0 to force refresh.
    """
    if assets is None:
        assets = ASSETS
    os.makedirs(DATA_DIR, exist_ok=True)

    for asset in assets:
        candle_path = os.path.join(DATA_DIR, f"{asset}_candles.parquet")
        funding_path = os.path.join(DATA_DIR, f"{asset}_funding.parquet")

        # Check if data exists and is recent enough
        need_download = True
        if max_age_hours > 0 and os.path.exists(candle_path):
            existing = pd.read_parquet(candle_path)
            if len(existing) > 0:
                last_ts = existing["timestamp"].max()
                hours_old = (pd.Timestamp.now(tz="UTC") - last_ts).total_seconds() / 3600
                if hours_old < max_age_hours:
                    print(f"  [{asset}] Candles already up-to-date (last: {last_ts})")
                    need_download = False
                else:
                    print(f"  [{asset}] Candles are {hours_old:.0f}h old, refreshing...")

        if need_download:
            candles_df = download_candles(asset, lookback_days)
            if len(candles_df) > 0:
                candles_df.to_parquet(candle_path, index=False)

            funding_df = download_funding(asset, lookback_days)
            if len(funding_df) > 0:
                funding_df.to_parquet(funding_path, index=False)


def load_market_data(asset: str) -> pd.DataFrame:
    """Load cached market data and merge candles with funding rates."""
    candle_path = os.path.join(DATA_DIR, f"{asset}_candles.parquet")
    funding_path = os.path.join(DATA_DIR, f"{asset}_funding.parquet")

    if not os.path.exists(candle_path):
        raise FileNotFoundError(
            f"No data for {asset}. Run: uv run prepare.py"
        )

    df = pd.read_parquet(candle_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Merge funding rates (forward-fill to candle frequency)
    if os.path.exists(funding_path):
        funding = pd.read_parquet(funding_path)
        funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True)
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            funding[["timestamp", "funding_rate", "premium"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        df["funding_rate"] = df["funding_rate"].fillna(0.0)
        df["premium"] = df["premium"].fillna(0.0)
    else:
        df["funding_rate"] = 0.0
        df["premium"] = 0.0

    return df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Feature Engineering (fixed base features available to the strategy)
# ---------------------------------------------------------------------------

def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a comprehensive set of features from OHLCV + funding data.
    The strategy can use any subset of these. All features are backward-looking
    (no future information leakage).

    Returns a copy with feature columns added.
    """
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # --- Returns at various horizons ---
    for period in [1, 4, 8, 24, 48, 168]:  # 1h, 4h, 8h, 1d, 2d, 7d
        df[f"return_{period}h"] = c.pct_change(period)

    # --- Log returns (for statistical properties) ---
    df["log_return_1h"] = np.log(c / c.shift(1))

    # --- Volatility (rolling std of log returns) ---
    log_ret = df["log_return_1h"]
    for window in [24, 48, 168, 720]:  # 1d, 2d, 7d, 30d
        df[f"volatility_{window}h"] = log_ret.rolling(window, min_periods=max(1, window // 2)).std()

    # --- Realized volatility (Garman-Klass, more efficient estimator) ---
    log_hl = np.log(h / l)
    log_co = np.log(c / df["open"])
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    for window in [24, 168]:
        df[f"gk_volatility_{window}h"] = gk.rolling(window, min_periods=max(1, window // 2)).mean().apply(np.sqrt)

    # --- Simple Moving Averages ---
    for period in [8, 20, 50, 100, 200]:
        df[f"sma_{period}"] = c.rolling(period, min_periods=1).mean()

    # --- Exponential Moving Averages ---
    for period in [8, 20, 50]:
        df[f"ema_{period}"] = c.ewm(span=period, min_periods=1).mean()

    # --- SMA cross signals (price relative to SMA) ---
    for period in [20, 50, 200]:
        sma = df[f"sma_{period}"]
        df[f"price_vs_sma_{period}"] = (c - sma) / sma

    # --- MACD ---
    ema_12 = c.ewm(span=12, min_periods=1).mean()
    ema_26 = c.ewm(span=26, min_periods=1).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, min_periods=1).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # --- RSI (14 period) ---
    for period in [14, 28]:
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands (20 period, 2 std) ---
    sma_20 = df["sma_20"]
    std_20 = c.rolling(20, min_periods=1).std()
    df["bb_upper"] = sma_20 + 2 * std_20
    df["bb_lower"] = sma_20 - 2 * std_20
    bb_width = df["bb_upper"] - df["bb_lower"]
    df["bb_width"] = bb_width / sma_20  # normalized
    df["bb_position"] = (c - df["bb_lower"]) / bb_width.replace(0, np.nan)  # 0=lower band, 1=upper

    # --- ATR (Average True Range) ---
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    for period in [14, 48]:
        df[f"atr_{period}"] = tr.rolling(period, min_periods=1).mean()
        df[f"atr_{period}_pct"] = df[f"atr_{period}"] / c  # normalized

    # --- Volume features ---
    for window in [24, 168]:
        vol_ma = v.rolling(window, min_periods=1).mean()
        df[f"relative_volume_{window}h"] = v / vol_ma.replace(0, np.nan)
    df["volume_momentum"] = v.rolling(24, min_periods=1).mean() / v.rolling(168, min_periods=1).mean().replace(0, np.nan)

    # --- On-Balance Volume (simplified) ---
    obv = (np.sign(c.diff()) * v).cumsum()
    df["obv_slope_24h"] = obv.diff(24) / (obv.rolling(24).std().replace(0, np.nan))

    # --- Funding rate features ---
    fr = df["funding_rate"]
    df["funding_rate_8h"] = fr  # raw (already per 8h)
    df["funding_cumulative_24h"] = fr.rolling(3, min_periods=1).sum()   # ~3 periods of 8h
    df["funding_cumulative_7d"] = fr.rolling(21, min_periods=1).sum()  # ~21 periods of 8h
    fr_mean = fr.rolling(21, min_periods=1).mean()
    fr_std = fr.rolling(21, min_periods=1).std().replace(0, np.nan)
    df["funding_zscore"] = (fr - fr_mean) / fr_std

    # --- Price momentum (Rate of Change) ---
    for period in [24, 168, 720]:
        df[f"roc_{period}h"] = (c - c.shift(period)) / c.shift(period)

    # --- Donchian Channel position ---
    for period in [48, 168]:
        highest = h.rolling(period, min_periods=1).max()
        lowest = l.rolling(period, min_periods=1).min()
        channel_width = highest - lowest
        df[f"donchian_position_{period}h"] = (c - lowest) / channel_width.replace(0, np.nan)

    # --- Mean reversion z-score ---
    for lookback in [48, 168]:
        rolling_mean = c.rolling(lookback, min_periods=1).mean()
        rolling_std = c.rolling(lookback, min_periods=1).std().replace(0, np.nan)
        df[f"zscore_{lookback}h"] = (c - rolling_mean) / rolling_std

    # --- Hour of day (cyclical encoding) ---
    if "timestamp" in df.columns:
        hour = df["timestamp"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        # Day of week
        dow = df["timestamp"].dt.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return df


# ---------------------------------------------------------------------------
# Backtesting Engine (fixed, do not modify)
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Immutable result of a backtest run."""
    equity_curve: pd.Series         # timestamp -> portfolio value
    trades: pd.DataFrame            # log of all trades
    total_return: float             # (final - initial) / initial
    annualized_return: float
    sharpe_ratio: float             # annualized, from hourly returns
    sortino_ratio: float
    max_drawdown: float             # as positive fraction (0.15 = 15%)
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_duration_hours: float
    exposure_pct: float             # % of time with open position


class BacktestEngine:
    """
    Vectorized backtester. Fixed, not modifiable by agent.

    Execution model:
    - Signals generated at bar close
    - Trades execute at NEXT bar open (one-bar delay, no look-ahead)
    - Commissions and slippage applied to each trade
    - Funding costs deducted every FUNDING_RATE_INTERVAL_H hours
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        max_leverage: float = MAX_LEVERAGE,
        commission_bps: float = COMMISSION_BPS,
        slippage_bps: float = SLIPPAGE_BPS,
    ):
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.commission_rate = commission_bps / 10_000
        self.slippage_rate = slippage_bps / 10_000

    def run(self, prices_df: pd.DataFrame, signals_df: pd.DataFrame,
            funding_df: pd.DataFrame = None) -> BacktestResult:
        """
        Run backtest.

        Args:
            prices_df: DataFrame indexed by timestamp, columns = asset names, values = close prices.
            signals_df: DataFrame indexed by timestamp, columns = asset names,
                        values in [-1, +1] (desired position as fraction of capital).
            funding_df: DataFrame indexed by timestamp, columns = asset names,
                        values = funding rate per period.

        Returns:
            BacktestResult with all metrics.
        """
        assets = signals_df.columns.tolist()
        timestamps = signals_df.index

        # Clip signals to [-1, +1] range
        signals = signals_df.clip(-1.0, 1.0)

        # One-bar delay: shift signals forward (signal at t → trade at t+1)
        signals = signals.shift(1).fillna(0.0)

        equity = np.full(len(timestamps), self.initial_capital, dtype=np.float64)
        position_value = {a: np.zeros(len(timestamps)) for a in assets}
        position_units = {a: 0.0 for a in assets}

        trades_log = []
        current_equity = self.initial_capital

        for i in range(1, len(timestamps)):
            ts = timestamps[i]
            prev_ts = timestamps[i - 1]

            # Update portfolio value from price changes
            pnl = 0.0
            for asset in assets:
                if position_units[asset] != 0:
                    price_now = prices_df.loc[ts, asset] if ts in prices_df.index else prices_df.iloc[min(i, len(prices_df) - 1)][asset]
                    price_prev = prices_df.loc[prev_ts, asset] if prev_ts in prices_df.index else prices_df.iloc[min(i - 1, len(prices_df) - 1)][asset]
                    if not (np.isnan(price_now) or np.isnan(price_prev) or price_prev == 0):
                        pnl += position_units[asset] * (price_now - price_prev)

            current_equity += pnl

            # Deduct funding costs
            if funding_df is not None:
                for asset in assets:
                    if position_units[asset] != 0 and ts in funding_df.index:
                        fr = funding_df.loc[ts, asset] if asset in funding_df.columns else 0.0
                        if not np.isnan(fr):
                            notional = abs(position_units[asset] * prices_df.loc[ts, asset])
                            # Long pays positive funding, short receives
                            funding_cost = notional * fr * np.sign(position_units[asset])
                            current_equity -= funding_cost

            # Rebalance positions based on signals
            for asset in assets:
                target_signal = signals.loc[ts, asset] if ts in signals.index else 0.0
                if np.isnan(target_signal):
                    target_signal = 0.0

                price = prices_df.loc[ts, asset] if ts in prices_df.index else np.nan
                if np.isnan(price) or price == 0:
                    continue

                # Target position in units
                target_notional = target_signal * current_equity * self.max_leverage
                target_units = target_notional / price

                # Trade delta
                delta_units = target_units - position_units[asset]

                if abs(delta_units) * price > current_equity * 0.001:  # min trade size
                    # Apply slippage
                    trade_price = price * (1 + self.slippage_rate * np.sign(delta_units))
                    trade_notional = abs(delta_units) * trade_price
                    commission = trade_notional * self.commission_rate

                    current_equity -= commission

                    trades_log.append({
                        "timestamp": ts,
                        "asset": asset,
                        "side": "buy" if delta_units > 0 else "sell",
                        "units": abs(delta_units),
                        "price": trade_price,
                        "notional": trade_notional,
                        "commission": commission,
                    })

                    position_units[asset] = target_units

            # Prevent negative equity
            current_equity = max(current_equity, 0.0)
            equity[i] = current_equity

        # --- Compute metrics ---
        equity_series = pd.Series(equity, index=timestamps)
        trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame(
            columns=["timestamp", "asset", "side", "units", "price", "notional", "commission"]
        )

        total_return = (equity[-1] - self.initial_capital) / self.initial_capital

        # Hourly returns
        hourly_returns = equity_series.pct_change().dropna()
        hours_per_year = 8760

        if len(hourly_returns) > 1 and hourly_returns.std() > 0:
            sharpe_ratio = (hourly_returns.mean() / hourly_returns.std()) * np.sqrt(hours_per_year)
        else:
            sharpe_ratio = 0.0

        # Sortino (downside deviation)
        downside = hourly_returns[hourly_returns < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino_ratio = (hourly_returns.mean() / downside.std()) * np.sqrt(hours_per_year)
        else:
            sortino_ratio = 0.0

        # Max drawdown
        cummax = equity_series.cummax()
        drawdown = (cummax - equity_series) / cummax.replace(0, np.nan)
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0
        if np.isnan(max_drawdown):
            max_drawdown = 0.0

        # Annualized return
        n_hours = len(timestamps)
        if n_hours > 0 and equity[0] > 0:
            annualized_return = (equity[-1] / equity[0]) ** (hours_per_year / max(n_hours, 1)) - 1
        else:
            annualized_return = 0.0

        # Calmar
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

        # Win rate and profit factor
        if len(trades_df) > 1:
            # Group trades into round-trips (simplified: consecutive buy/sell pairs per asset)
            num_trades = len(trades_df)
            # Approximate win rate from equity changes between trades
            trade_returns = []
            for asset in assets:
                asset_trades = trades_df[trades_df["asset"] == asset].sort_values("timestamp")
                if len(asset_trades) >= 2:
                    for j in range(1, len(asset_trades)):
                        t0_idx = timestamps.get_loc(asset_trades.iloc[j - 1]["timestamp"])
                        t1_idx = timestamps.get_loc(asset_trades.iloc[j]["timestamp"])
                        if t1_idx > t0_idx:
                            tr = equity[t1_idx] - equity[t0_idx]
                            trade_returns.append(tr)

            if trade_returns:
                wins = sum(1 for r in trade_returns if r > 0)
                win_rate = wins / len(trade_returns)
                gross_profit = sum(r for r in trade_returns if r > 0)
                gross_loss = abs(sum(r for r in trade_returns if r < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            else:
                win_rate = 0.0
                profit_factor = 0.0
        else:
            num_trades = len(trades_df)
            win_rate = 0.0
            profit_factor = 0.0

        # Average trade duration
        if len(trades_df) >= 2:
            trade_times = pd.to_datetime(trades_df["timestamp"])
            avg_duration = trade_times.diff().dropna().mean()
            avg_trade_duration_hours = avg_duration.total_seconds() / 3600 if pd.notna(avg_duration) else 0.0
        else:
            avg_trade_duration_hours = 0.0

        # Exposure: % of time with any open position
        signals_abs = signals.abs()
        exposure_mask = signals_abs.sum(axis=1) > 0.01
        exposure_pct = exposure_mask.mean() * 100

        return BacktestResult(
            equity_curve=equity_series,
            trades=trades_df,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades,
            avg_trade_duration_hours=avg_trade_duration_hours,
            exposure_pct=exposure_pct,
        )


# ---------------------------------------------------------------------------
# Strategy Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_strategy(strategy_module, split: str = "val") -> dict:
    """
    Fixed evaluation harness. DO NOT CHANGE.

    Loads data, computes features, calls strategy.generate_signals(),
    runs backtest on the appropriate split, returns metrics dict.

    The composite score (higher is better):
        If return > 0: score = sharpe_ratio * (1 - max_drawdown)
        If return <= 0: score = -abs(sharpe_ratio) * (1 + max_drawdown)

    This penalizes:
    - Low risk-adjusted returns (low Sharpe)
    - Large drawdowns (high max_drawdown)
    - Negative returns (score is always negative)

    Minimum trade count: strategies with < 10 trades get score = 0 (insufficient signal).
    """
    # Load data for all assets
    features = {}
    prices = {}
    funding = {}

    for asset in ASSETS:
        df = load_market_data(asset)
        df_feat = compute_base_features(df)

        # Temporal split
        n_total = len(df_feat)
        hours_per_day = 24
        eval_hours = EVAL_DAYS * hours_per_day
        train_hours = n_total - eval_hours

        if train_hours < hours_per_day * 30:
            raise ValueError(f"Not enough data for {asset}: {n_total} candles, need at least {eval_hours + hours_per_day * 30}")

        if split == "val":
            df_split = df_feat.iloc[train_hours:].copy()
        elif split == "train":
            df_split = df_feat.iloc[:train_hours].copy()
        elif split == "full":
            df_split = df_feat.copy()
        else:
            raise ValueError(f"Unknown split: {split}")

        df_split = df_split.set_index("timestamp")
        features[asset] = df_split
        prices[asset] = df_split["close"]
        funding[asset] = df_split["funding_rate"]

    # Build prices and funding DataFrames
    prices_df = pd.DataFrame(prices)
    funding_df = pd.DataFrame(funding)

    # Call strategy
    signals_df = strategy_module.generate_signals(features)

    # Ensure signals are aligned with prices
    common_idx = prices_df.index.intersection(signals_df.index)
    if len(common_idx) == 0:
        return {
            "composite_score": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "num_trades": 0,
            "avg_trade_duration_hours": 0.0,
            "exposure_pct": 0.0,
        }

    prices_df = prices_df.loc[common_idx]
    signals_df = signals_df.loc[common_idx]
    funding_df = funding_df.loc[common_idx] if len(funding_df) > 0 else None

    # Run backtest
    engine = BacktestEngine()
    result = engine.run(prices_df, signals_df, funding_df)

    # Composite score
    if result.num_trades < 10:
        composite_score = 0.0  # insufficient trades
    elif result.total_return <= 0:
        # Negative return strategies always get negative score
        composite_score = -abs(result.sharpe_ratio) * (1.0 + result.max_drawdown)
    else:
        # Positive return: reward sharpe, penalize drawdown
        composite_score = result.sharpe_ratio * (1.0 - result.max_drawdown)

    return {
        "composite_score": composite_score,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "max_drawdown": result.max_drawdown,
        "calmar_ratio": result.calmar_ratio,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "num_trades": result.num_trades,
        "avg_trade_duration_hours": result.avg_trade_duration_hours,
        "exposure_pct": result.exposure_pct,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare market data for autotrading")
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS,
                        help=f"Days of historical data to download (default: {LOOKBACK_DAYS})")
    parser.add_argument("--assets", nargs="+", default=ASSETS,
                        help=f"Assets to download (default: {' '.join(ASSETS)})")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Assets: {args.assets}")
    print(f"Lookback: {args.lookback_days} days")
    print()

    download_all_data(assets=args.assets, lookback_days=args.lookback_days)

    print()
    print("Verifying loaded data...")
    for asset in args.assets:
        df = load_market_data(asset)
        df_feat = compute_base_features(df)
        print(f"  [{asset}] {len(df)} candles, {len(df_feat.columns)} features")
        print(f"    Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"    Columns: {', '.join(df_feat.columns[:10])}...")

    print()
    print("Done! Ready to run strategies.")
