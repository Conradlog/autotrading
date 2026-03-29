"""
Autotrading strategy. Single file, agent-modifiable.
This is the ONLY file the autonomous agent edits.

Usage: uv run strategy.py
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tunable Parameters (agent modifies these)
# ---------------------------------------------------------------------------

# Trend following
FAST_MA = 20                    # fast moving average period (hours)
SLOW_MA = 50                    # slow moving average period (hours)
TREND_FILTER_PERIOD = 200       # only trade in direction of long-term trend
USE_TREND_FILTER = False        # disable trend filter

# Mean reversion filter
RSI_PERIOD = 14                 # RSI lookback
RSI_OVERBOUGHT = 75             # RSI threshold for overbought (wider band)
RSI_OVERSOLD = 25               # RSI threshold for oversold (wider band)

# Position sizing
POSITION_SIZE = 0.3             # base position size (fraction of capital per signal)
VOL_SCALING = True              # scale position by inverse volatility
VOL_LOOKBACK = 168              # hours for volatility calculation (7 days)
VOL_TARGET = 0.15               # annualized volatility target

# Risk management
STOP_LOSS_ATR_MULT = 2.5        # stop loss as multiple of ATR
TAKE_PROFIT_ATR_MULT = 4.0      # take profit as multiple of ATR
MAX_POSITION = 0.8              # max absolute position per asset

# Funding rate signal
USE_FUNDING_SIGNAL = True       # use funding rate as contrarian signal
FUNDING_THRESHOLD = 2.0         # z-score threshold for funding-based signal

# Multi-asset
CORRELATION_FILTER = False      # reduce position when assets are highly correlated
CORRELATION_LOOKBACK = 168      # hours


# ---------------------------------------------------------------------------
# Strategy Logic
# ---------------------------------------------------------------------------

def compute_signal(df: pd.DataFrame) -> pd.Series:
    """
    Compute trading signal for a single asset.
    Returns a Series of values in [-1, +1].
    """
    c = df["close"]
    signal = pd.Series(0.0, index=df.index)

    # --- Trend signal: MA crossover ---
    fast_ma = c.rolling(FAST_MA, min_periods=1).mean()
    slow_ma = c.rolling(SLOW_MA, min_periods=1).mean()

    trend_signal = pd.Series(0.0, index=df.index)
    trend_signal[fast_ma > slow_ma] = 1.0
    trend_signal[fast_ma < slow_ma] = -1.0

    # --- Long-term trend filter (optional) ---
    if USE_TREND_FILTER:
        trend_ma = c.rolling(TREND_FILTER_PERIOD, min_periods=1).mean()
        bull_market = c > trend_ma
        bear_market = c < trend_ma
        trend_signal[(trend_signal > 0) & bear_market] = 0.0
        trend_signal[(trend_signal < 0) & bull_market] = 0.0

    signal = trend_signal

    # --- RSI filter: reduce position at extremes ---
    if f"rsi_{RSI_PERIOD}" in df.columns:
        rsi = df[f"rsi_{RSI_PERIOD}"]
    else:
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD).mean()
        avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

    # Dampen signal when RSI is extreme (contrarian dampening)
    rsi_dampener = pd.Series(1.0, index=df.index)
    rsi_dampener[rsi > RSI_OVERBOUGHT] = 0.3   # reduce long when overbought
    rsi_dampener[rsi < RSI_OVERSOLD] = 0.3      # reduce short when oversold
    signal = signal * rsi_dampener

    # --- Funding rate contrarian signal ---
    if USE_FUNDING_SIGNAL and "funding_zscore" in df.columns:
        fz = df["funding_zscore"]
        # High positive funding = crowded long -> slight short bias
        # High negative funding = crowded short -> slight long bias
        funding_signal = pd.Series(0.0, index=df.index)
        funding_signal[fz > FUNDING_THRESHOLD] = -0.2   # contrarian
        funding_signal[fz < -FUNDING_THRESHOLD] = 0.2    # contrarian
        signal = signal + funding_signal

    # --- Volatility-adjusted position sizing ---
    if VOL_SCALING and f"volatility_{VOL_LOOKBACK}h" in df.columns:
        vol = df[f"volatility_{VOL_LOOKBACK}h"]
        ann_vol = vol * np.sqrt(8760)  # annualize from hourly
        vol_scalar = VOL_TARGET / ann_vol.replace(0, np.nan)
        vol_scalar = vol_scalar.clip(0.2, 3.0)  # bound the scaling
        signal = signal * vol_scalar

    # Apply base position size
    signal = signal * POSITION_SIZE

    # Clip to max position
    signal = signal.clip(-MAX_POSITION, MAX_POSITION)

    return signal


def generate_signals(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate trading signals for all assets.

    Args:
        features: dict mapping asset name -> DataFrame of features
                  (columns: open, high, low, close, volume, funding_rate,
                   plus all computed features from prepare.py)

    Returns:
        DataFrame indexed by timestamp, with one column per asset.
        Values in [-1.0, +1.0] representing desired position sizing.
        +1.0 = max long, -1.0 = max short, 0.0 = flat.
    """
    signals = {}
    for asset, df in features.items():
        signals[asset] = compute_signal(df)

    result = pd.DataFrame(signals)

    # --- Correlation filter (optional portfolio-level adjustment) ---
    if CORRELATION_FILTER and len(features) > 1:
        assets = list(features.keys())
        returns = pd.DataFrame({a: features[a]["close"].pct_change() for a in assets})
        rolling_corr = returns[assets[0]].rolling(CORRELATION_LOOKBACK).corr(returns[assets[1]])

        # When highly correlated, reduce combined exposure
        high_corr = rolling_corr.abs() > 0.8
        for asset in assets:
            result.loc[high_corr, asset] = result.loc[high_corr, asset] * 0.5

    return result


# ---------------------------------------------------------------------------
# Main: evaluate and print results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time
    from prepare import evaluate_strategy

    t0 = _time.time()
    results = evaluate_strategy(strategy_module=__import__(__name__), split="val")
    t1 = _time.time()

    print("---")
    print(f"composite_score:  {results['composite_score']:.6f}")
    print(f"sharpe_ratio:     {results['sharpe_ratio']:.4f}")
    print(f"sortino_ratio:    {results['sortino_ratio']:.4f}")
    print(f"total_return:     {results['total_return']:.4f}")
    print(f"annualized_return:{results['annualized_return']:.4f}")
    print(f"max_drawdown:     {results['max_drawdown']:.4f}")
    print(f"calmar_ratio:     {results['calmar_ratio']:.4f}")
    print(f"win_rate:         {results['win_rate']:.3f}")
    print(f"profit_factor:    {results['profit_factor']:.4f}")
    print(f"num_trades:       {results['num_trades']}")
    print(f"exposure_pct:     {results['exposure_pct']:.1f}")
    print(f"eval_seconds:     {t1 - t0:.1f}")
