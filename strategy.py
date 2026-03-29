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

# MA crossover
FAST_MA = 10                    # fast moving average period (hours)
SLOW_MA = 42                    # slow moving average period (hours)
SPREAD_NORM_WINDOW = 72         # window for normalizing MA spread

# Position sizing
POSITION_SIZE = 0.3             # base position size (fraction of capital per signal)
VOL_SCALING = True              # scale position by inverse volatility
VOL_LOOKBACK = 24               # hours for volatility calculation
VOL_TARGET = 0.25               # annualized volatility target
MAX_POSITION = 0.5              # max absolute position per asset

# Multi-asset
CORRELATION_FILTER = True       # reduce position when assets are highly correlated
CORRELATION_LOOKBACK = 168      # hours
CORRELATION_THRESHOLD = 0.8     # correlation threshold to reduce exposure


# ---------------------------------------------------------------------------
# Strategy Logic
# ---------------------------------------------------------------------------

def compute_signal(df: pd.DataFrame) -> pd.Series:
    """
    Compute trading signal for a single asset.
    Returns a Series of values in [-1, +1].
    """
    c = df["close"]

    # --- Continuous MA spread signal ---
    fast_ma = c.rolling(FAST_MA, min_periods=1).mean()
    slow_ma = c.rolling(SLOW_MA, min_periods=1).mean()

    ma_spread = (fast_ma - slow_ma) / slow_ma
    spread_std = ma_spread.rolling(SPREAD_NORM_WINDOW, min_periods=12).std().replace(0, np.nan)
    signal = (ma_spread / spread_std).clip(-2, 2) / 2

    # --- Volatility-adjusted position sizing ---
    if VOL_SCALING and f"volatility_{VOL_LOOKBACK}h" in df.columns:
        vol = df[f"volatility_{VOL_LOOKBACK}h"]
        ann_vol = vol * np.sqrt(8760)
        vol_scalar = VOL_TARGET / ann_vol.replace(0, np.nan)
        vol_scalar = vol_scalar.clip(0.2, 3.0)
        signal = signal * vol_scalar

    # Apply base position size
    signal = signal * POSITION_SIZE

    # Dead zone: zero out very weak signals to avoid noise trades
    signal = signal.where(signal.abs() > 0.06, 0.0)

    # Clip to max position
    signal = signal.clip(-MAX_POSITION, MAX_POSITION)

    return signal


def generate_signals(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate trading signals for all assets.

    Args:
        features: dict mapping asset name -> DataFrame of features

    Returns:
        DataFrame indexed by timestamp, with one column per asset.
        Values in [-1.0, +1.0] representing desired position sizing.
    """
    signals = {}
    for asset, df in features.items():
        signals[asset] = compute_signal(df)

    result = pd.DataFrame(signals)

    # --- Drawdown circuit breaker ---
    for asset in features:
        ret_96h = features[asset]["close"].pct_change(96)
        big_drop = ret_96h < -0.05
        result.loc[big_drop, asset] = result.loc[big_drop, asset] * 0.5

    # --- Volatility regime scaling ---
    for asset in features:
        if "volatility_168h" in features[asset].columns:
            vol_7d = features[asset]["volatility_168h"]
            vol_30d = features[asset]["volatility_720h"] if "volatility_720h" in features[asset].columns else vol_7d
            vol_ratio = vol_7d / vol_30d.replace(0, np.nan)
            # High vol regime (vol_ratio > 1.5): reduce exposure
            # Low vol regime (vol_ratio < 0.7): boost exposure
            regime_scale = (1.0 / vol_ratio.clip(0.5, 2.0)).fillna(1.0)
            result[asset] = result[asset] * regime_scale

    # --- Correlation filter ---
    if CORRELATION_FILTER and len(features) > 1:
        assets = list(features.keys())
        returns = pd.DataFrame({a: features[a]["close"].pct_change() for a in assets})
        rolling_corr = returns[assets[0]].rolling(CORRELATION_LOOKBACK).corr(returns[assets[1]])

        high_corr = rolling_corr.abs() > CORRELATION_THRESHOLD
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
