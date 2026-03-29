"""
Autotrading strategy — BTC SPOT only, no leverage, no liquidation.
Signal: 0 = all USDC (flat), 1 = all BTC (max long).
No shorting. No leverage. No liquidation risk.

Usage: uv run strategy.py
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tunable Parameters (agent modifies these)
# ---------------------------------------------------------------------------

# Trend detection
FAST_MA = 12                    # fast MA (hours)
SLOW_MA = 42                    # slow MA (hours)
SPREAD_NORM_WINDOW = 72         # z-score normalization window

# Entry/exit thresholds (hysteresis to avoid whipsaw)
ENTRY_THRESHOLD = 0.5           # z-score to buy BTC (strong uptrend)
EXIT_THRESHOLD = -0.1           # z-score to sell BTC (trend weakening)

# Position sizing
BASE_POSITION = 0.7             # fraction of capital to deploy when signal is on
VOL_SCALING = True              # adjust size by volatility
VOL_LOOKBACK = 24               # hours
VOL_TARGET = 0.30               # annualized vol target (higher for spot, no liquidation)


# ---------------------------------------------------------------------------
# Strategy Logic
# ---------------------------------------------------------------------------

def compute_signal(df: pd.DataFrame) -> pd.Series:
    """
    BTC spot signal: 0 = flat (USDC), positive = long BTC.
    Uses hysteresis (different entry/exit thresholds) to minimize trades.
    """
    c = df["close"]

    # --- MA spread z-score ---
    fast_line = c.rolling(FAST_MA, min_periods=1).mean()
    slow_line = c.rolling(SLOW_MA, min_periods=1).mean()
    ma_spread = (fast_line - slow_line) / slow_line
    spread_std = ma_spread.rolling(SPREAD_NORM_WINDOW, min_periods=12).std().replace(0, np.nan)
    zscore = (ma_spread / spread_std).fillna(0)

    # --- Hysteresis: buy on strong uptrend, sell when trend fades ---
    position = pd.Series(0.0, index=df.index)
    in_position = False

    for i in range(len(df)):
        z = zscore.iloc[i]

        if not in_position:
            # Flat: buy only when strong uptrend confirmed
            if z > ENTRY_THRESHOLD:
                in_position = True
        else:
            # Long: exit when trend weakens
            if z < EXIT_THRESHOLD:
                in_position = False

        position.iloc[i] = 1.0 if in_position else 0.0

    # --- Volatility-adjusted sizing (at entry only) ---
    if VOL_SCALING and f"volatility_{VOL_LOOKBACK}h" in df.columns:
        vol = df[f"volatility_{VOL_LOOKBACK}h"]
        ann_vol = vol * np.sqrt(8760)
        vol_scalar = VOL_TARGET / ann_vol.replace(0, np.nan)
        vol_scalar = vol_scalar.clip(0.3, 1.5)

        # Freeze vol scaling at entry (don't change while holding)
        pos_changed = position != position.shift(1)
        entry_vol = vol_scalar.copy()
        entry_vol[~pos_changed] = np.nan
        entry_vol = entry_vol.ffill().fillna(1.0)
        signal = position * entry_vol * BASE_POSITION
    else:
        signal = position * BASE_POSITION

    # Cap at 1.0 (can't invest more than 100% in spot)
    signal = signal.clip(0.0, 1.0)

    return signal


def generate_signals(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate signals. Only BTC spot — ETH flat.
    Signal 0 = hold USDC, signal > 0 = hold BTC.
    """
    signals = {}

    if "BTC" in features:
        signals["BTC"] = compute_signal(features["BTC"])

    if "ETH" in features:
        signals["ETH"] = pd.Series(0.0, index=features["ETH"].index)

    return pd.DataFrame(signals)


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
