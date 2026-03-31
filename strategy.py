"""
Autotrading strategy — BTC perpetual futures, 3x leverage.
Can go long, short, or flat. $500 capital.

Usage: uv run strategy.py
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tunable Parameters (agent modifies these)
# ---------------------------------------------------------------------------

# Trend detection
FAST_MA = 12                    # fast MA (hours)
SLOW_MA = 48                    # slow MA (hours)
SPREAD_NORM_WINDOW = 72         # z-score normalization window

# Entry/exit thresholds (hysteresis)
ENTRY_THRESHOLD = 0.6           # z-score to enter (strong signal)
EXIT_THRESHOLD = 0.08           # z-score to exit (weak signal)

# Position sizing
POSITION_SIZE = 0.60            # fraction of capital per signal
VOL_SCALING = True              # adjust size by volatility
VOL_LOOKBACK = 24               # hours
VOL_TARGET = 0.20               # annualized vol target
MAX_POSITION = 0.8              # max absolute position


# ---------------------------------------------------------------------------
# Strategy Logic
# ---------------------------------------------------------------------------

def compute_signal(df: pd.DataFrame) -> pd.Series:
    """
    BTC perp signal with hysteresis + MACD confirmation.
    +1 = long, -1 = short, 0 = flat.
    """
    c = df["close"]

    # --- MA spread z-score ---
    fast_line = c.rolling(FAST_MA, min_periods=1).mean()
    slow_line = c.rolling(SLOW_MA, min_periods=1).mean()
    ma_spread = (fast_line - slow_line) / slow_line
    spread_std = ma_spread.rolling(SPREAD_NORM_WINDOW, min_periods=12).std().replace(0, np.nan)
    zscore = (ma_spread / spread_std).fillna(0)

    # --- MACD confirmation ---
    macd_hist = df["macd_histogram"] if "macd_histogram" in df.columns else pd.Series(0.0, index=df.index)

    # --- Hysteresis: enter strong, exit weak ---
    position = pd.Series(0.0, index=df.index)
    current_pos = 0.0

    for i in range(len(df)):
        z = zscore.iloc[i]
        mh = macd_hist.iloc[i]

        if current_pos == 0:
            if z > ENTRY_THRESHOLD and mh > 0:
                current_pos = 1.0
            elif z < -ENTRY_THRESHOLD and mh < 0:
                current_pos = -1.0
        elif current_pos > 0:
            if z < -ENTRY_THRESHOLD and mh < 0:
                current_pos = -1.0       # reverse to short
            elif z < -EXIT_THRESHOLD:
                current_pos = 0.0        # exit long
        elif current_pos < 0:
            if z > ENTRY_THRESHOLD and mh > 0:
                current_pos = 1.0        # reverse to long
            elif z > EXIT_THRESHOLD:
                current_pos = 0.0        # exit short

        position.iloc[i] = current_pos

    # --- Vol sizing at entry only (use GK vol if available) ---
    gk_col = f"gk_volatility_{VOL_LOOKBACK}h"
    std_col = f"volatility_{VOL_LOOKBACK}h"
    vol_col = gk_col if gk_col in df.columns else std_col
    if VOL_SCALING and vol_col in df.columns:
        vol = df[vol_col]
        ann_vol = vol * np.sqrt(8760)
        vol_scalar = VOL_TARGET / ann_vol.replace(0, np.nan)
        vol_scalar = vol_scalar.clip(0.3, 2.0)
        pos_changed = position != position.shift(1)
        entry_vol = vol_scalar.copy()
        entry_vol[~pos_changed] = np.nan
        entry_vol = entry_vol.ffill().fillna(1.0)
        signal = position * entry_vol
    else:
        signal = position

    # --- Long-only: zero out all shorts ---
    signal = signal.clip(lower=0)

    signal = signal * POSITION_SIZE
    signal = signal.clip(-MAX_POSITION, MAX_POSITION)

    return signal


def generate_signals(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """BTC only. ETH flat."""
    signals = {}

    if "BTC" in features:
        signals["BTC"] = compute_signal(features["BTC"])

    if "ETH" in features:
        signals["ETH"] = pd.Series(0.0, index=features["ETH"].index)

    result = pd.DataFrame(signals)

    # --- Circuit breaker ---
    if "BTC" in features:
        ret_96h = features["BTC"]["close"].pct_change(96)
        big_drop = ret_96h < -0.05
        result.loc[big_drop, "BTC"] = 0.0

    # --- Volatility regime scaling ---
    if "BTC" in features:
        df = features["BTC"]
        if "gk_volatility_24h" in df.columns and "gk_volatility_168h" in df.columns:
            vol_7d = df["gk_volatility_24h"]
            vol_30d = df["gk_volatility_168h"]
            vol_ratio = vol_7d / vol_30d.replace(0, np.nan)
            regime_scale = (1.0 / vol_ratio.clip(0.5, 2.0)).fillna(1.0)
            result["BTC"] = result["BTC"] * regime_scale

    # Final clip to enforce [-1, +1] contract after all scaling
    result = result.clip(-1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time
    from prepare import evaluate_strategy

    t0 = _time.time()
    import sys
    ds = "extended" if "--extended" in sys.argv else "default"
    results = evaluate_strategy(strategy_module=__import__(__name__), split="val", dataset=ds)
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
