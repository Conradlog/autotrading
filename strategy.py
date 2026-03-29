"""
Autotrading strategy. Single file, agent-modifiable.
Focus: BTC only, few high-conviction trades, minimize fees.

Usage: uv run strategy.py
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tunable Parameters (agent modifies these)
# ---------------------------------------------------------------------------

# Signal generation
FAST_MA = 12                    # fast MA for trend detection (hours)
SLOW_MA = 42                    # slow MA for trend detection (hours)
SPREAD_NORM_WINDOW = 72         # z-score normalization window
SIGNAL_THRESHOLD = 0.5          # only trade when z-score > this (high conviction)

# Position sizing
POSITION_SIZE = 0.5             # larger positions, fewer trades
VOL_SCALING = True              # scale by inverse volatility
VOL_LOOKBACK = 24               # hours
VOL_TARGET = 0.25               # annualized vol target
MAX_POSITION = 0.8              # max absolute position

# Trade frequency control
MIN_HOLD_HOURS = 6              # minimum hours between signal changes
ENTRY_THRESHOLD = 0.5           # z-score to enter (strong signal)
EXIT_THRESHOLD = 0.1            # z-score to exit (weak signal = close)


# ---------------------------------------------------------------------------
# Strategy Logic
# ---------------------------------------------------------------------------

def compute_signal(df: pd.DataFrame) -> pd.Series:
    """
    BTC-only signal: high conviction trend following.
    Trades only when the MA spread z-score is strong.
    Uses hysteresis (different entry/exit thresholds) to reduce whipsaw.
    """
    c = df["close"]

    # --- MA spread z-score ---
    fast_line = c.rolling(FAST_MA, min_periods=1).mean()
    slow_line = c.rolling(SLOW_MA, min_periods=1).mean()
    ma_spread = (fast_line - slow_line) / slow_line
    spread_std = ma_spread.rolling(SPREAD_NORM_WINDOW, min_periods=12).std().replace(0, np.nan)
    zscore = (ma_spread / spread_std).fillna(0)

    # --- Hysteresis-based signal: enter on strong, exit on weak ---
    # This drastically reduces trade count vs continuous signal
    position = pd.Series(0.0, index=df.index)
    current_pos = 0.0

    for i in range(len(df)):
        z = zscore.iloc[i]

        if current_pos == 0:
            # Flat: enter only on strong signal
            if z > ENTRY_THRESHOLD:
                current_pos = 1.0
            elif z < -ENTRY_THRESHOLD:
                current_pos = -1.0
        elif current_pos > 0:
            # Long: exit when signal weakens below exit threshold
            if z < EXIT_THRESHOLD:
                current_pos = 0.0
            # Or reverse on strong opposite signal
            if z < -ENTRY_THRESHOLD:
                current_pos = -1.0
        elif current_pos < 0:
            # Short: exit when signal weakens above -exit threshold
            if z > -EXIT_THRESHOLD:
                current_pos = 0.0
            # Or reverse on strong opposite signal
            if z > ENTRY_THRESHOLD:
                current_pos = 1.0

        position.iloc[i] = current_pos

    # --- Fixed size at entry, no continuous vol adjustment ---
    # Vol scaling only at point of entry (when position changes), not continuously
    if VOL_SCALING and f"volatility_{VOL_LOOKBACK}h" in df.columns:
        vol = df[f"volatility_{VOL_LOOKBACK}h"]
        ann_vol = vol * np.sqrt(8760)
        vol_scalar = VOL_TARGET / ann_vol.replace(0, np.nan)
        vol_scalar = vol_scalar.clip(0.3, 2.0)
        # Only apply vol scaling when position changes (entry/exit)
        pos_changed = position != position.shift(1)
        entry_vol = vol_scalar.copy()
        entry_vol[~pos_changed] = np.nan
        entry_vol = entry_vol.ffill().fillna(1.0)
        signal = position * entry_vol
    else:
        signal = position

    # Apply position size
    signal = signal * POSITION_SIZE

    # Clip
    signal = signal.clip(-MAX_POSITION, MAX_POSITION)

    return signal


def generate_signals(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate signals. Only trades BTC — ETH position always zero.
    """
    signals = {}

    # BTC: active trading
    if "BTC" in features:
        signals["BTC"] = compute_signal(features["BTC"])

    # ETH: flat (no trading to save fees)
    if "ETH" in features:
        signals["ETH"] = pd.Series(0.0, index=features["ETH"].index)

    result = pd.DataFrame(signals)

    # --- Drawdown circuit breaker ---
    if "BTC" in features:
        ret_96h = features["BTC"]["close"].pct_change(96)
        big_drop = ret_96h < -0.05
        result.loc[big_drop, "BTC"] = result.loc[big_drop, "BTC"] * 0.5

    # --- Volatility regime scaling ---
    if "BTC" in features:
        df = features["BTC"]
        if "volatility_168h" in df.columns and "volatility_720h" in df.columns:
            vol_7d = df["volatility_168h"]
            vol_30d = df["volatility_720h"]
            vol_ratio = vol_7d / vol_30d.replace(0, np.nan)
            regime_scale = (1.0 / vol_ratio.clip(0.5, 2.0)).fillna(1.0)
            result["BTC"] = result["BTC"] * regime_scale

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
