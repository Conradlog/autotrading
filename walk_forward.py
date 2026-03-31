"""
Walk-Forward Optimization for strategy validation.

Splits extended dataset into rolling train/validation windows,
runs the strategy on each validation window, reports combined metrics.

Usage:
    uv run walk_forward.py
    uv run walk_forward.py --windows 5 --train-months 18 --val-months 6
    uv run walk_forward.py --dataset extended
"""

import argparse
import time as _time

import numpy as np
import pandas as pd

from prepare import (
    load_market_data, compute_base_features, BacktestEngine,
    INITIAL_CAPITAL, MAX_LEVERAGE, COMMISSION_BPS, SLIPPAGE_BPS,
)


# ---------------------------------------------------------------------------
# Walk-Forward Engine
# ---------------------------------------------------------------------------

def run_walk_forward(
    strategy_module,
    n_windows: int = 5,
    train_months: int = 18,
    val_months: int = 6,
    slide_months: int = 6,
    dataset: str = "extended",
):
    """
    Run walk-forward validation across rolling windows.

    Each window uses `train_months` of data for feature warm-up
    and `val_months` for out-of-sample evaluation. Windows slide
    by `slide_months`.
    """
    print("=" * 70)
    print("WALK-FORWARD OPTIMIZATION")
    print("=" * 70)

    # 1. Load full extended dataset
    print(f"\nLoading dataset '{dataset}'...")
    df = load_market_data("BTC", dataset=dataset)
    df_feat = compute_base_features(df)
    df_feat = df_feat.set_index("timestamp")

    total_days = len(df_feat) / 24
    print(f"  {len(df_feat)} candles ({total_days:.0f} days)")
    print(f"  Range: {df_feat.index.min()} → {df_feat.index.max()}")

    # 2. Determine window boundaries
    data_start = df_feat.index.min()
    data_end = df_feat.index.max()
    windows = []

    for i in range(n_windows):
        train_start = data_start + pd.DateOffset(months=slide_months * i)
        train_end = train_start + pd.DateOffset(months=train_months)
        val_start = train_end
        val_end = val_start + pd.DateOffset(months=val_months)

        # Check if window fits in data
        if val_end > data_end + pd.Timedelta(days=7):  # 7 day tolerance
            print(f"\n  Window {i+1} extends beyond data ({val_end.date()} > {data_end.date()}), stopping at {n_windows - (n_windows - i)} windows.")
            break

        windows.append({
            "id": i + 1,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": min(val_end, data_end),
        })

    if not windows:
        print("ERROR: Not enough data for any walk-forward window.")
        return None

    print(f"\n  Windows: {len(windows)}")
    for w in windows:
        vs = w["val_start"].strftime("%Y-%m")
        ve = w["val_end"].strftime("%Y-%m")
        ts = w["train_start"].strftime("%Y-%m")
        te = w["train_end"].strftime("%Y-%m")
        print(f"    Window {w['id']}: Train [{ts} → {te}] → Val [{vs} → {ve}]")

    # 3. Run strategy on each validation window
    print("\n" + "-" * 70)
    print("Running backtests...")
    print("-" * 70)

    window_results = []

    for w in windows:
        t0 = _time.time()

        # Slice validation data (features already computed on full dataset for proper warm-up)
        val_data = df_feat.loc[w["val_start"]:w["val_end"]].copy()

        if len(val_data) < 24:
            print(f"  Window {w['id']}: Skipped (only {len(val_data)} candles)")
            continue

        # Generate signals
        features = {"BTC": val_data}
        signals_df = strategy_module.generate_signals(features)

        # Prepare backtest inputs
        prices_df = pd.DataFrame({"BTC": val_data["close"]})
        funding_df = pd.DataFrame({"BTC": val_data["funding_rate"]})

        # Align indices
        common_idx = prices_df.index.intersection(signals_df.index)
        prices_df = prices_df.loc[common_idx]
        signals_df = signals_df.loc[common_idx]
        funding_df = funding_df.loc[common_idx]

        # Run backtest
        engine = BacktestEngine()
        result = engine.run(prices_df, signals_df, funding_df)

        t1 = _time.time()

        # BTC price range for context
        btc_start = val_data["close"].iloc[0]
        btc_end = val_data["close"].iloc[-1]
        btc_return = (btc_end - btc_start) / btc_start

        window_results.append({
            "window": w["id"],
            "val_start": w["val_start"],
            "val_end": w["val_end"],
            "result": result,
            "btc_return": btc_return,
            "btc_start": btc_start,
            "btc_end": btc_end,
            "n_candles": len(val_data),
            "elapsed": t1 - t0,
        })

        # Compute composite score
        sr = result.sharpe_ratio
        dd = result.max_drawdown
        tr = result.total_return
        if result.trades is not None and len(result.trades) >= 10 and tr > 0:
            score = sr * (1 - dd)
        elif tr <= 0:
            score = -abs(sr) * (1 + dd)
        else:
            score = 0.0

        status = "+" if tr > 0 else "-"
        print(f"  Window {w['id']} [{w['val_start'].strftime('%Y-%m')} → {w['val_end'].strftime('%Y-%m')}]  "
              f"Score={score:>7.3f}  Sharpe={sr:>6.2f}  Return={tr:>+7.1%}  "
              f"MaxDD={dd:>5.1%}  Trades={len(result.trades) if result.trades is not None else 0:>5}  "
              f"BTC={btc_return:>+6.1%}  ({t1-t0:.1f}s)")

    if not window_results:
        print("ERROR: No windows completed.")
        return None

    # 4. Compute combined metrics
    print("\n" + "=" * 70)
    print("COMBINED RESULTS")
    print("=" * 70)

    # Chain equity curves
    all_returns = []
    chained_equity = [INITIAL_CAPITAL]
    total_trades = 0

    for wr in window_results:
        r = wr["result"]
        eq = r.equity_curve
        # Hourly returns from equity curve
        hourly_ret = eq.pct_change().dropna()
        all_returns.extend(hourly_ret.values)
        # Chain equity
        if len(eq) > 0:
            window_return = r.total_return
            chained_equity.append(chained_equity[-1] * (1 + window_return))
        if r.trades is not None:
            total_trades += len(r.trades)

    all_returns = np.array(all_returns)
    all_returns = all_returns[~np.isnan(all_returns)]

    # Combined Sharpe (annualized from hourly returns)
    if len(all_returns) > 0 and all_returns.std() > 0:
        combined_sharpe = (all_returns.mean() / all_returns.std()) * np.sqrt(8760)
    else:
        combined_sharpe = 0.0

    # Combined return (product of 1+r across windows)
    combined_return = 1.0
    for wr in window_results:
        combined_return *= (1 + wr["result"].total_return)
    combined_return -= 1.0

    # Combined max drawdown from chained equity
    eq_series = pd.Series(chained_equity)
    rolling_max = eq_series.cummax()
    drawdowns = (eq_series - rolling_max) / rolling_max
    combined_max_dd = abs(drawdowns.min())

    # Also compute max DD from hourly returns
    cum_returns = (1 + pd.Series(all_returns)).cumprod()
    rolling_max_hr = cum_returns.cummax()
    dd_hr = (cum_returns - rolling_max_hr) / rolling_max_hr
    combined_max_dd_hr = abs(dd_hr.min())

    # Combined composite score
    if combined_return > 0:
        combined_score = combined_sharpe * (1 - combined_max_dd_hr)
    else:
        combined_score = -abs(combined_sharpe) * (1 + combined_max_dd_hr)

    # Per-window stats
    window_sharpes = [wr["result"].sharpe_ratio for wr in window_results]
    window_returns = [wr["result"].total_return for wr in window_results]
    profitable_windows = sum(1 for r in window_returns if r > 0)

    # BTC buy-and-hold comparison
    btc_total = 1.0
    for wr in window_results:
        btc_total *= (1 + wr["btc_return"])
    btc_total -= 1.0

    print(f"\n  {'Metric':<28} {'Value':>12}")
    print(f"  {'-'*28} {'-'*12}")
    print(f"  {'Composite Score':<28} {combined_score:>12.3f}")
    print(f"  {'Sharpe Ratio (combined)':<28} {combined_sharpe:>12.2f}")
    print(f"  {'Total Return (chained)':<28} {combined_return:>+11.1%}")
    print(f"  {'Max Drawdown (hourly)':<28} {combined_max_dd_hr:>11.1%}")
    print(f"  {'Total Trades':<28} {total_trades:>12}")
    print(f"  {'Profitable Windows':<28} {profitable_windows}/{len(window_results)}")
    print(f"  {'Sharpe Range':<28} {min(window_sharpes):>+5.2f} to {max(window_sharpes):>+5.2f}")
    print(f"  {'Sharpe Std (consistency)':<28} {np.std(window_sharpes):>12.2f}")
    print(f"  {'Validation Period':<28} {window_results[0]['val_start'].strftime('%Y-%m')} → {window_results[-1]['val_end'].strftime('%Y-%m')}")

    print(f"\n  {'BTC Buy & Hold (same period)':<28} {btc_total:>+11.1%}")
    alpha = combined_return - btc_total
    print(f"  {'Strategy Alpha vs B&H':<28} {alpha:>+11.1%}")

    # Detailed per-window table
    print(f"\n  {'Window':<10} {'Period':<18} {'Score':>8} {'Sharpe':>8} {'Return':>8} {'MaxDD':>7} {'Trades':>7} {'BTC':>8}")
    print(f"  {'-'*10} {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*8}")
    for wr in window_results:
        r = wr["result"]
        sr = r.sharpe_ratio
        dd = r.max_drawdown
        tr = r.total_return
        n_trades = len(r.trades) if r.trades is not None else 0
        if n_trades >= 10 and tr > 0:
            score = sr * (1 - dd)
        elif tr <= 0:
            score = -abs(sr) * (1 + dd)
        else:
            score = 0.0
        period = f"{wr['val_start'].strftime('%Y-%m')} → {wr['val_end'].strftime('%Y-%m')}"
        print(f"  {wr['window']:<10} {period:<18} {score:>+8.3f} {sr:>+8.2f} {tr:>+7.1%} {dd:>6.1%} {n_trades:>7} {wr['btc_return']:>+7.1%}")

    print(f"\n{'='*70}")

    # Robustness assessment
    print("\nROBUSTNESS ASSESSMENT:")
    if profitable_windows == len(window_results):
        print("  [OK] All windows profitable")
    elif profitable_windows >= len(window_results) * 0.6:
        print(f"  [WARN] {profitable_windows}/{len(window_results)} windows profitable (>60%)")
    else:
        print(f"  [FAIL] Only {profitable_windows}/{len(window_results)} windows profitable")

    sharpe_std = np.std(window_sharpes)
    if sharpe_std < 1.0:
        print(f"  [OK] Sharpe consistency good (std={sharpe_std:.2f})")
    elif sharpe_std < 2.0:
        print(f"  [WARN] Sharpe consistency moderate (std={sharpe_std:.2f})")
    else:
        print(f"  [FAIL] Sharpe inconsistent across windows (std={sharpe_std:.2f})")

    if combined_return > btc_total:
        print(f"  [OK] Outperforms buy & hold by {alpha:+.1%}")
    else:
        print(f"  [FAIL] Underperforms buy & hold by {alpha:.1%}")

    if combined_max_dd_hr < 0.15:
        print(f"  [OK] Max drawdown controlled ({combined_max_dd_hr:.1%})")
    elif combined_max_dd_hr < 0.25:
        print(f"  [WARN] Max drawdown moderate ({combined_max_dd_hr:.1%})")
    else:
        print(f"  [FAIL] Max drawdown high ({combined_max_dd_hr:.1%})")

    return {
        "combined_score": combined_score,
        "combined_sharpe": combined_sharpe,
        "combined_return": combined_return,
        "combined_max_dd": combined_max_dd_hr,
        "total_trades": total_trades,
        "profitable_windows": profitable_windows,
        "n_windows": len(window_results),
        "window_results": window_results,
        "btc_buyhold": btc_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import strategy

    parser = argparse.ArgumentParser(description="Walk-forward strategy validation")
    parser.add_argument("--windows", type=int, default=5, help="Number of windows")
    parser.add_argument("--train-months", type=int, default=12, help="Training months per window")
    parser.add_argument("--val-months", type=int, default=6, help="Validation months per window")
    parser.add_argument("--slide-months", type=int, default=6, help="Slide between windows")
    parser.add_argument("--dataset", default="extended", help="Dataset name (default: extended)")
    args = parser.parse_args()

    result = run_walk_forward(
        strategy,
        n_windows=args.windows,
        train_months=args.train_months,
        val_months=args.val_months,
        slide_months=args.slide_months,
        dataset=args.dataset,
    )
