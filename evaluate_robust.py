"""
Robust evaluation: run walk-forward on extended dataset and print composite score.
Used as drop-in replacement for 'uv run strategy.py' during optimization.

Usage:
    uv run evaluate_robust.py
"""

import sys
import time as _time
import numpy as np
import pandas as pd

from prepare import (
    load_market_data, compute_base_features, BacktestEngine,
    INITIAL_CAPITAL, EVAL_DAYS,
)
import strategy


def evaluate_robust():
    """Run strategy on multiple time windows and return robust metrics."""
    # Load extended dataset
    df = load_market_data("BTC", dataset="extended")
    df_feat = compute_base_features(df)
    df_feat = df_feat.set_index("timestamp")

    data_start = df_feat.index.min()
    data_end = df_feat.index.max()

    # Define validation windows (6 months each, sliding by 6 months)
    # Start validation from 2023-01 to have warm-up period
    windows = [
        ("2023-01-01", "2023-07-01"),
        ("2023-07-01", "2024-01-01"),
        ("2024-01-01", "2024-07-01"),
        ("2024-07-01", "2025-01-01"),
        ("2025-01-01", "2026-03-30"),  # last window to present
    ]

    all_sharpes = []
    all_returns = []
    all_scores = []
    all_dds = []
    total_trades = 0

    for vs, ve in windows:
        vs_ts = pd.Timestamp(vs, tz="UTC")
        ve_ts = pd.Timestamp(ve, tz="UTC")

        # Clip to available data
        if vs_ts > data_end:
            continue
        ve_ts = min(ve_ts, data_end)

        val_data = df_feat.loc[vs_ts:ve_ts].copy()
        if len(val_data) < 48:
            continue

        # Generate signals
        features = {"BTC": val_data}
        signals_df = strategy.generate_signals(features)

        prices_df = pd.DataFrame({"BTC": val_data["close"]})
        funding_df = pd.DataFrame({"BTC": val_data["funding_rate"]})

        common_idx = prices_df.index.intersection(signals_df.index)
        prices_df = prices_df.loc[common_idx]
        signals_df = signals_df.loc[common_idx]
        funding_df = funding_df.loc[common_idx]

        engine = BacktestEngine()
        result = engine.run(prices_df, signals_df, funding_df)

        sr = result.sharpe_ratio
        dd = result.max_drawdown
        tr = result.total_return
        n_trades = len(result.trades) if result.trades is not None else 0

        if n_trades >= 10 and tr > 0:
            score = sr * (1 - dd)
        elif tr <= 0:
            score = -abs(sr) * (1 + dd)
        else:
            score = 0.0

        all_sharpes.append(sr)
        all_returns.append(tr)
        all_scores.append(score)
        all_dds.append(dd)
        total_trades += n_trades

    # Robust composite: average of per-window scores, penalized by inconsistency
    avg_score = np.mean(all_scores)
    avg_sharpe = np.mean(all_sharpes)
    avg_return = np.mean(all_returns)
    avg_dd = np.mean(all_dds)
    profitable_pct = sum(1 for r in all_returns if r > 0) / len(all_returns)

    # Penalize inconsistency: multiply by fraction of profitable windows
    robust_score = avg_score * profitable_pct

    return {
        "composite_score": robust_score,
        "avg_score": avg_score,
        "sharpe_ratio": avg_sharpe,
        "total_return": avg_return,
        "max_drawdown": avg_dd,
        "num_trades": total_trades,
        "profitable_windows": f"{sum(1 for r in all_returns if r > 0)}/{len(all_returns)}",
        "window_scores": all_scores,
    }


if __name__ == "__main__":
    t0 = _time.time()
    results = evaluate_robust()
    t1 = _time.time()

    print("---")
    print(f"composite_score:  {results['composite_score']:.6f}")
    print(f"avg_window_score: {results['avg_score']:.6f}")
    print(f"sharpe_ratio:     {results['sharpe_ratio']:.4f}")
    print(f"total_return:     {results['total_return']:.4f}")
    print(f"max_drawdown:     {results['max_drawdown']:.4f}")
    print(f"num_trades:       {results['num_trades']}")
    print(f"profitable:       {results['profitable_windows']}")
    print(f"window_scores:    {['%.2f' % s for s in results['window_scores']]}")
    print(f"eval_seconds:     {t1 - t0:.1f}")
