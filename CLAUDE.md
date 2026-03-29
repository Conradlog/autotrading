# CLAUDE.md

## Project overview

Autonomous trading strategy optimization framework for Bitcoin and commodities on HyperLiquid. An AI agent iterates on `strategy.py`, backtests via `prepare.py`, and keeps improvements.

## Key commands

```bash
# Download/refresh market data
uv run prepare.py

# Run strategy backtest (main evaluation)
uv run strategy.py

# Run and capture output
uv run strategy.py > run.log 2>&1

# Extract key metric
grep "^composite_score:" run.log

# Paper trading (manual, not part of loop)
uv run execute.py --once
```

## Architecture

- `prepare.py` — **DO NOT MODIFY.** Fixed infrastructure: HyperLiquid data download, 60+ features, vectorized backtester, `evaluate_strategy()`.
- `strategy.py` — **THE ONLY FILE TO EDIT.** Must expose `generate_signals(features) -> DataFrame`. Signals in [-1, +1].
- `program.md` — Full instructions for the autonomous optimization loop. Read this before starting.
- `execute.py` — Paper/live bridge. Not used during optimization.

## Optimization loop

See `program.md` for full details. Summary:

1. Edit `strategy.py` with an idea
2. `git commit`
3. `uv run strategy.py > run.log 2>&1`
4. `grep "^composite_score:" run.log`
5. If improved → keep. If worse → `git reset --hard HEAD~1`
6. Log to `results.tsv` (untracked)
7. Repeat forever

## Metric

```
composite_score = sharpe_ratio * (1 - max_drawdown) * sign(total_return)
```

Higher is better. Minimum 10 trades required.

## Constraints

- Only edit `strategy.py`
- No new pip dependencies
- No look-ahead bias in signals
- Signals must be in [-1, +1] range
- Paper mode only during optimization
- Do not commit `results.tsv`

## Data location

Cached at `~/.cache/autotrading/data/` as parquet files. Refresh with `uv run prepare.py`.
