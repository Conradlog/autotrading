# autotrading

Autonomous trading strategy optimization on Bitcoin and commodities via [HyperLiquid](https://hyperliquid.xyz/).

Inspired by [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — an AI agent autonomously iterates on a trading strategy, backtests it, and keeps improvements.

## How it works

An AI agent (Claude Code) runs in a loop:

1. Modifies `strategy.py` with an experimental idea
2. Runs `uv run strategy.py` to backtest on 90 days of out-of-sample data
3. If `composite_score` improves → keeps the commit
4. If worse → git resets and tries something else
5. Repeats indefinitely

**All optimization runs in paper mode.** No real funds are ever at risk during the loop.

## Files

| File | Role | Editable? |
|---|---|---|
| `prepare.py` | Data pipeline, 60+ features, backtesting engine, evaluation harness | NO (fixed) |
| `strategy.py` | Trading strategy logic and parameters | YES (agent edits this) |
| `program.md` | Instructions for the autonomous agent | Human edits |
| `execute.py` | Paper/live trading bridge to HyperLiquid | Manual use only |
| `analysis.ipynb` | Experiment progress visualization | Analysis only |

## Quick start

```bash
# Install dependencies
uv sync

# Download market data from HyperLiquid (~5 min)
uv run prepare.py

# Run baseline strategy evaluation
uv run strategy.py

# Start autonomous optimization (in Claude Code)
# Just say: "read program.md and start the loop"
```

## Evaluation metric

```
composite_score = sharpe_ratio x (1 - max_drawdown) x sign(total_return)
```

Rewards high risk-adjusted returns while penalizing drawdowns. Strategies with < 10 trades score 0.

## Data

- Source: HyperLiquid API (free, no API key needed)
- Assets: BTC, ETH (perpetual futures)
- Resolution: 1-hour candles
- Lookback: 365 days
- Split: first 275 days (train) / last 90 days (validation, out-of-sample)

## Paper trading

```bash
# Run paper trading with the current best strategy
uv run execute.py

# Single evaluation
uv run execute.py --once
```

Live trading requires explicit confirmation and `HL_PRIVATE_KEY` environment variable.

## Results

Experiments are logged in `results.tsv` (untracked). Use `analysis.ipynb` to visualize progress.
